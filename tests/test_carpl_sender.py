"""CARPL dicom-sender: contexts, retries, dead-letter, idempotency, bytes on the wire."""

import json
import logging
import socket
import tempfile
import unittest
from pathlib import Path

import pydicom
from pydicom.uid import generate_uid

import carpl_test_utils as utils
import carpl_common as cc

sender_mod = utils.load_module("carpl_sender", "carpl/dicom-sender/sender.py")


class StubSender(sender_mod.Sender):
    def __init__(self, *args, outcomes, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = list(outcomes)
        self.calls = 0

    def send_files(self, files):
        self.calls += 1
        ok = self.outcomes.pop(0)
        return ok, [{"file": f.name, "status": 0 if ok else 0xA700} for f in files]


class SenderTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.paths = cc.DataPaths(Path(self._tmp.name))
        self.paths.ensure()
        self.broker = cc.InMemoryBroker()
        self.config = sender_mod.SenderConfig(max_attempts=3, backoff_base=0, backoff_max=0)
        self.study = generate_uid()
        self.job_id = cc.new_job_id()
        self.job = self.paths.job(self.job_id)
        utils.write_combined_rtstruct(self.job.output / cc.RESULT_FILES[0], self.study, "P1")
        utils.write_sr(self.job.output / cc.RESULT_FILES[1], self.study, "P1")
        cc.update_job_state(self.job, "send_queued")
        self.payload = {
            "job_id": self.job_id, "study_uid": self.study, "fingerprint": "f" * 64,
            "files": [str(self.job.output / name) for name in cc.RESULT_FILES],
        }

    def tearDown(self):
        self._tmp.cleanup()

    def sender(self, outcomes):
        return StubSender(self.config, self.paths, self.broker, logging.getLogger("test"), outcomes=outcomes)

    def test_contexts_use_each_files_own_transfer_syntax(self):
        files = [Path(f) for f in self.payload["files"]]
        contexts = sender_mod.file_contexts(files, allow_transcode=False)
        self.assertEqual(contexts, {
            cc.RTSTRUCT_SOP_CLASS: [cc.EXPLICIT_VR_LITTLE_ENDIAN],
            cc.COMPREHENSIVE_SR_SOP_CLASS: [cc.EXPLICIT_VR_LITTLE_ENDIAN],
        })
        widened = sender_mod.file_contexts(files, allow_transcode=True)
        self.assertEqual(widened[cc.RTSTRUCT_SOP_CLASS], [cc.EXPLICIT_VR_LITTLE_ENDIAN, cc.IMPLICIT_VR_LITTLE_ENDIAN])

    def test_delivery_after_retry(self):
        sender = self.sender([False, True])
        self.assertTrue(sender.process(self.payload))
        self.assertEqual(sender.calls, 2)
        state = cc.read_json(self.job.state_json)
        self.assertEqual(state["state"], "delivered")
        self.assertEqual(state["destination"], "CARPL-RECIEVE@rapier-orthanc-prefect:4242")
        fp = json.loads(self.broker.get(cc.fingerprint_key(self.payload["fingerprint"])))
        self.assertEqual(fp["state"], "delivered")

    def test_dead_letter_after_max_attempts(self):
        sender = self.sender([False, False, False])
        self.assertTrue(sender.process(self.payload))
        self.assertEqual(sender.calls, 3)
        self.assertEqual(cc.job_state(self.job), "send_failed")
        self.assertEqual(self.broker.items(cc.SEND_FAILED)[0]["job_id"], self.job_id)

    def test_already_delivered_is_not_sent_twice(self):
        cc.update_job_state(self.job, "delivered")
        sender = self.sender([])
        self.assertTrue(sender.process(self.payload))
        self.assertEqual(sender.calls, 0)

    def test_refuses_files_outside_the_job_output(self):
        outside = Path(self._tmp.name) / "elsewhere.dcm"
        utils.write_sr(outside, self.study, "P1")
        sender = self.sender([])
        self.assertTrue(sender.process(dict(self.payload, files=[str(outside)])))
        self.assertEqual(sender.calls, 0)
        self.assertEqual(cc.job_state(self.job), "send_failed")

    def test_missing_result_file_is_dead_lettered(self):
        (self.job.output / cc.RESULT_FILES[1]).unlink()
        sender = self.sender([])
        self.assertTrue(sender.process(self.payload))
        self.assertIn("missing", self.broker.items(cc.SEND_FAILED)[0]["reason"])


@unittest.skipUnless(utils.has_module("pynetdicom"), "pynetdicom not installed")
class SenderNetworkTests(unittest.TestCase):
    """Real association against a stub SCP: bytes must arrive unmodified."""

    def test_files_arrive_byte_identical(self):
        from pynetdicom import AE, AllStoragePresentationContexts, evt

        parity = utils.load_module("carpl_parity", "carpl/tools/parity_check.py")
        received = {}

        def handle_store(event):
            received[str(event.dataset.SOPInstanceUID)] = event.encoded_dataset()
            return 0x0000

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        scp = AE(ae_title="CARPL-RECIEVE")
        for context in AllStoragePresentationContexts:
            scp.add_supported_context(context.abstract_syntax, [cc.EXPLICIT_VR_LITTLE_ENDIAN])
        server = scp.start_server(("127.0.0.1", port), block=False, evt_handlers=[(evt.EVT_C_STORE, handle_store)])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                study = generate_uid()
                files = [
                    utils.write_combined_rtstruct(Path(tmp) / "rt.dcm", study, "P1"),
                    utils.write_sr(Path(tmp) / "sr.dcm", study, "P1"),
                ]
                config = sender_mod.SenderConfig(dest_host="127.0.0.1", dest_port=port)
                sender = sender_mod.Sender(config, cc.DataPaths(Path(tmp)), cc.InMemoryBroker(), logging.getLogger("t"))
                ok, results = sender.send_files(files)
                self.assertTrue(ok, results)
                self.assertEqual(len(received), 2)
                for path in files:
                    sop = str(pydicom.dcmread(path, stop_before_pixels=True).SOPInstanceUID)
                    stored = Path(tmp) / f"received-{sop}.dcm"
                    stored.write_bytes(received[sop])
                    self.assertEqual(parity.dataset_bytes(path), parity.dataset_bytes(stored))

                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    closed_port = sock.getsockname()[1]
                unreachable = sender_mod.Sender(
                    sender_mod.SenderConfig(dest_host="127.0.0.1", dest_port=closed_port),
                    cc.DataPaths(Path(tmp)), cc.InMemoryBroker(), logging.getLogger("t"),
                )
                ok, results = unreachable.send_files(files)
                self.assertFalse(ok)
                self.assertIn("rejected or aborted", results[0]["error"])
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
