"""Shared CARPL helpers: validation, fingerprints, job state, broker semantics."""

import tempfile
import unittest
from pathlib import Path

import carpl_test_utils  # noqa: F401  (sets up sys.path)
import carpl_common as cc


class ValidationTests(unittest.TestCase):

    def test_ae_titles(self):
        self.assertEqual(cc.validate_ae_title(" PROSTATE_AI "), "PROSTATE_AI")
        self.assertEqual(cc.validate_ae_title("CARPL-RECIEVE"), "CARPL-RECIEVE")
        for bad in ("", "   ", "X" * 17, "BAD\\AE", "tab\tAE"):
            with self.assertRaises(ValueError, msg=bad):
                cc.validate_ae_title(bad)

    def test_uids_cannot_escape_the_data_folder(self):
        self.assertEqual(cc.validate_uid("1.2.840.10008.5.1.4.1.1.4"), "1.2.840.10008.5.1.4.1.1.4")
        for bad in ("", "../../etc", "1.2.a", "1..2", "1.2.", "1" * 65, None):
            with self.assertRaises(ValueError, msg=bad):
                cc.validate_uid(bad)

    def test_fingerprint_ignores_order_and_duplicates(self):
        self.assertEqual(
            cc.study_fingerprint(["1.2.3", "1.2.4", "1.2.3"]),
            cc.study_fingerprint(["1.2.4", "1.2.3"]),
        )
        self.assertNotEqual(cc.study_fingerprint(["1.2.3"]), cc.study_fingerprint(["1.2.3", "1.2.5"]))

    def test_job_ids_are_unique_and_sortable(self):
        ids = {cc.new_job_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)
        self.assertRegex(next(iter(ids)), r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")


class JobStateTests(unittest.TestCase):

    def test_state_history_is_appended_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = cc.JobPaths(Path(tmp) / "job")
            cc.update_job_state(job, "queued", fingerprint="abc")
            cc.update_job_state(job, "running", attempts=1)
            state = cc.read_json(job.state_json)
            self.assertEqual(state["state"], "running")
            self.assertEqual(state["fingerprint"], "abc")
            self.assertEqual([h["state"] for h in state["history"]], ["queued", "running"])
            self.assertEqual(cc.job_state(job), "running")
            self.assertEqual(list(Path(tmp, "job").glob(".state.json.*")), [])


class InMemoryBrokerTests(unittest.TestCase):

    def test_reliable_queue_semantics(self):
        broker = cc.InMemoryBroker()
        broker.push("q", {"job_id": "a"})
        broker.push("q", {"job_id": "b"})
        raw, payload = broker.reserve("q", "q:processing", timeout=0.1)
        self.assertEqual(payload["job_id"], "a")
        self.assertEqual(broker.length("q:processing"), 1)
        # Crash before ack: the item returns to the head of the queue.
        self.assertEqual(broker.requeue_processing("q:processing", "q"), 1)
        self.assertEqual([p["job_id"] for p in broker.items("q")], ["a", "b"])
        raw, payload = broker.reserve("q", "q:processing", timeout=0.1)
        broker.ack("q:processing", raw)
        self.assertEqual(broker.length("q:processing"), 0)
        self.assertIsNone(broker.reserve("empty", "empty:processing", timeout=0.05))

    def test_set_nx(self):
        broker = cc.InMemoryBroker()
        self.assertTrue(broker.set_nx("k", "1", 10))
        self.assertFalse(broker.set_nx("k", "2", 10))
        self.assertEqual(broker.get("k"), "1")


if __name__ == "__main__":
    unittest.main()
