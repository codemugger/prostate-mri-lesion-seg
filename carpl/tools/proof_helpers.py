"""Helpers for ``carpl/CARPL_end_to_end_proof.ipynb``.

The notebook demonstrates, on real studies, that the CARPL DICOM Push stack
returns exactly what the standard pipeline produces.  Everything heavy runs
inside the project's Docker images, so the notebook kernel only needs
pydicom, numpy, pandas and matplotlib.

Every run uses its own folder under ``carpl/proof_output/`` (git-ignored),
so existing data is never touched.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

REPO = Path(__file__).resolve().parents[2]
CARPL = REPO / "carpl"
COMPOSE_FILE = CARPL / "docker-compose.yml"
NETWORK = "prostate-mri-carpl-dev_default"
LISTENER_HOST = "prostate-dev-dicom-listener"
UID_GID = f"{os.getuid()}:{os.getgid()}"


# --------------------------------------------------------------------- #
# Shell helpers
# --------------------------------------------------------------------- #
def _run(cmd: "list[str]", *, env: "dict | None" = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=check,
        env=dict(os.environ, **(env or {})),
    )


def _compose_env(version: str, data_dir: Path) -> dict:
    return {
        "VERSION": version,
        "CARPL_DATA_DIR": str(data_dir),
        "PUID": str(os.getuid()),
        "PGID": str(os.getgid()),
        "STUDY_QUIET_SECONDS": "20",
    }


def new_run_folder() -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = CARPL / "proof_output" / stamp
    (folder / "data" / "redis").mkdir(parents=True)
    (folder / "direct").mkdir()
    return folder


# --------------------------------------------------------------------- #
# Environment and stack control
# --------------------------------------------------------------------- #
def check_prerequisites(version: str) -> pd.DataFrame:
    rows = []

    def add(check: str, ok: bool, detail: str) -> None:
        rows.append({"check": check, "result": "PASS" if ok else "FAIL", "detail": detail})

    docker = _run(["docker", "version", "--format", "{{.Server.Version}}"], check=False)
    add("Docker engine reachable", docker.returncode == 0, docker.stdout.strip() or docker.stderr.strip()[:120])
    for image in (
        f"prostate-mri-seg:{version}",
        f"prostate-carpl-ai-worker:{version}",
        f"prostate-carpl-dicom-listener:{version}",
        f"prostate-carpl-dicom-sender:{version}",
        f"prostate-carpl-receiver:{version}",
        "redis:7.4-alpine",
    ):
        found = _run(["docker", "image", "inspect", "-f", "{{.Id}}", image], check=False)
        add(f"Image {image}", found.returncode == 0, found.stdout.strip()[:19])
    gpu = _run(
        ["docker", "run", "--rm", "--gpus", "all", "--entrypoint", "python", f"prostate-carpl-ai-worker:{version}",
         "-c", "import torch; p=torch.cuda.get_device_properties(0); print(f'{p.name}, {p.total_memory/1024**3:.1f} GB')"],
        check=False,
    )
    add("NVIDIA GPU visible in containers", gpu.returncode == 0, gpu.stdout.strip() or gpu.stderr.strip()[-120:])
    models = [p.relative_to(REPO / "models").as_posix() for p in sorted((REPO / "models").rglob("*")) if p.is_file()]
    add("7 model files in models/", len(models) == 7, ", ".join(models))
    return pd.DataFrame(rows)


def start_stack(version: str, data_dir: Path, timeout: float = 300) -> pd.DataFrame:
    env = _compose_env(version, data_dir)
    _run(["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--no-build"], env=env)
    deadline = time.time() + timeout
    while time.time() < deadline:
        logs = _run(["docker", "logs", "prostate-dev-ai-worker"], check=False)
        text = logs.stdout + logs.stderr
        if "Preflight passed" in text:
            break
        if "Refusing to process" in text:
            raise RuntimeError("Worker preflight failed:\n" + text[-2000:])
        time.sleep(3)
    else:
        raise TimeoutError("Worker did not pass preflight in time")
    while time.time() < deadline:
        status = container_status()
        if all("healthy" in s or "Up" in s for s in status["status"]) and not any("starting" in s for s in status["status"]):
            return status
        time.sleep(3)
    return container_status()


def stop_stack(version: str, data_dir: Path) -> str:
    result = _run(["docker", "compose", "-f", str(COMPOSE_FILE), "down"], env=_compose_env(version, data_dir), check=False)
    return (result.stdout + result.stderr).strip().splitlines()[-1] if (result.stdout + result.stderr).strip() else ""


def container_status() -> pd.DataFrame:
    out = _run(["docker", "ps", "--filter", "name=prostate-", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"]).stdout
    rows = [dict(zip(("container", "image", "status"), line.split("\t"))) for line in out.strip().splitlines()]
    return pd.DataFrame(rows).sort_values("container").reset_index(drop=True)


def worker_preflight_line() -> str:
    logs = _run(["docker", "logs", "prostate-dev-ai-worker"], check=False)
    for line in (logs.stdout + logs.stderr).splitlines():
        if "GPU probe" in line:
            return line.split("GPU probe: ", 1)[-1]
    return ""


# --------------------------------------------------------------------- #
# Pushing studies (the role CARPL plays)
# --------------------------------------------------------------------- #
def push_study(case_dir: Path, version: str, include: "str | None" = None, exclude: "str | None" = None) -> dict:
    """C-ECHO + C-STORE a study folder to the listener, exactly as CARPL would."""
    cmd = [
        "docker", "run", "--rm", "--network", NETWORK, "-u", UID_GID,
        "-v", f"{Path(case_dir).resolve()}:/study:ro", "-v", f"{CARPL / 'tools'}:/tools:ro",
        "--entrypoint", "python", f"prostate-carpl-dicom-listener:{version}",
        "/tools/send_study.py", "/study", "--host", LISTENER_HOST, "--port", "11112",
        "--aec", "PROSTATE_AI", "--aet", "CARPL_SIM",
    ]
    if include:
        cmd += ["--include-series", include]
    if exclude:
        cmd += ["--exclude-series", exclude]
    started = time.time()
    result = _run(cmd, check=False)
    lines = result.stdout.strip().splitlines()
    return {
        "study": Path(case_dir).name,
        "pushed_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%H:%M:%S"),
        "push_seconds": round(time.time() - started, 1),
        "c_echo": next((l.split(": ", 1)[1] for l in lines if l.startswith("C-ECHO")), "?"),
        "c_store": next((l.split(": ", 1)[1] for l in lines if l.startswith("C-STORE")), result.stderr.strip()[-200:]),
    }


def study_uid(case_dir: Path) -> str:
    for path in sorted(Path(case_dir).rglob("*.dcm")):
        return str(pydicom.dcmread(path, stop_before_pixels=True).StudyInstanceUID)
    raise FileNotFoundError(case_dir)


# --------------------------------------------------------------------- #
# Reading what happened
# --------------------------------------------------------------------- #
TERMINAL_EVENTS = {"results_delivered", "job_failed", "job_held", "send_failed", "job_superseded", "duplicate_skipped"}


def events(data_dir: Path) -> "list[dict]":
    path = Path(data_dir) / "audit" / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wait_for_outcomes(data_dir: Path, count: int, timeout: float = 1800) -> "list[dict]":
    """Wait until ``count`` terminal events exist in the event log."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        done = [e for e in events(data_dir) if e["event"] in TERMINAL_EVENTS]
        if len(done) >= count:
            return done
        time.sleep(5)
    raise TimeoutError(f"Only {len(done)} of {count} outcomes after {timeout}s")


def _jobs_by_study(data_dir: Path) -> "dict[str, list[dict]]":
    jobs: "dict[str, list[dict]]" = {}
    for job_json in sorted((Path(data_dir) / "jobs").glob("*/job.json")):
        manifest = json.loads(job_json.read_text())
        state = json.loads((job_json.parent / "state.json").read_text())
        manifest["state"] = state
        jobs.setdefault(manifest["study_instance_uid"], []).append(manifest)
    return jobs


def _received(data_dir: Path, uid: str) -> "list[Path]":
    return sorted((Path(data_dir) / "test_carpl" / uid).glob("*.dcm"))


def study_timeline(data_dir: Path, cases: "dict[str, Path]") -> pd.DataFrame:
    """One row per job: what was received, what the pipeline decided, what CARPL got."""
    jobs = _jobs_by_study(data_dir)
    evs = events(data_dir)
    rows = []
    for name, case_dir in cases.items():
        uid = study_uid(case_dir)
        for job in jobs.get(uid, []):
            state = job["state"]
            history = {h["state"]: h["at"] for h in state.get("history", [])}
            job_events = {e["event"]: e for e in evs if e.get("job_id") == job["job_id"]}
            delivered = job_events.get("results_delivered", {})
            rows.append({
                "study": name,
                "job_id": job["job_id"],
                "images_received": job["instance_count"],
                "series": len(job["series"]),
                "pipeline_status": state.get("status", ""),
                "final_state": state.get("state", ""),
                "inference_s": state.get("elapsed_seconds", ""),
                "queued_utc": history.get("queued", "")[11:19],
                "delivered_utc": history.get("delivered", "")[11:19],
                "objects_to_CARPL": len(delivered.get("results", [])) if delivered else 0,
            })
        skipped = [e for e in evs if e["event"] == "duplicate_skipped" and e.get("study_uid") == uid]
        for e in skipped:
            rows.append({"study": name, "job_id": "(resend)", "images_received": e.get("instances"),
                         "final_state": "duplicate_skipped", "queued_utc": e["at"][11:19], "objects_to_CARPL": 0})
    return pd.DataFrame(rows).convert_dtypes()


def received_objects(data_dir: Path, cases: "dict[str, Path]") -> pd.DataFrame:
    rows = []
    for name, case_dir in cases.items():
        for path in _received(data_dir, study_uid(case_dir)):
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            rows.append({
                "study": name,
                "object": ds.SOPClassUID.name,
                "modality": ds.Modality,
                "series_description": ds.get("SeriesDescription", ""),
                "same_StudyInstanceUID": str(ds.StudyInstanceUID) == study_uid(case_dir),
                "transfer_syntax": ds.file_meta.TransferSyntaxUID.name,
            })
    return pd.DataFrame(rows)


def received_file(data_dir: Path, case_dir: Path, modality: str) -> "Path | None":
    for path in _received(data_dir, study_uid(case_dir)):
        if pydicom.dcmread(path, stop_before_pixels=True).Modality == modality:
            return path
    return None


# --------------------------------------------------------------------- #
# RTSTRUCT and SR content
# --------------------------------------------------------------------- #
def rtstruct_table(path: Path) -> pd.DataFrame:
    ds = pydicom.dcmread(path)
    types = {o.ReferencedROINumber: o.RTROIInterpretedType for o in ds.get("RTROIObservationsSequence", [])}
    contours = {rc.ReferencedROINumber: len(rc.get("ContourSequence", [])) for rc in ds.get("ROIContourSequence", [])}
    geometry = {
        rc.ReferencedROINumber: ",".join(sorted({c.ContourGeometricType for c in rc.get("ContourSequence", [])}))
        for rc in ds.get("ROIContourSequence", [])
    }
    return pd.DataFrame([
        {"ROI": r.ROINumber, "name": r.ROIName, "type": types.get(r.ROINumber, ""),
         "contours": contours.get(r.ROINumber, 0), "geometry": geometry.get(r.ROINumber, "")}
        for r in ds.get("StructureSetROISequence", [])
    ] or [{"ROI": "-", "name": "(no ROIs: nothing segmented)", "type": "", "contours": 0, "geometry": ""}])


def sr_findings(path: Path) -> "tuple[pd.DataFrame, pd.DataFrame, dict]":
    """Return (prostate measurements, AI lesion findings, header facts) from the SR."""
    ds = pydicom.dcmread(path)
    groups = []
    for item in ds.ContentSequence:
        if item.ConceptNameCodeSequence[0].CodeMeaning != "Imaging Measurements":
            continue
        for group in item.ContentSequence:
            row: dict = {}
            for child in group.ContentSequence:
                name = child.ConceptNameCodeSequence[0].CodeMeaning
                if child.ValueType == "NUM":
                    mv = child.MeasuredValueSequence[0]
                    unit = mv.MeasurementUnitsCodeSequence[0].CodeValue
                    row[f"{name} ({unit})"] = float(mv.NumericValue)
                elif child.ValueType == "CODE" and name not in ("Finding Site",):
                    row[name] = child.ConceptCodeSequence[0].CodeMeaning
                elif child.ValueType == "TEXT" and name == "Tracking Identifier":
                    row["SR finding"] = child.TextValue
            groups.append(row)
    prostate = [g for g in groups if g.get("Finding") == "Prostate"]
    lesions = [g for g in groups if g.get("Finding") == "Lesion"]
    prostate_df = pd.DataFrame(
        [{"measurement": k, "value": v} for g in prostate for k, v in g.items() if "(" in k]
    )
    columns = ["SR finding", "Finding", "AI PI-RADS assessment category", "Probability (%)", "Lesion",
               "AI lesion volume (mm3)", "AI lesion major axis length (mm)"]
    lesions_df = pd.DataFrame(lesions)
    lesions_df = lesions_df[[c for c in columns if c in lesions_df.columns]].fillna("") if lesions else pd.DataFrame(
        [{"SR finding": "(none: no prostate segmented, so no lesion claim is made)"}])
    observer = []
    ds.walk(lambda d, el: observer.append(str(el.value)) if el.keyword == "TextValue" and "Prostate MRI" in str(el.value) else None)
    templates = [str(item.TemplateIdentifier) for item in ds.get("ContentTemplateSequence", [])]
    facts = {
        "SOP class": ds.SOPClassUID.name,
        "Content template": "TID " + ", ".join(templates) if templates else "(not declared)",
        "Document title": ds.ConceptNameCodeSequence[0].CodeMeaning,
        "AI observer (device)": observer[0] if observer else "",
        "Completion / verification": f"{ds.CompletionFlag} / {ds.VerificationFlag}",
        "Referenced T2 images": sum(
            len(series.ReferencedSOPSequence)
            for study in ds.CurrentRequestedProcedureEvidenceSequence
            for series in study.ReferencedSeriesSequence
        ),
        "StudyInstanceUID": str(ds.StudyInstanceUID),
    }
    return prostate_df, lesions_df, facts


def sr_compliance(sr_path: Path, case_dir: Path) -> pd.DataFrame:
    """Check one received SR against CARPL DICOM Push guide 6.2 (Method 2)."""
    ds = pydicom.dcmread(sr_path)
    uid = study_uid(case_dir)
    input_sops = set()
    for path in Path(case_dir).rglob("*.dcm"):
        input_sops.add(str(pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["SOPInstanceUID"]).SOPInstanceUID))
    referenced = {
        str(sop.ReferencedSOPInstanceUID)
        for study in ds.CurrentRequestedProcedureEvidenceSequence
        for series in study.ReferencedSeriesSequence
        for sop in series.ReferencedSOPSequence
    }
    _prostate, lesions, facts = sr_findings(sr_path)
    findings = lesions[lesions["Finding"] == "Lesion"] if "Finding" in lesions.columns else lesions.iloc[0:0]
    absent = "Lesion" in findings.columns and (findings["Lesion"] == "Absent").any()
    detected = findings[findings["Lesion"] != "Absent"] if "Lesion" in findings.columns else findings
    with_probability = (
        detected[detected["Probability (%)"] != ""] if "Probability (%)" in detected.columns else detected.iloc[0:0]
    )
    no_prostate = findings.empty
    rows = [
        ("Output object type: DICOM SR", ds.Modality == "SR" and "SR" in ds.SOPClassUID.name, ds.SOPClassUID.name),
        ("Template: TID 1500 Measurement Report", facts["Content template"] == "TID 1500"
         and facts["Document title"] == "Imaging Measurement Report", f"{facts['Content template']}, '{facts['Document title']}'"),
        ("Observer context (AI algorithm identity)", bool(facts["AI observer (device)"]), f"Device: {facts['AI observer (device)']}"),
        ("Referenced study/series/SOP", str(ds.StudyInstanceUID) == uid and referenced and referenced <= input_sops,
         f"{len(referenced)} T2 images of the received study"),
        ("Finding concept name", not findings.empty or no_prostate,
         "no prostate segmented: no lesion claim" if no_prostate else
         "SCT 52988006 'Lesion' in the prostate" + (": Absent" if absent else f" x {len(detected)}")),
        ("Probability / confidence score", absent or no_prostate or (len(detected) > 0 and len(with_probability) == len(detected)),
         "not applicable: no lesion detected" if (absent or no_prostate) else
         ", ".join(f"{r['SR finding']}: {r['Probability (%)']}%" for _, r in with_probability.iterrows())),
        ("Results associated with the correct input study", str(ds.StudyInstanceUID) == uid, "StudyInstanceUID matches"),
    ]
    return pd.DataFrame([{"CARPL requirement (guide 6.2)": r, "result": "PASS" if ok else "FAIL", "evidence": e}
                         for r, ok, e in rows])


def delivered_job_output(data_dir: Path, case_dir: Path) -> Path:
    """Output folder of the job whose results were delivered for this study."""
    for job in reversed(_jobs_by_study(data_dir).get(study_uid(case_dir), [])):
        if job["state"].get("state") == "delivered":
            return Path(data_dir) / "jobs" / job["job_id"] / "output"
    raise FileNotFoundError(f"No delivered job for {Path(case_dir).name}")


# --------------------------------------------------------------------- #
# Contours on T2
# --------------------------------------------------------------------- #
_COLOURS = ["#00e676", "#ff1744", "#ffab00", "#d500f9", "#00b0ff", "#ff6d00"]


def _t2_index(case_dir: Path, sop_uids: "set[str]") -> "dict[str, Path]":
    index = {}
    for path in Path(case_dir).rglob("*.dcm"):
        ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["SOPInstanceUID"])
        if str(ds.SOPInstanceUID) in sop_uids:
            index[str(ds.SOPInstanceUID)] = path
    return index


def _to_pixels(points_mm: np.ndarray, image: pydicom.Dataset) -> np.ndarray:
    origin = np.array(image.ImagePositionPatient, dtype=float)
    iop = np.array(image.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    spacing_row, spacing_col = (float(v) for v in image.PixelSpacing)
    rel = points_mm - origin
    x = rel @ row_dir / spacing_col
    y = rel @ col_dir / spacing_row
    return np.stack([x, y], axis=1)


def plot_contours(rtstruct_path: Path, case_dir: Path, title: str, max_panels: int = 3):
    """Show the received RTSTRUCT contours on the original T2 images."""
    import matplotlib.pyplot as plt

    ds = pydicom.dcmread(rtstruct_path)
    names = {r.ROINumber: r.ROIName for r in ds.get("StructureSetROISequence", [])}
    by_sop: "dict[str, list[tuple[int, np.ndarray]]]" = {}
    area: "dict[int, dict[str, float]]" = {}
    for rc in ds.get("ROIContourSequence", []):
        for contour in rc.get("ContourSequence", []):
            sop = str(contour.ContourImageSequence[0].ReferencedSOPInstanceUID)
            pts = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            by_sop.setdefault(sop, []).append((int(rc.ReferencedROINumber), pts))
            x, y = pts[:, 0], pts[:, 1]
            area.setdefault(int(rc.ReferencedROINumber), {})
            area[int(rc.ReferencedROINumber)][sop] = area[int(rc.ReferencedROINumber)].get(sop, 0.0) + \
                0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    if not by_sop:
        print(f"{title}: RTSTRUCT has no contours (nothing segmented), so there is nothing to draw.")
        return None
    lesion_rois = sorted(r for r in area if r != 1)
    chosen = []
    for roi in lesion_rois:
        sop = max(area[roi], key=area[roi].get)
        if sop not in chosen:
            chosen.append(sop)
    if not chosen and 1 in area:
        chosen.append(max(area[1], key=area[1].get))
    chosen = chosen[:max_panels]
    images = _t2_index(case_dir, set(chosen))
    fig, axes = plt.subplots(1, len(chosen), figsize=(5.2 * len(chosen), 5.6), squeeze=False)
    for ax, sop in zip(axes[0], chosen):
        image = pydicom.dcmread(images[sop])
        pixels = image.pixel_array.astype(float)
        lo, hi = np.percentile(pixels, [1, 99.5])
        ax.imshow(pixels, cmap="gray", vmin=lo, vmax=hi)
        drawn = set()
        all_xy = []
        for roi, pts in by_sop.get(sop, []):
            xy = _to_pixels(pts, image)
            all_xy.append(xy)
            closed = np.vstack([xy, xy[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=_COLOURS[(roi - 1) % len(_COLOURS)], linewidth=1.6,
                    label=names.get(roi, str(roi)) if roi not in drawn else None)
            drawn.add(roi)
        xy = np.vstack(all_xy)
        cx, cy = xy.mean(axis=0)
        half = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) * 0.9 + 15
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + half, cy - half)
        ax.set_title(f"T2 image {image.get('InstanceNumber', '')}", fontsize=10)
        ax.axis("off")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- #
# Reference run and parity
# --------------------------------------------------------------------- #
def direct_run(case_dir: Path, output_dir: Path, version: str) -> dict:
    """Run the unchanged inference image directly on the original folder (as on HIVE)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = _run([
        "docker", "run", "--rm", "--gpus", "all", "--shm-size=1g",
        "-v", f"{Path(case_dir).resolve()}:/input:ro", "-v", f"{output_dir.resolve()}:/output",
        "-v", f"{REPO / 'models'}:/models:ro", f"prostate-mri-seg:{version}",
    ], check=False)
    # The HIVE-style run is root; hand the files back to the notebook user.
    _run(["docker", "run", "--rm", "-v", f"{output_dir.resolve()}:/output", "--entrypoint", "chown",
          f"prostate-mri-seg:{version}", "-R", UID_GID, "/output"], check=False)
    (output_dir.parent / f"{output_dir.name}.log").write_text(result.stdout + result.stderr)
    return {"study": Path(case_dir).name, "exit_code": result.returncode, "seconds": round(time.time() - started)}


def parity(reference: Path, current: Path, version: str) -> dict:
    """Run carpl/tools/parity_check.py inside the worker image."""
    ref = "/repo/" + Path(reference).resolve().relative_to(REPO).as_posix()
    cur = "/repo/" + Path(current).resolve().relative_to(REPO).as_posix()
    result = _run([
        "docker", "run", "--rm", "-u", UID_GID, "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "python",
        "-v", f"{REPO}:/repo:ro", f"prostate-carpl-ai-worker:{version}",
        "/repo/carpl/tools/parity_check.py", "outputs", ref, cur,
    ], check=False)
    lines = [l for l in result.stdout.splitlines() if l.strip()]
    pick = lambda key: next((l.split(": ", 1)[1] for l in lines if l.strip().startswith(key)), "")
    return {
        "verdict": "PASS" if result.returncode == 0 else "FAIL",
        "case_status": pick("case_status"),
        "NIfTI files compared": pick("NIfTI files compared"),
        "series selected (T2/ADC/high-b)": "same" if all("same" in l for l in lines if l.strip().startswith("selected")) else "DIFFERENT",
        "RTSTRUCT round-trip (CARPL output)": pick("validate_rtstruct_outputs[current]"),
        "notes": "; ".join(l.strip()[6:] for l in lines if l.strip().startswith("note:")),
        "problems": "; ".join(l.strip()[2:] for l in lines if l.strip().startswith("- ")),
        "raw": result.stdout,
    }
