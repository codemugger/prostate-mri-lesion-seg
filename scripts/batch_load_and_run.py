"""Batch-process multiple patients through the Dockerised pipeline.

Mirrors the behaviour of scripts/test_batch.sh (resume, cooldown, progress
tracking, per-patient output directories) but runs each case inside a fresh
Docker container for crash isolation and clean GPU memory between patients.

Usage:
    python scripts/batch_load_and_run.py \
        -i test-data/ -o output/ -m models/ -r --cooldown 5

HIVE usage:
    python batch_load_and_run.py \
        -i /hive/data/patients/ \
        -o /hive/output/ \
        -m /hive/models \
        --tar /hive/docker_image.tar \
        -r --cooldown 5
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Make scripts/ importable so we can use the shared audit logger.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from case_status import (  # noqa: E402
    append_csv_row,
    already_logged,
    inspect_case,
    summarise,
)

IMAGE_NAME = "prostate-mri-seg"
IMAGE_TAG = "latest"
CONTAINER_NAME = "prostate-mri-seg-batch"
AUDIT_CSV_FILENAME = "batch_audit_log.csv"

EXPECTED_MODELS = [
    "organ/model.ts",
    "fold0/model_best_fold0.pth.tar",
    "fold1/model_best_fold1.pth.tar",
    "fold2/model_best_fold2.pth.tar",
    "fold3/model_best_fold3.pth.tar",
    "fold4/model_best_fold4.pth.tar",
    "classifier/model_best.pth.tar",
]

RESUME_CHECK_EXTENSIONS = {".nii.gz", ".dcm"}


def _check_docker() -> None:
    result = subprocess.run(
        ["docker", "--version"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("  [FAIL] Docker is not installed or not in PATH.")
        sys.exit(1)
    print(f"  Docker : {result.stdout.strip()}")


def _check_gpu() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            print(f"  GPU    : {line.strip()}")
        return True
    print("  GPU    : not detected (nvidia-smi unavailable)")
    return False


def _image_exists() -> bool:
    result = subprocess.run(
        ["docker", "images", "-q", f"{IMAGE_NAME}:{IMAGE_TAG}"],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def _validate_models(models_path: Path) -> None:
    missing = [m for m in EXPECTED_MODELS if not (models_path / m).exists()]
    if missing:
        print(f"\n  [FAIL] Missing model files in {models_path}:")
        for m in missing:
            print(f"         - {m}")
        print("\n  Download models from the Google Drive link in README.md")
        sys.exit(1)
    print(f"  Models : all {len(EXPECTED_MODELS)} files present")


def _count_output_files(directory: Path) -> int:
    """Count .nii.gz and .dcm files (same logic as test_batch.sh resume check)."""
    if not directory.exists():
        return 0
    count = 0
    for f in directory.rglob("*"):
        if f.is_file() and any(f.name.endswith(ext) for ext in RESUME_CHECK_EXTENSIONS):
            count += 1
    return count


def _run_single_patient(
    patient_dir: Path,
    output_dir: Path,
    models_path: Path,
    use_cpu: bool,
) -> int:
    """Run one patient through docker and return the exit code."""
    # Kill any leftover container with the same name
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME], capture_output=True
    )

    run_cmd = [
        "docker", "run",
        "--rm",
        "--name", CONTAINER_NAME,
        "--shm-size=1g",
        "-v", f"{patient_dir}:/input:ro",
        "-v", f"{output_dir}:/output",
        "-v", f"{models_path}:/models:ro",
    ]
    if not use_cpu:
        run_cmd += ["--gpus", "all", "-e", "NVIDIA_VISIBLE_DEVICES=all"]
    run_cmd.append(f"{IMAGE_NAME}:{IMAGE_TAG}")

    result = subprocess.run(run_cmd)
    return result.returncode


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    docker_dir = root_dir / "docker"
    default_tar = docker_dir / "docker_image.tar"

    parser = argparse.ArgumentParser(
        description="Batch-process patients through Dockerised segmentation pipeline"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Root directory containing one subdirectory per patient",
    )
    parser.add_argument(
        "-o", "--output", type=str,
        default=str(root_dir / "output"),
        help="Root output directory (per-patient subdirectories created automatically)",
    )
    parser.add_argument(
        "-m", "--models", type=str,
        default=str(root_dir / "prostate_mri_lesion_seg_app" / "models"),
        help="Path to models directory",
    )
    parser.add_argument(
        "--tar", type=str,
        default=str(default_tar),
        help="Path to docker_image.tar",
    )
    parser.add_argument(
        "-r", "--resume", action="store_true",
        help="Resume mode: skip patients that already have output files",
    )
    parser.add_argument(
        "--cooldown", type=int, default=30,
        help="Seconds to pause between patients (default: 30)",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Disable GPU (much slower)",
    )
    parser.add_argument(
        "--skip-load", action="store_true",
        help="Skip docker load (image already loaded)",
    )
    parser.add_argument(
        "--audit-csv", type=str, default="",
        help="Path to the per-case audit CSV "
             f"(default <output>/{AUDIT_CSV_FILENAME}).",
    )
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    models_path = Path(args.models).resolve()
    tar_path = Path(args.tar).resolve()
    audit_csv = (
        Path(args.audit_csv).resolve()
        if args.audit_csv
        else output_root / AUDIT_CSV_FILENAME
    )

    # ---- Banner ----
    print("=" * 60)
    print("  Prostate MRI Seg – Batch Docker Processing")
    print("=" * 60)
    print(f"  Image    : {IMAGE_NAME}:{IMAGE_TAG}")
    print(f"  Input    : {input_root}")
    print(f"  Output   : {output_root}")
    print(f"  Models   : {models_path}")
    print(f"  GPU      : {'disabled (CPU mode)' if args.cpu else 'enabled'}")
    print(f"  Resume   : {'yes' if args.resume else 'no'}")
    print(f"  Cooldown : {args.cooldown}s")
    print(f"  Audit CSV: {audit_csv}")

    # ---- Prerequisites ----
    print(f"\n{'─' * 60}")
    print("  Prerequisite checks")
    print(f"{'─' * 60}")
    _check_docker()
    _check_gpu()

    if not input_root.exists():
        print(f"  [FAIL] Input root not found: {input_root}")
        sys.exit(1)
    print(f"  Input  : OK")

    if not models_path.exists():
        print(f"  [FAIL] Models directory not found: {models_path}")
        sys.exit(1)
    _validate_models(models_path)

    # ---- Load Docker image ----
    print(f"\n{'─' * 60}")
    print("  Docker image")
    print(f"{'─' * 60}")
    if not args.skip_load:
        if not tar_path.exists():
            print(f"  [FAIL] Tar file not found: {tar_path}")
            print("         Build first with: python scripts/build_and_save.py")
            print("         Or use --skip-load if the image is already loaded.")
            sys.exit(1)
        print(f"  Loading {tar_path.name} ...")
        result = subprocess.run(["docker", "load", "-i", str(tar_path)])
        if result.returncode != 0:
            print("  [FAIL] docker load failed!")
            sys.exit(1)
        print(f"  Image loaded: {IMAGE_NAME}:{IMAGE_TAG}")
    else:
        print("  Skipping docker load (--skip-load)")

    if not _image_exists():
        print(f"  [FAIL] Image {IMAGE_NAME}:{IMAGE_TAG} not found.")
        sys.exit(1)
    print(f"  Image verified: {IMAGE_NAME}:{IMAGE_TAG}")

    # ---- Discover patients ----
    patient_dirs = sorted(
        [d for d in input_root.iterdir() if d.is_dir()]
    )
    total = len(patient_dirs)
    if total == 0:
        print(f"\n  [FAIL] No patient subdirectories found in {input_root}")
        sys.exit(1)

    # Prepare output root.
    # On a fresh (non-resume) run we wipe the output tree AND the audit
    # CSV so each run produces a self-contained audit record.  On resume
    # both are preserved and any previously-logged case IDs are skipped
    # by the CSV appender so rows never duplicate.
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for child in output_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            elif child.is_file():
                child.unlink()
        if audit_csv.exists():
            audit_csv.unlink()
    else:
        audit_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        print(f"\n  *** RESUME mode: skipping cases that already have output files ***")
        print(f"  *** RESUME mode: existing audit CSV will be appended to         ***")

    # ---- Batch loop ----
    print(f"\n{'─' * 60}")
    print(f"  Batch processing {total} patients")
    print(f"{'─' * 60}")

    succeeded = 0
    failed = 0
    skipped = 0
    failed_patients: list[str] = []

    for idx, patient_dir in enumerate(patient_dirs, start=1):
        patient_id = patient_dir.name
        case_output = output_root / patient_id

        print(f"\n{'=' * 60}")
        print(f"  [{idx}/{total}] Patient: {patient_id}")
        print(f"  Input:  {patient_dir}")
        print(f"  Output: {case_output}")

        # Resume check (same logic as test_batch.sh)
        if args.resume and case_output.exists():
            existing = _count_output_files(case_output)
            if existing > 0:
                print(f"  SKIPPING (resume mode, {existing} output files already exist)")
                skipped += 1
                # Still inspect and log unless the case was already logged
                # in a previous run of this batch (prevents row duplication
                # across repeat resumes).
                if not already_logged(audit_csv, patient_id):
                    try:
                        case = inspect_case(
                            case_output,
                            patient_id=patient_id,
                            input_dir=patient_dir,
                            exit_code=None,
                            elapsed_seconds=None,
                            resume_skipped=True,
                        )
                        append_csv_row(audit_csv, case)
                        print(f"  Audit: {case.status}")
                    except Exception as e:  # noqa: BLE001
                        print(f"  WARNING: audit log append failed for "
                              f"{patient_id}: {e}")
                continue

        case_output.mkdir(parents=True, exist_ok=True)

        print(f"  Running {'on CPU' if args.cpu else 'on GPU'}...")
        case_start = time.monotonic()
        exit_code = _run_single_patient(
            patient_dir=patient_dir,
            output_dir=case_output,
            models_path=models_path,
            use_cpu=args.cpu,
        )
        elapsed = time.monotonic() - case_start

        if exit_code != 0:
            print(f"  *** FAILED (exit code {exit_code}) ***")
            failed += 1
            failed_patients.append(patient_id)
        else:
            print(f"  DONE (exit 0)")
            succeeded += 1

        # Always append an audit row, success or failure.
        try:
            case = inspect_case(
                case_output,
                patient_id=patient_id,
                input_dir=patient_dir,
                exit_code=exit_code,
                elapsed_seconds=elapsed,
                resume_skipped=False,
            )
            append_csv_row(audit_csv, case)
            miss = (
                f" [missing: {','.join(case.missing_modalities)}]"
                if case.missing_modalities else ""
            )
            print(f"  Audit: {case.status}{miss}")
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: audit log append failed for {patient_id}: {e}")

        # Cooldown between patients (skip after the last one)
        if idx < total:
            print(f"  Cooldown {args.cooldown}s before next case...")
            time.sleep(args.cooldown)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  Batch processing complete")
    print(f"{'=' * 60}")
    print(f"  Total: {total}  Succeeded: {succeeded}  Failed: {failed}  Skipped: {skipped}")
    print(f"  Exit-code counters above reflect the container's raw return status.")
    print(f"  For the full per-case outcome (including silent empty outputs)")
    print(f"  inspect the audit CSV: {audit_csv}")
    if failed_patients:
        print(f"\n  Failed patients (non-zero exit):")
        for p in failed_patients:
            print(f"    - {p}")

    # Per-status breakdown from the audit CSV.
    if audit_csv.exists():
        status_counts = summarise(audit_csv)
        if status_counts:
            print(f"\n  Per-status breakdown from audit CSV:")
            for status in sorted(status_counts):
                print(f"    {status:<24} {status_counts[status]:>5}")
    print()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
