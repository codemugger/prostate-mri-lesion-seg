"""Load the prostate-mri-seg Docker image and run the pipeline.

Usage (from project root or anywhere):
    python scripts/load_and_run.py -i <input_dir> -o <output_dir> -m <models_dir>

Typical HIVE usage:
    python load_and_run.py \\
        -i /hive/data/ProstateX-0004 \\
        -o /hive/output/ProstateX-0004 \\
        -m /hive/models \\
        --tar /hive/docker_image.tar
"""

import argparse
import subprocess
import sys
from pathlib import Path

IMAGE_NAME = "prostate-mri-seg"
IMAGE_TAG = "latest"
CONTAINER_NAME = "prostate-mri-seg"

EXPECTED_MODELS = [
    "organ/model.ts",
    "fold0/model_best_fold0.pth.tar",
    "fold1/model_best_fold1.pth.tar",
    "fold2/model_best_fold2.pth.tar",
    "fold3/model_best_fold3.pth.tar",
    "fold4/model_best_fold4.pth.tar",
    "classifier/model_best.pth.tar",
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    docker_dir = root_dir / "docker"
    default_tar = docker_dir / "docker_image.tar"

    parser = argparse.ArgumentParser(
        description="Load and run prostate MRI lesion segmentation container"
    )
    parser.add_argument(
        "-i", "--input", type=str,
        default=str(root_dir / "test-data" / "ProstateX-0004"),
        help="Path to input DICOM directory",
    )
    parser.add_argument(
        "-o", "--output", type=str,
        default=str(root_dir / "output"),
        help="Path to output directory",
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
        "--cpu", action="store_true",
        help="Disable GPU (much slower)",
    )
    parser.add_argument(
        "--skip-load", action="store_true",
        help="Skip docker load (image already loaded)",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only run prerequisite checks, do not start container",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    models_path = Path(args.models).resolve()
    tar_path = Path(args.tar).resolve()

    # ---- Banner ----
    print("=" * 60)
    print("  Prostate MRI Seg – Docker Load & Run")
    print("=" * 60)
    print(f"  Image  : {IMAGE_NAME}:{IMAGE_TAG}")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Models : {models_path}")
    print(f"  GPU    : {'disabled (CPU mode)' if args.cpu else 'enabled'}")

    # ---- Prerequisites ----
    _print_section("Prerequisite checks")
    _check_docker()
    has_gpu = _check_gpu()
    if not args.cpu and not has_gpu:
        print("\n  [WARN] No GPU detected but GPU mode requested.")
        print("         Pass --cpu to run without GPU, or install")
        print("         nvidia-container-toolkit for GPU support.")

    # ---- Validate paths ----
    _print_section("Path validation")
    if not input_path.exists():
        print(f"  [FAIL] Input directory not found: {input_path}")
        sys.exit(1)
    print(f"  Input  : OK")

    if not models_path.exists():
        print(f"  [FAIL] Models directory not found: {models_path}")
        sys.exit(1)
    _validate_models(models_path)

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"  Output : OK ({output_path})")

    if args.check_only:
        _print_section("Check-only mode – exiting without running")
        return

    # ---- Load image ----
    _print_section("Docker image")
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
        print(f"  [FAIL] Image {IMAGE_NAME}:{IMAGE_TAG} not found in Docker.")
        print("         Remove --skip-load and ensure tar file is present.")
        sys.exit(1)
    print(f"  Image verified: {IMAGE_NAME}:{IMAGE_TAG}")

    # ---- Remove stale container ----
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
    )

    # ---- Run ----
    _print_section("Running pipeline")
    run_cmd = [
        "docker", "run",
        "--rm",
        "--name", CONTAINER_NAME,
        "--shm-size=1g",
        "-v", f"{input_path}:/input:ro",
        "-v", f"{output_path}:/output",
        "-v", f"{models_path}:/models:ro",
    ]
    if not args.cpu:
        run_cmd += ["--gpus", "all", "-e", "NVIDIA_VISIBLE_DEVICES=all"]
    run_cmd.append(f"{IMAGE_NAME}:{IMAGE_TAG}")

    print(f"  $ {' '.join(run_cmd)}\n")
    result = subprocess.run(run_cmd)

    if result.returncode != 0:
        print(f"\n  [FAIL] Container exited with code {result.returncode}")
        sys.exit(result.returncode)

    # ---- Done ----
    _print_section("Pipeline completed successfully")
    print(f"  Results saved to: {output_path}")
    print()
    expected_outputs = [
        "t2/t2.nii.gz", "adc/adc.nii.gz", "highb/highb.nii.gz",
        "organ/organ.nii.gz", "organ/cleaned_organ.nii.gz",
        "organ/cleanup_metrics.json",
        "organ/organ_RTSTRUCT.dcm",
        "organ/cleaned_organ_RTSTRUCT.dcm",
        "lesion/lesion_mask.nii.gz", "lesion/merged_lesion_prob.nii.gz",
        "lesion/lesion_RTSTRUCT.dcm",
        "lesions.txt", "prostate_measurements_SR.dcm",
        "combined_organ_lesion_RTSTRUCT.dcm",
    ]
    missing_outputs = []
    for f in expected_outputs:
        full = output_path / f
        status = "OK" if full.exists() else "MISSING"
        print(f"  [{status}] {f}")
        if not full.exists():
            missing_outputs.append(f)
    print()
    if missing_outputs:
        print(
            "  [FAIL] Container exited successfully but required output "
            "artifacts are missing."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
