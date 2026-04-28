"""Build the prostate-mri-seg Docker image and save it as a .tar file.

Run from the project root:
    python scripts/build_and_save.py

The resulting docker/docker_image.tar can be transferred to HIVE (or any
air-gapped machine) and loaded with:
    docker load -i docker/docker_image.tar
"""

import subprocess
import sys
from pathlib import Path

IMAGE_NAME = "prostate-mri-seg"
IMAGE_TAG = "latest"


def _run(cmd: list[str], *, fail_msg: str) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nError: {fail_msg}")
        sys.exit(1)


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    docker_dir = root_dir / "docker"
    app_dir = root_dir / "prostate_mri_lesion_seg_app"
    dockerfile = docker_dir / "Dockerfile"
    req_file = docker_dir / "requirements.txt"
    tar_file = docker_dir / "docker_image.tar"
    dockerignore = root_dir / ".dockerignore"

    print("=" * 60)
    print("  Prostate MRI Seg – Docker Build & Save")
    print("=" * 60)
    print(f"  Project root : {root_dir}")
    print(f"  Image        : {IMAGE_NAME}:{IMAGE_TAG}")
    print(f"  Dockerfile   : {dockerfile}")
    print(f"  Output tar   : {tar_file}")
    print()

    # ------------------------------------------------------------------
    # Pre-flight checks
    # ------------------------------------------------------------------
    errors: list[str] = []
    if not dockerfile.exists():
        errors.append(f"Dockerfile not found: {dockerfile}")
    if not req_file.exists():
        errors.append(f"requirements.txt not found: {req_file}")
    if not app_dir.exists():
        errors.append(f"Application directory not found: {app_dir}")
    if not (app_dir / "__main__.py").exists():
        errors.append(f"__main__.py not found in {app_dir}")
    if not (app_dir / "app.py").exists():
        errors.append(f"app.py not found in {app_dir}")
    if errors:
        for e in errors:
            print(f"  [FAIL] {e}")
        sys.exit(1)

    if not dockerignore.exists():
        print("  [WARN] .dockerignore not found – build context may include")
        print("         large files (models, data). Performance will suffer.\n")

    result = subprocess.run(
        ["docker", "--version"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("  [FAIL] Docker is not installed or not in PATH.")
        sys.exit(1)
    print(f"  Docker : {result.stdout.strip()}")
    print()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    print(f"Building image {IMAGE_NAME}:{IMAGE_TAG} ...")
    print("(first build downloads ~4 GB CUDA base + PyTorch; subsequent")
    print(" builds use the Docker cache and are much faster)\n")

    _run(
        [
            "docker", "build",
            "-f", str(dockerfile),
            "-t", f"{IMAGE_NAME}:{IMAGE_TAG}",
            str(root_dir),
        ],
        fail_msg="Docker build failed!",
    )
    print(f"\nImage {IMAGE_NAME}:{IMAGE_TAG} built successfully.")

    # ------------------------------------------------------------------
    # Save to tar
    # ------------------------------------------------------------------
    print(f"\nSaving image to {tar_file} ...")
    print("(this may take several minutes for a large image)\n")

    _run(
        ["docker", "save", "-o", str(tar_file), f"{IMAGE_NAME}:{IMAGE_TAG}"],
        fail_msg="Failed to save Docker image!",
    )

    size_bytes = tar_file.stat().st_size
    size_gb = size_bytes / (1024 ** 3)
    print(f"\nImage saved: {tar_file}")
    print(f"File size  : {size_bytes:,} bytes ({size_gb:.2f} GB)")

    # ------------------------------------------------------------------
    # Next steps
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  BUILD COMPLETE – Next steps for HIVE transfer")
    print("=" * 60)
    print(f"""
  1. Copy these items to HIVE:
     a) {tar_file.name}  (Docker image)
     b) models/ directory  (7 model weight files)
     c) DICOM input data   (e.g. test-data/ProstateX-0004/)
     d) scripts/load_and_run.py  (run helper)

  2. On HIVE, load and run:
     docker load -i docker_image.tar
     python scripts/load_and_run.py -i <input_dir> -o <output_dir> -m <models_dir>
""")


if __name__ == "__main__":
    main()
