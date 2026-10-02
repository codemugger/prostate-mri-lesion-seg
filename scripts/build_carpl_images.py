#!/usr/bin/env python3
"""Build, verify and package the prostate MRI CARPL images.

    python scripts/build_carpl_images.py --version 1.0.0 --build-base
    python scripts/build_carpl_images.py --version 1.0.0            # reuse an existing base image

Steps:

1. Base inference image ``prostate-mri-seg:<version>``.  With
   ``--build-base`` it is built from the unchanged ``docker/Dockerfile``;
   otherwise it must already exist, e.g. an image already validated on HIVE.
2. ``prostate-carpl-ai-worker:<version>``, built FROM the base image.
3. ``prostate-carpl-dicom-listener``, ``prostate-carpl-dicom-sender`` and
   the dev-only ``prostate-carpl-receiver`` (context ``carpl/``).
4. Gates, all of which must pass:
   * the base image's layers are a prefix of the worker's layers, so the
     inference environment and app code are byte-identical;
   * ``pip freeze`` is identical in the base and worker images;
   * the base image has monai-deploy-app-sdk 2.0.0, the version the input
     layout and model-folder rules were verified against;
   * the worker's CARPL imports load;
   * ``tests/test_carpl_*.py`` pass inside the listener image.
5. Unless ``--no-save``, write a deploy bundle to ``dist/prostate-carpl-<version>/``:
   ``images.tar`` (base, worker, listener, sender, redis) with its
   ``.sha256``, ``docker-compose-carpl.yml``, ``.env.example``,
   ``README.md`` and ``build_manifest.json``.

``scripts/build_and_save.py`` (HIVE image) is unchanged.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARPL = ROOT / "carpl"
REDIS_IMAGE = "redis:7.4-alpine"
EXPECTED_SDK_VERSION = "2.0.0"


def run(cmd: "list[str]", *, capture: bool = False, check: bool = True, env=None) -> subprocess.CompletedProcess:
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture, env=env)


def image_exists(ref: str) -> bool:
    return run(["docker", "image", "inspect", ref], capture=True, check=False).returncode == 0


def inspect(ref: str, template: str) -> str:
    return run(["docker", "image", "inspect", "-f", template, ref], capture=True).stdout.strip()


def docker_python(image: str, code: str, *extra: str) -> str:
    return run(
        ["docker", "run", "--rm", *extra, "--entrypoint", "python", image, "-c", code],
        capture=True,
    ).stdout


def build(dockerfile: Path, tag: str, context: Path, build_args: "dict[str, str] | None" = None) -> None:
    cmd = ["docker", "build", "-f", str(dockerfile), "-t", tag]
    for key, value in (build_args or {}).items():
        cmd += ["--build-arg", f"{key}={value}"]
    env = dict(os.environ, DOCKER_BUILDKIT="1")  # needed for Dockerfile.dockerignore
    run(cmd + [str(context)], env=env)


def gate(condition: bool, message: str) -> None:
    print(("  [PASS] " if condition else "  [FAIL] ") + message, flush=True)
    if not condition:
        raise SystemExit(f"Gate failed: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="image tag, e.g. 1.0.0")
    parser.add_argument("--build-base", action="store_true", help="build prostate-mri-seg:<version> from docker/Dockerfile")
    parser.add_argument("--base-image", help="use this existing base image instead of prostate-mri-seg:<version>")
    parser.add_argument("--no-save", action="store_true", help="skip writing the deploy bundle")
    parser.add_argument("--skip-tests", action="store_true", help="skip the CARPL unit tests inside the listener image")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()

    version = args.version
    base = args.base_image or f"prostate-mri-seg:{version}"
    worker = f"prostate-carpl-ai-worker:{version}"
    listener = f"prostate-carpl-dicom-listener:{version}"
    sender = f"prostate-carpl-dicom-sender:{version}"
    receiver = f"prostate-carpl-receiver:{version}"

    print("\n=== 1. Base inference image ===")
    if args.build_base:
        build(ROOT / "docker" / "Dockerfile", base, ROOT)
    elif not image_exists(base):
        raise SystemExit(f"Base image {base} not found; pass --build-base or --base-image")
    base_id = inspect(base, "{{.Id}}")
    print(f"Base image {base} = {base_id}")

    print("\n=== 2-3. CARPL images ===")
    build(CARPL / "ai-worker" / "Dockerfile", worker, ROOT, {"BASE_IMAGE": base, "BASE_IMAGE_ID": base_id})
    build(CARPL / "dicom-listener" / "Dockerfile", listener, CARPL)
    build(CARPL / "dicom-sender" / "Dockerfile", sender, CARPL)
    build(CARPL / "carpl-receiver" / "Dockerfile", receiver, CARPL)

    print("\n=== 4. Gates ===")
    base_layers = json.loads(inspect(base, "{{json .RootFS.Layers}}"))
    worker_layers = json.loads(inspect(worker, "{{json .RootFS.Layers}}"))
    gate(worker_layers[: len(base_layers)] == base_layers,
         f"worker reuses all {len(base_layers)} base layers unchanged (+{len(worker_layers) - len(base_layers)} CARPL layers)")

    freeze_code = "import subprocess,sys; print(subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout)"
    base_freeze = docker_python(base, freeze_code)
    worker_freeze = docker_python(worker, freeze_code)
    gate(base_freeze == worker_freeze and bool(base_freeze.strip()), "pip freeze identical in base and worker images")

    # The only allowed site-packages change is holoscan's wheel-axle
    # finalisation, which root runs of the base image also perform at startup.
    site = "/usr/local/lib/python3.10/dist-packages"
    list_cmd = ["docker", "run", "--rm", "--entrypoint", "sh"]
    base_files = set(run(list_cmd + [base, "-c", f"cd {site} && find . | sort"], capture=True).stdout.split())
    worker_files = set(run(list_cmd + [worker, "-c", f"cd {site} && find . | sort"], capture=True).stdout.split())
    symlinks = run(list_cmd + [base, "-c", f"cat {site}/holoscan-*.dist-info/symlinks.txt"], capture=True).stdout
    allowed_added = {"./" + line.split(",")[0].split("/purelib/", 1)[1] for line in symlinks.splitlines() if "/purelib/" in line}
    allowed_added |= {p for p in worker_files if p.startswith("./holoscan-") and p.endswith(".dist-info/axle.done")}
    allowed_removed = {p for p in base_files if p.startswith("./holoscan-") and p.endswith(".pth")}
    added, removed = worker_files - base_files, base_files - worker_files
    gate(added <= allowed_added and removed <= allowed_removed,
         f"site-packages differs only by holoscan wheel-axle finalisation (+{len(added)} symlinks/marker, -{len(removed)} .pth)"
         + ("" if added <= allowed_added and removed <= allowed_removed
            else f"; unexpected: {sorted(added - allowed_added)[:5]} {sorted(removed - allowed_removed)[:5]}"))
    non_root = run(["docker", "run", "--rm", "-u", "1000:1000", "-e", "HOME=/tmp", "--entrypoint", "python", worker,
                    "-c", "import holoscan.core, monai.deploy.core; print('ok')"], capture=True, check=False)
    gate(non_root.returncode == 0, "worker imports holoscan/MONAI Deploy as a non-root user")

    sdk = docker_python(base, "import importlib.metadata as m; print(m.version('monai-deploy-app-sdk'))").strip()
    gate(sdk == EXPECTED_SDK_VERSION, f"monai-deploy-app-sdk {sdk} == {EXPECTED_SDK_VERSION}")

    smoke = docker_python(
        worker,
        "import sys; sys.path.insert(0,'/opt/carpl/lib'); import redis, case_status, carpl_common; "
        "sys.path.insert(0,'/opt/carpl'); import worker; "
        "print(redis.__version__, len(worker.load_selection_rules(worker.WorkerConfig().app_dir/'app.py')))",
    ).strip()
    gate(smoke.endswith(" 3"), f"worker imports and reads the 3 selection rules ({smoke})")

    if not args.skip_tests:
        result = run(
            ["docker", "run", "--rm", "-v", f"{ROOT}:/repo:ro", "-w", "/repo",
             "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "python", listener,
             "-m", "unittest", "discover", "-s", "tests", "-p", "test_carpl_*.py", "-v"],
            check=False,
        )
        gate(result.returncode == 0, "CARPL unit tests pass inside the listener image")

    if args.no_save:
        print("\nImages built; bundle not saved (--no-save).")
        return 0

    print("\n=== 5. Deploy bundle ===")
    if not image_exists(REDIS_IMAGE):
        run(["docker", "pull", REDIS_IMAGE])
    bundle = args.output_dir / f"prostate-carpl-{version}"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    tar_path = bundle / "images.tar"
    images = [base, worker, listener, sender, REDIS_IMAGE]
    run(["docker", "save", "-o", str(tar_path), *images])
    tar_sha = sha256_file(tar_path)
    (bundle / "images.tar.sha256").write_text(f"{tar_sha}  images.tar\n", encoding="utf-8")
    shutil.copy2(CARPL / "docker-compose-carpl.yml", bundle / "docker-compose-carpl.yml")
    shutil.copy2(CARPL / ".env.example", bundle / ".env.example")
    shutil.copy2(CARPL / "README.md", bundle / "README.md")
    manifest = {
        "version": version,
        "built_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "git": git_state(),
        "images": {ref: inspect(ref, "{{.Id}}") for ref in images},
        "base_image": base,
        "monai_deploy_app_sdk": sdk,
        "pip_freeze_sha256": hashlib.sha256(base_freeze.encode()).hexdigest(),
        "images_tar_sha256": tar_sha,
        "images_tar_bytes": tar_path.stat().st_size,
    }
    (bundle / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nBundle: {bundle}")
    print(f"images.tar: {tar_path.stat().st_size / 1024**3:.2f} GB, sha256 {tar_sha}")
    if manifest["git"]["dirty"]:
        print("WARNING: built from a working tree with uncommitted changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
