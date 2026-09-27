#!/usr/bin/env python3
"""Safe, reproducible HIVE Docker entry points used by the training notebooks."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DEFAULT_HIVE_RUN = Path(
    "/home/ddsadmin/Documents/Prostate HIVE Inference 4 jun 2026 - highb fix"
)
DEFAULT_IMAGE = "prostate-mri-training:latest"
REQUIRED_WEIGHTS = [
    "organ/model.ts",
    *(f"fold{fold}/model_best_fold{fold}.pth.tar" for fold in range(5)),
    "classifier/model_best.pth.tar",
]


def _run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    print(f"\n$ {shlex.join(str(part) for part in command)}", flush=True)
    subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd is not None else None,
        check=True,
    )


def _assert_gpu_idle(gpu_id: int) -> None:
    """Refuse to launch on a GPU that already has a compute process."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu_id),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = result.stdout.strip()
    if processes:
        raise RuntimeError(
            f"Physical GPU {gpu_id} already has compute process(es):\n{processes}\n"
            "Wait for it to become idle or explicitly select another permitted GPU."
        )
    info = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            str(gpu_id),
            "--query-gpu=name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"GPU_IDLE_CHECK=PASS physical_gpu={gpu_id} {info}", flush=True)


def _container_path(path: Path, *, project: Path, hive_run: Path) -> str:
    resolved = path.expanduser().resolve()
    mappings = (
        (project.resolve(), Path("/workspace")),
        ((hive_run / "output").resolve(), Path("/hive-output")),
        ((hive_run / "models").resolve(), Path("/pretrained-models")),
    )
    for host_root, container_root in mappings:
        try:
            relative = resolved.relative_to(host_root)
        except ValueError:
            continue
        return str(container_root / relative)
    raise ValueError(
        f"{resolved} is outside the mounted project, HIVE output, and model roots"
    )


def _docker_base(
    *,
    project: Path,
    hive_run: Path,
    image: str,
    gpu_id: int | None,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--shm-size=16g",
    ]
    if gpu_id is not None:
        # Docker exposes one physical GPU; CUDA sees that device as logical cuda:0.
        command.extend(
            [
                "--gpus",
                f"device={gpu_id}",
                "--env",
                "CUDA_VISIBLE_DEVICES=0",
                "--env",
                "HIVE_REQUIRE_SINGLE_GPU=1",
            ]
        )
    command.extend(
        [
            "--volume",
            f"{project.resolve()}:/workspace",
            "--volume",
            f"{(hive_run / 'output').resolve()}:/hive-output:ro",
            "--volume",
            f"{(hive_run / 'models').resolve()}:/pretrained-models:ro",
            "--workdir",
            "/workspace",
            image,
        ]
    )
    return command


def _run_container(
    inner_command: Sequence[str],
    *,
    project: Path,
    hive_run: Path,
    image: str,
    gpu_id: int | None = None,
) -> None:
    _run(
        [
            *_docker_base(
                project=project,
                hive_run=hive_run,
                image=image,
                gpu_id=gpu_id,
            ),
            *inner_command,
        ],
        cwd=project,
    )


@contextmanager
def _single_job_lock(project: Path, gpu_id: int) -> Iterator[None]:
    lock_dir = project / "experiments"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ".hive_single_gpu_training.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another HIVE preparation/training job already holds {lock_path}. "
                "Do not prepare data or run organ/lesion jobs concurrently."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            f"pid={os.getpid()} gpu_id={gpu_id} started_utc="
            f"{datetime.now(timezone.utc).isoformat()}\n"
        )
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_layout(project: Path, hive_run: Path) -> None:
    required_project = [
        project / "audit_assignments.csv",
        project / "configs" / "organ.yaml",
        project / "configs" / "lesion.yaml",
        project / "data" / "preprocessed",
        project / "training" / "engine.py",
    ]
    missing = [str(path) for path in required_project if not path.exists()]
    output_root = hive_run / "output"
    model_root = hive_run / "models"
    if not output_root.is_dir():
        missing.append(str(output_root))
    missing.extend(
        str(model_root / relative)
        for relative in REQUIRED_WEIGHTS
        if not (model_root / relative).is_file()
    )
    if missing:
        raise FileNotFoundError("Required HIVE input(s) missing:\n- " + "\n- ".join(missing))


def _audit_counts(project: Path) -> tuple[int, int, int]:
    with (project / "audit_assignments.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    organ = sum(
        row["Reviewed"] == "Y"
        and row["Q1: Prostate contour acceptable"] == "Y"
        and row["Q4: Include in training set"] == "Y"
        for row in rows
    )
    lesion = sum(
        row["Reviewed"] == "Y"
        and row["Q2: All suspicious lesions contoured"] == "Y"
        and row["Q3: No incorrectly contoured regions"] == "Y"
        and row["Q4: Include in training set"] == "Y"
        for row in rows
    )
    return len(rows), organ, lesion


def preflight(project: Path, hive_run: Path, image: str, gpu_id: int) -> None:
    _validate_layout(project, hive_run)
    _run(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "IMAGE={{.RepoTags}} ID={{.Id}} SIZE={{.Size}}",
            image,
        ],
        cwd=project,
    )
    assigned, organ, lesion = _audit_counts(project)
    output_cases = sum(path.is_dir() for path in (hive_run / "output").iterdir())
    print(
        "\nHost inputs:"
        f"\n  HIVE output root: {(hive_run / 'output').resolve()}"
        f"\n  HIVE output case directories: {output_cases:,}"
        f"\n  Audit rows: {assigned:,}"
        f"\n  Audit-policy organ candidates: {organ:,}"
        f"\n  Audit-policy lesion candidates: {lesion:,}"
        f"\n  Transfer-weight files: {len(REQUIRED_WEIGHTS)}",
        flush=True,
    )
    check_code = """
import torch
count = torch.cuda.device_count()
print(f"CUDA_VISIBLE_DEVICE_COUNT={count}")
assert count == 1, f"Safety check failed: expected exactly 1 visible GPU, got {count}"
print(f"GPU_0={torch.cuda.get_device_name(0)}")
"""
    with _single_job_lock(project, gpu_id):
        _assert_gpu_idle(gpu_id)
        _run_container(
            ["python", "-c", check_code],
            project=project,
            hive_run=hive_run,
            image=image,
            gpu_id=gpu_id,
        )
    weights_code = """
from pathlib import Path
import yaml
from training.engine import _validate_config_contract, build_model
from training.export import load_weights

organ_cfg = yaml.safe_load(Path("configs/organ.yaml").read_text())
_validate_config_contract(organ_cfg, task="organ")
organ = build_model(organ_cfg)
load_weights(organ, "/pretrained-models/organ/model.ts", map_location="cpu")
lesion_cfg = yaml.safe_load(Path("configs/lesion.yaml").read_text())
_validate_config_contract(lesion_cfg, task="lesion")
for fold in range(5):
    model = build_model(lesion_cfg)
    load_weights(
        model,
        f"/pretrained-models/fold{fold}/model_best_fold{fold}.pth.tar",
        map_location="cpu",
    )
print("STRICT_TRANSFER_WEIGHT_CHECK=PASS")
"""
    _run_container(
        ["python", "-c", weights_code],
        project=project,
        hive_run=hive_run,
        image=image,
    )
    print("\nPREFLIGHT=PASS (one visible GPU; all transfer weights strict-load)", flush=True)


def prepare(project: Path, hive_run: Path, image: str, gpu_id: int) -> None:
    _validate_layout(project, hive_run)
    with _single_job_lock(project, gpu_id):
        _run_container(
            [
                "python",
                "scripts/prepare_training_data.py",
                "--audit-csv",
                "/workspace/audit_assignments.csv",
                "--sgh-output-root",
                "/hive-output",
                "--prostatex-root",
                "/workspace/data/preprocessed",
                "--output-dir",
                "/workspace/artifacts/training",
            ],
            project=project,
            hive_run=hive_run,
            image=image,
        )
    validate_code = """
from training.datasets import select_records
tasks = (("organ", None), *(("lesion", fold) for fold in range(5)))
for task, fold in tasks:
    split = f"artifacts/training/{task}_splits.csv"
    train = select_records(
        "artifacts/training/training_manifest.csv", split,
        task=task, partition="train", fold=fold,
    )
    val = select_records(
        "artifacts/training/training_manifest.csv", split,
        task=task, partition="val", fold=fold,
    )
    print(f"{task} fold={fold}: train={len(train)} val={len(val)}")
print("LOCKED_SPLIT_VALIDATION=PASS")
"""
    _run_container(
        ["python", "-c", validate_code],
        project=project,
        hive_run=hive_run,
        image=image,
    )
    print("\nPREPARATION=PASS", flush=True)


def _require_manifest(project: Path) -> None:
    required = [
        project / "artifacts" / "training" / "training_manifest.csv",
        project / "artifacts" / "training" / "organ_splits.csv",
        project / "artifacts" / "training" / "lesion_splits.csv",
        project / "artifacts" / "training" / "audit_report.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Prepare the cohort before training; missing:\n- " + "\n- ".join(missing)
        )


def train_organ(
    project: Path,
    hive_run: Path,
    image: str,
    gpu_id: int,
    resume: Path | None,
) -> None:
    _validate_layout(project, hive_run)
    _require_manifest(project)
    inner = [
        "python",
        "-m",
        "training.train_organ",
        "--config",
        "configs/organ.yaml",
    ]
    if resume is None:
        inner.extend(["--initial-weights", "/pretrained-models/organ/model.ts"])
    else:
        if not resume.is_file():
            raise FileNotFoundError(resume)
        inner.extend(
            ["--resume", _container_path(resume, project=project, hive_run=hive_run)]
        )
    with _single_job_lock(project, gpu_id):
        _assert_gpu_idle(gpu_id)
        _run_container(
            inner,
            project=project,
            hive_run=hive_run,
            image=image,
            gpu_id=gpu_id,
        )


def train_lesion(
    project: Path,
    hive_run: Path,
    image: str,
    gpu_id: int,
    fold: int,
    resume: Path | None,
) -> None:
    _validate_layout(project, hive_run)
    _require_manifest(project)
    inner = [
        "python",
        "-m",
        "training.train_lesion",
        "--config",
        "configs/lesion.yaml",
        "--fold",
        str(fold),
    ]
    if resume is None:
        inner.extend(
            [
                "--initial-weights",
                f"/pretrained-models/fold{fold}/model_best_fold{fold}.pth.tar",
            ]
        )
    else:
        if not resume.is_file():
            raise FileNotFoundError(resume)
        inner.extend(
            ["--resume", _container_path(resume, project=project, hive_run=hive_run)]
        )
    with _single_job_lock(project, gpu_id):
        _assert_gpu_idle(gpu_id)
        _run_container(
            inner,
            project=project,
            hive_run=hive_run,
            image=image,
            gpu_id=gpu_id,
        )


def _read_metrics(path: Path) -> tuple[str, str, str]:
    if not path.is_file():
        return "-", "-", "-"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return "-", "-", "-"
    dices = [
        float(row["val_dice"])
        for row in rows
        if str(row.get("val_dice", "")).strip()
    ]
    return rows[-1]["epoch"], f"{max(dices):.5f}" if dices else "-", rows[-1].get(
        "elapsed_minutes", "-"
    )


def _iter_run_dirs(project: Path) -> Iterable[tuple[str, Path]]:
    organ_root = project / "experiments" / "organ"
    if organ_root.is_dir():
        for run in sorted(path for path in organ_root.iterdir() if path.is_dir()):
            yield "organ", run
    lesion_root = project / "experiments" / "lesion"
    if lesion_root.is_dir():
        for fold in range(5):
            fold_root = lesion_root / f"fold{fold}"
            if fold_root.is_dir():
                for run in sorted(path for path in fold_root.iterdir() if path.is_dir()):
                    yield f"lesion/fold{fold}", run


def status(project: Path) -> None:
    report_path = project / "artifacts" / "training" / "audit_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(
            "Prepared cohort:"
            f"\n  manifest cases: {report['manifest_cases']:,}"
            f"\n  organ eligible: {report['organ_eligible']:,}"
            f"\n  lesion eligible: {report['lesion_eligible']:,}"
        )
    else:
        print("Prepared cohort: NOT FOUND")

    rows = list(_iter_run_dirs(project))
    if not rows:
        print("\nTraining runs: none yet")
        return
    print("\nTraining runs:")
    print(f"{'task':<16} {'run':<25} {'epoch':>6} {'best dice':>10} {'state':>12}")
    for task, run in rows:
        epoch, best, _ = _read_metrics(run / "metrics.csv")
        if (run / "run_summary.json").is_file():
            state = "complete"
        elif (run / "FAILED_RUN_NOTE.md").is_file():
            state = "failed"
        else:
            state = "incomplete"
        print(f"{task:<16} {run.name:<25} {epoch:>6} {best:>10} {state:>12}")


def _best_completed_run(project: Path, task: str, fold: int | None) -> dict:
    root = project / "experiments" / task
    if fold is not None:
        root = root / f"fold{fold}"
    candidates: list[dict] = []
    if root.is_dir():
        for summary_path in root.glob("*/run_summary.json"):
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["_summary_path"] = str(summary_path)
            candidates.append(payload)
    if not candidates:
        suffix = "" if fold is None else f" fold {fold}"
        raise FileNotFoundError(f"No completed {task}{suffix} run is available")
    return max(candidates, key=lambda item: float(item["best_dice"]))


def _summary_host_path(project: Path, value: str) -> Path:
    """Translate engine paths recorded inside /workspace back to the host project."""
    path = Path(value)
    if not path.is_absolute():
        return project / path
    try:
        return project / path.relative_to("/workspace")
    except ValueError:
        return path


def package_best(
    project: Path,
    hive_run: Path,
    image: str,
    destination: Path | None,
) -> Path:
    organ = _best_completed_run(project, "organ", None)
    lesions = [_best_completed_run(project, "lesion", fold) for fold in range(5)]
    if destination is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = project / "training_exports" / f"deployment_{stamp}"
    destination = destination.resolve()
    output_container = _container_path(
        destination, project=project, hive_run=hive_run
    )
    organ_container = _container_path(
        _summary_host_path(project, organ["deployment_model"]),
        project=project,
        hive_run=hive_run,
    )
    inner = [
        "python",
        "scripts/package_models.py",
        "--output",
        output_container,
        "--organ",
        organ_container,
        "--classifier",
        "/pretrained-models/classifier/model_best.pth.tar",
    ]
    for payload in lesions:
        inner.extend(
            [
                "--lesion",
                _container_path(
                    _summary_host_path(project, payload["best_checkpoint"]),
                    project=project,
                    hive_run=hive_run,
                ),
            ]
        )
    _run_container(
        inner,
        project=project,
        hive_run=hive_run,
        image=image,
    )
    print(f"\nEXPORT_BUNDLE={destination}", flush=True)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-GPU-safe HIVE preparation and training commands."
    )
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--hive-run", type=Path, default=DEFAULT_HIVE_RUN)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--gpu-id", type=int, default=0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("prepare")
    organ = subparsers.add_parser("organ")
    organ.add_argument("--resume", type=Path)
    lesion = subparsers.add_parser("lesion")
    lesion.add_argument("--fold", type=int, choices=range(5), required=True)
    lesion.add_argument("--resume", type=Path)
    subparsers.add_parser("status")
    package = subparsers.add_parser("package")
    package.add_argument("--destination", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project = args.project.expanduser().resolve()
    hive_run = args.hive_run.expanduser().resolve()
    if args.command == "preflight":
        preflight(project, hive_run, args.image, args.gpu_id)
    elif args.command == "prepare":
        prepare(project, hive_run, args.image, args.gpu_id)
    elif args.command == "organ":
        resume = args.resume.expanduser().resolve() if args.resume else None
        train_organ(project, hive_run, args.image, args.gpu_id, resume)
    elif args.command == "lesion":
        resume = args.resume.expanduser().resolve() if args.resume else None
        train_lesion(
            project,
            hive_run,
            args.image,
            args.gpu_id,
            args.fold,
            resume,
        )
    elif args.command == "status":
        status(project)
    elif args.command == "package":
        destination = (
            args.destination.expanduser().resolve() if args.destination else None
        )
        package_best(project, hive_run, args.image, destination)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
