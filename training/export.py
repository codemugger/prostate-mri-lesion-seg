from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import torch


def load_weights(
    model: torch.nn.Module, path: str | Path, *, map_location: str | torch.device
) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".ts":
        payload = torch.jit.load(str(path), map_location=map_location)
        state_dict = payload.state_dict()
        metadata: dict = {}
    else:
        metadata = torch.load(
            str(path), map_location=map_location, weights_only=False
        )
        state_dict = metadata.get("state_dict", metadata)
    model.load_state_dict(state_dict, strict=True)
    return metadata


def export_organ_torchscript(
    model: torch.nn.Module,
    destination: str | Path,
    *,
    verify_shape: Sequence[int] = (32, 32, 16),
) -> Path:
    """Export and numerically guard the drop-in organ model.ts contract."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_model = copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        example = torch.randn(1, 1, *tuple(int(v) for v in verify_shape))
        eager = export_model(example)
        scripted = torch.jit.script(export_model)
        scripted_output = scripted(example)
        torch.testing.assert_close(
            eager, scripted_output, rtol=1e-4, atol=1e-5
        )
        if tuple(scripted_output.shape[:2]) != (1, 2):
            raise RuntimeError(
                f"Organ TorchScript output contract failed: {scripted_output.shape}"
            )
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.jit.save(scripted, str(temporary))
    temporary.replace(destination)
    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_deployment_bundle(
    *,
    destination: str | Path,
    organ_model: str | Path,
    lesion_models: Sequence[str | Path],
    classifier_model: str | Path,
) -> Path:
    """Create an inference-ready models tree without altering source models."""
    if len(lesion_models) != 5:
        raise ValueError("Exactly five lesion checkpoints are required")
    destination = Path(destination)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError(
                f"Deployment destination must be empty or absent: {destination}"
            )
    files: list[tuple[Path, Path]] = [
        (Path(organ_model), destination / "organ" / "model.ts"),
        (
            Path(classifier_model),
            destination / "classifier" / "model_best.pth.tar",
        ),
    ]
    files.extend(
        (
            Path(source),
            destination / f"fold{fold}" / f"model_best_fold{fold}.pth.tar",
        )
        for fold, source in enumerate(lesion_models)
    )
    missing = [source for source, _ in files if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing deployment source(s): " + ", ".join(str(path) for path in missing)
        )
    for source, target in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    manifest = {
        str(target.relative_to(destination)): {
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
        for _, target in files
    }
    (destination / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination
