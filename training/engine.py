from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
import monai
from monai.data import list_data_collate, worker_init_fn
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .cohort import clean_binary_organ_mask
from .datasets import (
    LesionDataset,
    build_organ_datasets,
    select_records,
    source_balanced_sampler,
)
from .export import export_organ_torchscript, load_weights, sha256_file
from .inference_tiler import lesion_tiled_predict
from .losses import get_loss
from .metrics import SegmentationMetrics
from .rrunet import RRUNet3D
from .transforms import get_organ_train_transforms, get_organ_val_transforms
from .utils import append_metrics_csv, load_yaml


@dataclass(frozen=True)
class TrainArtifacts:
    run_dir: str
    best_ckpt: str
    last_ckpt: str
    metrics_csv: str
    deploy_model: str


def build_model(cfg: Mapping[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    model_type = str(model_cfg["type"]).lower()
    if model_type == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=int(model_cfg["in_channels"]),
            out_channels=int(model_cfg["out_channels"]),
            channels=tuple(model_cfg["channels"]),
            strides=tuple(model_cfg["strides"]),
            num_res_units=int(model_cfg["num_res_units"]),
            act="PRELU",
            norm=str(model_cfg.get("norm", "BATCH")).upper(),
        )
    if model_type == "rrunet3d":
        return RRUNet3D(
            in_channels=int(model_cfg["in_channels"]),
            out_channels=int(model_cfg["out_channels"]),
            blocks_down=str(model_cfg["blocks_down"]),
            blocks_up=str(model_cfg["blocks_up"]),
            num_init_kernels=int(model_cfg["num_init_kernels"]),
            recurrent=bool(model_cfg["recurrent"]),
            residual=bool(model_cfg["residual"]),
            attention=bool(model_cfg["attention"]),
            se=bool(model_cfg["se"]),
            debug=False,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def _optimizer(cfg: Mapping[str, Any], model: torch.nn.Module):
    kind = str(cfg["optimizer"]["type"]).lower()
    kwargs = {
        "lr": float(cfg["train"]["lr"]),
        "weight_decay": float(cfg["train"]["weight_decay"]),
    }
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if kind == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer: {kind}")


def _scheduler(cfg: Mapping[str, Any], optimizer):
    scheduler_cfg = cfg["scheduler"]
    if str(scheduler_cfg["type"]).lower() != "cosine_anneal":
        raise ValueError(f"Unsupported scheduler: {scheduler_cfg['type']}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(scheduler_cfg.get("t_max", cfg["train"]["epochs"])),
        eta_min=float(scheduler_cfg.get("eta_min", 1e-6)),
    )


def _run_dir(cfg: Mapping[str, Any], task: str, fold: int | None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}-{os.getpid()}"
    base = Path(cfg["output"]["exp_dir"])
    if fold is not None:
        base = base / f"fold{fold}"
    path = base / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metric: float,
    task: str,
    fold: int | None,
    cfg: Mapping[str, Any],
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "best_dice": float(best_metric),
        "task": task,
        "fold": fold,
        "config": dict(cfg),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_resume(
    path: str | Path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    *,
    task: str,
    fold: int | None,
):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("task") not in {None, task}:
        raise ValueError(
            f"Resume checkpoint task={payload.get('task')} does not match {task}"
        )
    if payload.get("fold") not in {None, fold}:
        raise ValueError(
            f"Resume checkpoint fold={payload.get('fold')} does not match {fold}"
        )
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    if payload.get("scaler_state_dict"):
        scaler.load_state_dict(payload["scaler_state_dict"])
    return int(payload["epoch"]) + 1, float(
        payload.get("best_metric", payload.get("best_dice", -1.0))
    )


def _data_loaders(
    cfg: Mapping[str, Any], *, task: str, fold: int | None
) -> tuple[DataLoader, DataLoader, int, int]:
    dataset_cfg = cfg["dataset"]
    train_records = select_records(
        dataset_cfg["manifest"],
        dataset_cfg["split_csv"],
        task=task,
        partition="train",
        fold=fold,
    )
    val_records = select_records(
        dataset_cfg["manifest"],
        dataset_cfg["split_csv"],
        task=task,
        partition="val",
        fold=fold,
    )
    workers = int(dataset_cfg["num_workers"])
    source_weights = dataset_cfg.get("source_sampling_weights")
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "worker_init_fn": worker_init_fn,
    }
    if task == "organ":
        preprocessing = cfg["preprocess"]
        train_dataset, val_dataset = build_organ_datasets(
            train_records,
            val_records,
            train_transform=get_organ_train_transforms(
                preprocessing["spacing"],
                preprocessing["patch_size"],
                preprocessing["samples_per_case"],
                preprocessing["pos_to_neg_ratio"],
            ),
            val_transform=get_organ_val_transforms(preprocessing["spacing"]),
            cache_rate=float(dataset_cfg["cache_rate"]),
            num_workers=workers,
        )
        sampler = (
            source_balanced_sampler(train_records, source_weights=source_weights)
            if source_weights
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg["train"]["batch_size"]),
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=list_data_collate,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=list_data_collate,
            **loader_kwargs,
        )
    else:
        preprocessing = cfg["preprocess"]
        samples = int(preprocessing["samples_per_case"])
        train_dataset = LesionDataset(
            train_records,
            cache_dir=dataset_cfg["cache_dir"],
            spacing=preprocessing["spacing"],
            margin=int(preprocessing["roi_margin"]),
            patch_size=preprocessing["patch_size"],
            samples_per_case=samples,
            positive_fraction=float(preprocessing["positive_fraction"]),
            augment=bool(cfg.get("augment", {}).get("enabled", True)),
        )
        val_dataset = LesionDataset(
            val_records,
            cache_dir=dataset_cfg["cache_dir"],
            spacing=preprocessing["spacing"],
            margin=int(preprocessing["roi_margin"]),
            patch_size=None,
            samples_per_case=1,
            augment=False,
        )
        sampler = (
            source_balanced_sampler(
                train_records,
                repeats=samples,
                source_weights=source_weights,
            )
            if source_weights
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(cfg["train"]["batch_size"]),
            sampler=sampler,
            shuffle=sampler is None,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, **loader_kwargs
        )
    return train_loader, val_loader, len(train_records), len(val_records)


def _train_epoch(
    model,
    loader,
    loss_function,
    optimizer,
    scaler,
    device,
    *,
    accumulation_steps: int,
    log_interval: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    amp_enabled = scaler.is_enabled()
    started = time.monotonic()
    for step, batch in enumerate(loader, start=1):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(image)
            loss = loss_function(output, label)
            scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        if step % accumulation_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach())
        if step % log_interval == 0 or step == len(loader):
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = step / elapsed
            eta = (len(loader) - step) / max(rate, 1e-6)
            print(
                f"  train {step:>5}/{len(loader)}  "
                f"loss={total / step:.5f}  eta={eta / 60:.1f} min",
                flush=True,
            )
    return total / max(len(loader), 1)


def _validate_organ(model, loader, device, cfg) -> dict[str, float]:
    validation_cfg = cfg["validation"]
    inferer = SlidingWindowInferer(
        roi_size=tuple(validation_cfg["roi_size"]),
        sw_batch_size=int(validation_cfg["sw_batch_size"]),
        overlap=float(validation_cfg["overlap"]),
        mode="gaussian",
    )
    metrics = SegmentationMetrics("organ")
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits = inferer(batch["image"].to(device), model)
            prediction = torch.argmax(logits, dim=1)[0].cpu().numpy()
            target = batch["label"][0, 0].cpu().numpy() > 0
            cleaned, _ = clean_binary_organ_mask(prediction)
            metrics.update(cleaned, target)
    return metrics.compute()


def _validate_lesion(model, loader, device, cfg) -> dict[str, float]:
    threshold = float(cfg["validation"]["threshold"])
    metrics = SegmentationMetrics("lesion")
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            probability = lesion_tiled_predict(model, batch["image"].to(device))
            lesion_probability = probability[0, 1].cpu().numpy()
            organ = batch["organ"][0, 0].cpu().numpy() > 0
            prediction = (lesion_probability * organ) >= threshold
            target = batch["label"][0, 0].cpu().numpy() > 0
            metrics.update(prediction, target)
    return metrics.compute()


def train_from_config(
    config_path: str | Path,
    *,
    task: str,
    fold: int | None = None,
    initial_weights: str | Path | None = None,
    resume: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> TrainArtifacts:
    cfg = load_yaml(str(config_path))
    if task not in {"organ", "lesion"}:
        raise ValueError("task must be organ or lesion")
    if task == "lesion" and fold not in range(5):
        raise ValueError("lesion training requires --fold 0..4")
    if task == "organ":
        fold = None
    if output_dir is not None:
        cfg["output"]["exp_dir"] = str(output_dir)

    seed = int(cfg["train"]["seed"]) + (fold or 0)
    set_determinism(
        seed=seed,
        additional_settings=None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Task={task} fold={fold} device={device} seed={seed}", flush=True)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(
            f"GPU={properties.name} memory={properties.total_memory / 2**30:.1f} GiB "
            f"torch={torch.__version__} monai={monai.__version__}",
            flush=True,
        )
    train_loader, val_loader, train_cases, val_cases = _data_loaders(
        cfg, task=task, fold=fold
    )
    print(
        f"Locked cohort: train={train_cases} cases, val={val_cases} cases",
        flush=True,
    )

    model = build_model(cfg).to(device)
    if initial_weights is not None and resume is not None:
        raise ValueError("Use either initial_weights or resume, not both")
    if initial_weights is not None:
        metadata = load_weights(model, initial_weights, map_location=device)
        if metadata.get("task") not in {None, task}:
            raise ValueError(
                f"Initial checkpoint task={metadata.get('task')} does not match {task}"
            )
        if metadata.get("fold") not in {None, fold}:
            raise ValueError(
                f"Initial checkpoint fold={metadata.get('fold')} does not match {fold}"
            )
        print(f"Initialised strictly from {initial_weights}", flush=True)

    loss_function = get_loss(cfg["loss"], task=task)
    optimizer = _optimizer(cfg, model)
    scheduler = _scheduler(cfg, optimizer)
    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch, best_metric = 1, -1.0
    if resume is not None:
        start_epoch, best_metric = _load_resume(
            resume,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
            task=task,
            fold=fold,
        )
        print(f"Resumed from {resume} at epoch {start_epoch}", flush=True)

    run_dir = _run_dir(cfg, task, fold)
    metrics_csv = run_dir / "metrics.csv"
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )
    manifest_path = Path(cfg["dataset"]["manifest"]).resolve()
    split_path = Path(cfg["dataset"]["split_csv"]).resolve()
    provenance = {
        "task": task,
        "fold": fold,
        "seed": seed,
        "torch_version": torch.__version__,
        "monai_version": monai.__version__,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split_csv": str(split_path),
        "split_sha256": sha256_file(split_path),
        "initial_weights": str(Path(initial_weights).resolve())
        if initial_weights is not None
        else "",
        "initial_weights_sha256": sha256_file(Path(initial_weights))
        if initial_weights is not None
        else "",
        "resume": str(Path(resume).resolve()) if resume is not None else "",
        "resume_sha256": sha256_file(Path(resume)) if resume is not None else "",
    }
    (run_dir / "cohort_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    suffix = "" if fold is None else f"_fold{fold}"
    best_path = run_dir / f"model_best{suffix}.pth.tar"
    last_path = run_dir / f"model_last{suffix}.pth.tar"
    deploy_path = run_dir / "model.ts" if task == "organ" else best_path
    epochs = int(cfg["train"]["epochs"])
    started = time.monotonic()

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}", flush=True)
        train_loss = _train_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            scaler,
            device,
            accumulation_steps=int(cfg["train"]["grad_accum_steps"]),
            log_interval=int(cfg["train"]["log_interval"]),
        )
        validation: dict[str, float] = {}
        if epoch % int(cfg["train"]["val_interval"]) == 0:
            validation = (
                _validate_organ(model, val_loader, device, cfg)
                if task == "organ"
                else _validate_lesion(model, val_loader, device, cfg)
            )
            print(
                "  validation "
                + " ".join(f"{key}={value:.5f}" for key, value in validation.items()),
                flush=True,
            )
        scheduler.step()

        current_metric = validation.get("dice")
        if current_metric is not None and current_metric > best_metric:
            best_metric = current_metric
            _save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                task=task,
                fold=fold,
                cfg=cfg,
            )
            if task == "organ":
                export_organ_torchscript(model, deploy_path)
            print(f"  new best dice={best_metric:.5f}: {best_path}", flush=True)

        if epoch % int(cfg["output"]["save_last_interval"]) == 0 or epoch == epochs:
            _save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                task=task,
                fold=fold,
                cfg=cfg,
            )

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_minutes": (time.monotonic() - started) / 60.0,
            "val_dice": validation.get("dice", ""),
            "val_precision": validation.get("precision", ""),
            "val_recall": validation.get("recall", ""),
            "val_lesion_sensitivity": validation.get("lesion_sensitivity", ""),
            "val_false_positive_lesions_per_case": validation.get(
                "false_positive_lesions_per_case", ""
            ),
        }
        append_metrics_csv(str(metrics_csv), row)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/learning_rate", row["lr"], epoch)
        for name, value in validation.items():
            writer.add_scalar(f"validation/{name}", value, epoch)
        writer.flush()

    writer.close()
    summary = {
        "task": task,
        "fold": fold,
        "train_cases": train_cases,
        "validation_cases": val_cases,
        "best_dice": best_metric,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "deployment_model": str(deploy_path),
        "elapsed_minutes": (time.monotonic() - started) / 60.0,
        "manifest_sha256": provenance["manifest_sha256"],
        "split_sha256": provenance["split_sha256"],
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return TrainArtifacts(
        run_dir=str(run_dir),
        best_ckpt=str(best_path),
        last_ckpt=str(last_path),
        metrics_csv=str(metrics_csv),
        deploy_model=str(deploy_path),
    )


def train_organ_from_config(
    config_path: str | Path,
    pretrained_path: str | Path | None = None,
    *,
    resume: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> TrainArtifacts:
    return train_from_config(
        config_path,
        task="organ",
        initial_weights=pretrained_path,
        resume=resume,
        output_dir=output_dir,
    )


def train_lesion_from_config(
    config_path: str | Path,
    *,
    fold: int,
    pretrained_path: str | Path | None = None,
    resume: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> TrainArtifacts:
    return train_from_config(
        config_path,
        task="lesion",
        fold=fold,
        initial_weights=pretrained_path,
        resume=resume,
        output_dir=output_dir,
    )
