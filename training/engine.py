from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from monai.data import list_data_collate, pad_list_data_collate
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.inferers import SlidingWindowInferer
from monai.transforms import AsDiscrete, KeepLargestConnectedComponent
from monai.utils import set_determinism

from .datasets import (
    read_subject_ids,
    list_subject_ids_from_dirs,
    list_lesion_subject_ids_from_dirs,
    OrganDatasetPaths,
    LesionDatasetPaths,
    build_organ_items,
    build_lesion_items,
    create_monai_datasets,
)
from .transforms import (
    get_organ_train_transforms,
    get_organ_val_transforms,
    get_lesion_train_transforms,
    get_lesion_val_transforms,
    LabelToOneHot,
    PredToOneHot,
)
from .losses import get_loss
from .utils import load_yaml, ensure_dir, append_metrics_csv, get_device, get_next_experiment_number
from torch.utils.tensorboard import SummaryWriter
from .rrunet import RRUNet3D


@dataclass
class TrainArtifacts:
    best_ckpt: str
    last_ckpt: str
    metrics_csv: str


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    mcfg = cfg["model"]
    model_type = mcfg.get("type", "unet").lower()
    if model_type == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=mcfg["in_channels"],
            out_channels=mcfg["out_channels"],
            channels=mcfg["channels"],
            strides=mcfg["strides"],
            num_res_units=mcfg["num_res_units"],
            act="PReLU",
            norm="instance",
        )
    if model_type == "rrunet3d":
        return RRUNet3D(
            in_channels=mcfg["in_channels"],
            out_channels=mcfg["out_channels"],
            blocks_down=mcfg.get("blocks_down", "1,2,3,4"),
            blocks_up=mcfg.get("blocks_up", "3,2,1"),
            num_init_kernels=mcfg.get("num_init_kernels", 32),
            recurrent=mcfg.get("recurrent", False),
            residual=mcfg.get("residual", True),
            attention=mcfg.get("attention", False),
            se=mcfg.get("se", False),
            debug=False,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    opt_type = cfg["optimizer"]["type"].lower()
    lr = cfg["train"]["lr"]
    wd = cfg["train"]["weight_decay"]
    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {opt_type}")


def build_scheduler(cfg: Dict[str, Any], optimizer):
    scfg = cfg["scheduler"]
    s_type = scfg["type"].lower()
    if s_type == "cosine_anneal":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scfg.get("t_max", 50), eta_min=scfg.get("eta_min", 1.0e-6)
        )
    return None


def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler: torch.cuda.amp.GradScaler, log_interval: int):
    model.train()
    running = 0.0
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += float(loss.detach().cpu())
        if step % log_interval == 0:
            avg = running / step
            print(f"  step {step}/{len(loader)} - loss: {avg:.4f}")
    return running / max(len(loader), 1)


def validate(model, loader, device, inferer, post_pred, post_label, include_background: bool = False):
    """
    Validate model on validation set.
    
    Args:
        model: Model to validate
        loader: Validation data loader
        device: Device to run on
        inferer: Sliding window inferer
        post_pred: Post-processing for predictions
        post_label: Post-processing for labels
        include_background: Whether to include background class in dice calculation.
                           For organ segmentation, typically False (focus on TZ/PZ).
    
    Returns:
        mean_dice: Mean dice score (across foreground classes if include_background=False)
    """
    model.eval()
    # Use "mean_batch" to get per-class dice, then compute mean manually
    # This gives us more control and matches p158-inspiration approach
    dice_metric = DiceMetric(include_background=include_background, reduction="mean_batch")
    batch_count = 0
    
    lcc = KeepLargestConnectedComponent(applied_labels=[1])
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Debug: Check RAW label values BEFORE one-hot conversion (first 3 batches)
            if batch_count < 3:
                # Labels come from transforms as [C, H, W, D] where C=1 for multi-class labels
                raw_label = labels
                if raw_label.dim() == 4 and raw_label.shape[0] == 1:
                    # Remove channel dimension to get [H, W, D]
                    raw_label_3d = raw_label.squeeze(0)
                else:
                    raw_label_3d = raw_label
                
                raw_unique = torch.unique(raw_label_3d).cpu().tolist()
                raw_counts = torch.bincount(raw_label_3d.flatten().long(), minlength=3).cpu().tolist()
                # print(f"  [DEBUG] Batch {batch_count} - RAW label (before one-hot) unique: {raw_unique}, counts: {raw_counts}")
                # print(f"  [DEBUG] Batch {batch_count} - Label shape: {labels.shape}, Label dtype: {labels.dtype}, Label min/max: {labels.min().item():.2f}/{labels.max().item():.2f}")
            
            logits = inferer(inputs=images, network=model)
            preds = post_pred(logits)
            preds = lcc(preds)
            labs = post_label(labels)
            
            # Skip true-negative batches (both GT and pred are empty / background only)
            pred_argmax = preds.argmax(dim=1)  # [B,H,W,D]
            lab_argmax = labs.argmax(dim=1)    # [B,H,W,D]
            pred_has_fg = (pred_argmax != 0).any(dim=(1, 2, 3))
            lab_has_fg = (lab_argmax != 0).any(dim=(1, 2, 3))
            keep_mask = (pred_has_fg | lab_has_fg)
            if not keep_mask.any():
                batch_count += 1
                continue  # both empty → skip this batch
            # If mixed, keep only items that have foreground in either GT or pred
            if (~keep_mask).any():
                preds = preds[keep_mask]
                labs = labs[keep_mask]

            # Debug: Check predictions and one-hot labels (first 3 batches)
            if batch_count < 3:
                # print(f"  [DEBUG] Batch {batch_count} - Logits shape: {tuple(logits.shape)}")
                pred_argmax = preds.argmax(dim=1)
                label_argmax = labs.argmax(dim=1)
                pred_unique = torch.unique(pred_argmax).cpu().tolist()
                label_unique = torch.unique(label_argmax).cpu().tolist()
                pred_counts = torch.bincount(pred_argmax.flatten(), minlength=3).cpu().tolist()
                label_counts = torch.bincount(label_argmax.flatten(), minlength=3).cpu().tolist()
                # print(f"  [DEBUG] Batch {batch_count} - After one-hot: Pred classes: {pred_unique}, Label classes: {label_unique}")
                # print(f"  [DEBUG] Batch {batch_count} - After one-hot: Pred counts: {pred_counts}, Label counts: {label_counts}")
                # print(f"  [DEBUG] Batch {batch_count} - Pred shape: {preds.shape}, Label shape: {labs.shape}")
            
            
            # Debug: Check if predictions are changing (only first batch)
            if batch_count == 0:
                pred_argmax = preds.argmax(dim=1)
                label_argmax = labs.argmax(dim=1)
                pred_unique = torch.unique(pred_argmax).cpu().tolist()
                label_unique = torch.unique(label_argmax).cpu().tolist()
                pred_counts = torch.bincount(pred_argmax.flatten(), minlength=3).cpu().tolist()
                label_counts = torch.bincount(label_argmax.flatten(), minlength=3).cpu().tolist()
                # print(f"  [DEBUG] Batch 0 - Pred classes: {pred_unique}, Label classes: {label_unique}")
                # print(f"  [DEBUG] Batch 0 - Pred counts: {pred_counts}, Label counts: {label_counts}")
            
            dice_metric(y_pred=preds, y=labs)
            batch_count += 1
    
    # Get per-class dice scores (one per class)
    per_class_dice = dice_metric.aggregate()
    
    # Debug: Print per-class dice
    # print(f"  [DEBUG] Per-class dice: {per_class_dice.cpu().tolist()}")
    
    # Filter out NaN values (classes that don't exist in validation set)
    valid_dice = per_class_dice[~torch.isnan(per_class_dice)]
    
    if len(valid_dice) == 0:
        # No valid dice scores (all NaN) - return 0.0 as fallback
        mean_dice = 0.0
        print(f"  [DEBUG] All dice scores are NaN, returning 0.0")
    else:
        # Compute mean across valid classes (TZ and PZ, excluding background)
        mean_dice = valid_dice.mean().item()
        # print(f"  [DEBUG] Valid dice classes: {len(valid_dice)}, Mean dice: {mean_dice:.4f}")
    
    dice_metric.reset()
    return mean_dice
def train_organ_from_config(config_path: str, pretrained_path: str = None) -> TrainArtifacts:
    cfg = load_yaml(config_path)

    # Determinism
    set_determinism(seed=cfg["train"]["seed"]) if cfg["train"]["deterministic"] else None
    device = get_device()
    print(f"Using device: {device}")

    # Data: either read predefined splits or scan all and split internally
    train_ids = None
    val_ids = None
    use_csv = bool(cfg["dataset"].get("train_split_csv")) and bool(cfg["dataset"].get("val_split_csv"))
    if use_csv:
        train_ids = read_subject_ids(cfg["dataset"]["train_split_csv"])
        val_ids = read_subject_ids(cfg["dataset"]["val_split_csv"])
    else:
        # Scan directories and split
        all_ids = list_subject_ids_from_dirs(cfg["dataset"]["t2_dir"], cfg["dataset"]["label_dir"])
        # Deterministic shuffle
        import random
        rng = random.Random(cfg["train"]["seed"])
        rng.shuffle(all_ids)
        split_ratio = float(cfg["dataset"].get("split_ratio", 0.8))
        split_index = max(1, min(len(all_ids) - 1, int(round(len(all_ids) * split_ratio))))
        train_ids = all_ids[:split_index]
        val_ids = all_ids[split_index:]
    paths = OrganDatasetPaths(
        t2_dir=cfg["dataset"]["t2_dir"],
        label_dir=cfg["dataset"]["label_dir"],
    )
    train_items = build_organ_items(train_ids, paths)
    val_items = build_organ_items(val_ids, paths)
    train_transforms = get_organ_train_transforms(
        spacing=cfg["preprocess"]["spacing"],
        roi_size=cfg["preprocess"]["roi_size"],
        samples_per_image=cfg["preprocess"]["samples_per_image"],
        pos_to_neg=cfg["preprocess"]["pos_to_neg_ratio"],
    )
    val_transforms = get_organ_val_transforms(spacing=cfg["preprocess"]["spacing"])
    # Reduce worker count for lesion training to avoid worker OOM/kill
    worker_count = max(cfg["dataset"]["num_workers"], 4)

    train_ds, val_ds = create_monai_datasets(
        train_items,
        val_items,
        train_transforms,
        val_transforms,
        cache_rate=cfg["dataset"]["cache_rate"],
        num_workers=worker_count,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["dataset"]["num_workers"],
        collate_fn=list_data_collate, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=cfg["dataset"]["num_workers"],
        collate_fn=list_data_collate, pin_memory=torch.cuda.is_available()
    )

    # Model / loss / optimizer / scheduler
    model = build_model(cfg).to(device)

    if pretrained_path is not None:
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")
        if pretrained_path.endswith(".ts"):
            ts_model = torch.jit.load(pretrained_path, map_location=device)
            model.load_state_dict(ts_model.state_dict())
            del ts_model
        else:
            ckpt = torch.load(pretrained_path, map_location=device)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            del ckpt
        print(f"Loaded pretrained weights from: {pretrained_path}")

    loss_fn = get_loss(cfg["loss"])
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # Inferer and post processing for validation
    inferer = SlidingWindowInferer(
        roi_size=tuple(cfg["validation"]["inferer_roi_size"]),
        sw_batch_size=cfg["validation"]["inferer_sw_batch_size"],
        overlap=cfg["validation"]["inferer_overlap"],
        mode="gaussian",
    )
    # Robust conversions that preserve [B, C, ...] shape
    post_pred = PredToOneHot(num_classes=cfg["model"]["out_channels"])
    post_label = LabelToOneHot(num_classes=cfg["model"]["out_channels"])

    # Output dirs
    base_exp_dir = cfg["output"]["exp_dir"]
    ensure_dir(base_exp_dir)
    # Determine run directory (per-run subdir + experiment number)
    if bool(cfg["output"].get("per_run_subdir", True)):
        exp_no = get_next_experiment_number(base_exp_dir)
        run_dir = os.path.join(base_exp_dir, f"exp-{exp_no:03d}")
    else:
        exp_no = 1
        run_dir = base_exp_dir
    ensure_dir(run_dir)
    # TensorBoard writer
    writer = None
    if bool(cfg["output"].get("tensorboard", True)):
        tb_subdir = cfg["output"].get("tb_subdir", "tb")
        tb_dir = os.path.join(run_dir, tb_subdir)
        ensure_dir(tb_dir)
        writer = SummaryWriter(log_dir=tb_dir)
    # Checkpoint and metrics paths (file names templated with exp/epoch/dice)
    best_base = cfg["output"]["best_checkpoint_name"]
    last_base = cfg["output"]["last_checkpoint_name"]
    metrics_csv = os.path.join(run_dir, cfg["output"]["metrics_csv"])

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"] and torch.cuda.is_available()))
    best_dice = -1.0
    best_ckpt = None
    last_ckpt = None
    prev_best_ckpt = None  # Track previous best checkpoint to delete it

    # Get checkpoint saving interval (default: save every epoch for backward compatibility)
    save_last_interval = cfg["output"].get("save_last_interval", 1)

    epochs = cfg["train"]["epochs"]
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, cfg["train"]["log_interval"]
        )
        # TensorBoard: train loss and LR
        if writer is not None:
            writer.add_scalar("train/loss", float(train_loss), epoch)
            writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), epoch)
        val_dice = None
        if epoch % cfg["train"]["val_interval"] == 0:
            # For organ segmentation, exclude background from dice calculation
            # (we care about TZ and PZ segmentation quality, not background)
            include_bg = cfg.get("validation", {}).get("include_background", False)
            val_dice = validate(model, val_loader, device, inferer, post_pred, post_label, include_background=include_bg)
            print(f"  val mean dice: {val_dice:.4f}")
            if writer is not None and val_dice is not None:
                writer.add_scalar("val/dice", float(val_dice), epoch)

        if scheduler is not None:
            scheduler.step()

        # Save last checkpoint only every N epochs (or at final epoch)
        if epoch % save_last_interval == 0 or epoch == epochs:
            # Remove previous last checkpoint if it exists (to save disk space)
            if last_ckpt is not None and os.path.exists(last_ckpt):
                try:
                    os.remove(last_ckpt)
                except OSError:
                    pass  # Ignore errors if file doesn't exist or can't be deleted
            # exp-XXX_last_model_epoch_200.pth
            last_ckpt = os.path.join(run_dir, f"exp-{exp_no:03d}_{last_base}_epoch_{epoch}.pth")
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, last_ckpt)

        # Save best checkpoint only when dice improves (and remove old best)
        if val_dice is not None and val_dice > best_dice:
            # Remove previous best checkpoint if it exists
            if prev_best_ckpt is not None and os.path.exists(prev_best_ckpt):
                try:
                    os.remove(prev_best_ckpt)
                except OSError:
                    pass  # Ignore errors if file doesn't exist or can't be deleted
            
            best_dice = val_dice
            # exp-XXX_best_model_epoch_200_dice_0.2615.pth
            best_ckpt = os.path.join(run_dir, f"exp-{exp_no:03d}_{best_base}_epoch_{epoch}_dice_{best_dice:.4f}.pth")
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_dice": best_dice}, best_ckpt)
            prev_best_ckpt = best_ckpt  # Track for deletion on next improvement

        # Log
        append_metrics_csv(
            metrics_csv,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_dice": float(val_dice) if val_dice is not None else "",
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    print("Training complete.")
    if writer is not None:
        writer.flush()
        writer.close()
    return TrainArtifacts(best_ckpt=best_ckpt, last_ckpt=last_ckpt, metrics_csv=metrics_csv)


def train_lesion_from_config(config_path: str) -> TrainArtifacts:
    """
    Train lesion segmentation model from config file.
    Similar to train_organ_from_config but handles multi-modal inputs (T2, ADC, HIGHB)
    and ROI cropping from organ masks.
    """
    cfg = load_yaml(config_path)

    # Determinism
    set_determinism(seed=cfg["train"]["seed"]) if cfg["train"]["deterministic"] else None
    device = get_device()
    print(f"Using device: {device}")

    # Data: either read predefined splits or scan all and split internally
    train_ids = None
    val_ids = None
    use_csv = bool(cfg["dataset"].get("train_split_csv")) and bool(cfg["dataset"].get("val_split_csv"))
    if use_csv:
        train_ids = read_subject_ids(cfg["dataset"]["train_split_csv"])
        val_ids = read_subject_ids(cfg["dataset"]["val_split_csv"])
    else:
        # Scan directories and split
        all_ids = list_lesion_subject_ids_from_dirs(
            cfg["dataset"]["t2_dir"],
            cfg["dataset"]["organ_mask_dir"],
            cfg["dataset"]["lesion_mask_dir"],
        )
        # Deterministic shuffle
        import random
        rng = random.Random(cfg["train"]["seed"])
        rng.shuffle(all_ids)
        split_ratio = float(cfg["dataset"].get("split_ratio", 0.8))
        split_index = max(1, min(len(all_ids) - 1, int(round(len(all_ids) * split_ratio))))
        train_ids = all_ids[:split_index]
        val_ids = all_ids[split_index:]
    
    paths = LesionDatasetPaths(
        t2_dir=cfg["dataset"]["t2_dir"],
        adc_dir=cfg["dataset"]["adc_dir"],
        highb_dir=cfg["dataset"]["highb_dir"],
        organ_mask_dir=cfg["dataset"]["organ_mask_dir"],
        lesion_mask_dir=cfg["dataset"]["lesion_mask_dir"],
    )
    
    # Create temporary directory for resampled ADC/HIGHB
    import tempfile
    temp_resample_dir = tempfile.mkdtemp(prefix="lesion_resampled_")
    
    try:
        train_items = build_lesion_items(train_ids, paths, temp_resample_dir)
        val_items = build_lesion_items(val_ids, paths, temp_resample_dir)
    finally:
        # Clean up temporary directory (optional - can keep for debugging)
        pass
    
    train_transforms = get_lesion_train_transforms(
        spacing=cfg["preprocess"]["spacing"],
        roi_size=cfg["preprocess"]["roi_size"],
        roi_margin=cfg["preprocess"]["roi_margin"],
        samples_per_image=cfg["preprocess"]["samples_per_image"],
        pos_to_neg=cfg["preprocess"]["pos_to_neg_ratio"],
        augment=cfg.get("augment", {}),
    )
    val_transforms = get_lesion_val_transforms(
        spacing=cfg["preprocess"]["spacing"],
        roi_margin=cfg["preprocess"]["roi_margin"],
        roi_size=cfg["preprocess"]["roi_size"],
    )
    
    worker_count = max(cfg["dataset"]["num_workers"], 4)

    train_ds, val_ds = create_monai_datasets(
        train_items,
        val_items,
        train_transforms,
        val_transforms,
        cache_rate=cfg["dataset"]["cache_rate"],
        num_workers=worker_count,
    )
    # Use pad_list_data_collate for lesion training because CropROIFromOrganMask produces variable-sized outputs
    # that need to be padded before 
    # ing (RandSpatialCropSamplesd returns fixed-size crops, but from variable inputs)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=worker_count,
        collate_fn=pad_list_data_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=worker_count,
        collate_fn=pad_list_data_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    # Model / loss / optimizer / scheduler
    model = build_model(cfg).to(device)
    loss_fn = get_loss(cfg["loss"])
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # Inferer and post processing for validation
    inferer = SlidingWindowInferer(
        roi_size=tuple(cfg["validation"]["inferer_roi_size"]),
        sw_batch_size=cfg["validation"]["inferer_sw_batch_size"],
        overlap=cfg["validation"]["inferer_overlap"],
        mode="gaussian",
    )
    # Robust conversions that preserve [B, C, ...] shape
    post_pred = PredToOneHot(num_classes=cfg["model"]["out_channels"])
    post_label = LabelToOneHot(num_classes=cfg["model"]["out_channels"])

    # Output dirs
    base_exp_dir = cfg["output"]["exp_dir"]
    ensure_dir(base_exp_dir)
    # Determine run directory (per-run subdir + experiment number)
    if bool(cfg["output"].get("per_run_subdir", True)):
        exp_no = get_next_experiment_number(base_exp_dir)
        run_dir = os.path.join(base_exp_dir, f"exp-{exp_no:03d}")
    else:
        exp_no = 1
        run_dir = base_exp_dir
    ensure_dir(run_dir)
    # TensorBoard writer
    writer = None
    if bool(cfg["output"].get("tensorboard", True)):
        tb_subdir = cfg["output"].get("tb_subdir", "tb")
        tb_dir = os.path.join(run_dir, tb_subdir)
        ensure_dir(tb_dir)
        writer = SummaryWriter(log_dir=tb_dir)
    # Checkpoint and metrics paths (file names templated with exp/epoch/dice)
    best_base = cfg["output"]["best_checkpoint_name"]
    last_base = cfg["output"]["last_checkpoint_name"]
    metrics_csv = os.path.join(run_dir, cfg["output"]["metrics_csv"])

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"] and torch.cuda.is_available()))
    best_dice = -1.0
    best_ckpt = None
    last_ckpt = None
    prev_best_ckpt = None  # Track previous best checkpoint to delete it

    # Get checkpoint saving interval (default: save every epoch for backward compatibility)
    save_last_interval = cfg["output"].get("save_last_interval", 1)

    epochs = cfg["train"]["epochs"]
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, cfg["train"]["log_interval"]
        )
        # TensorBoard: train loss and LR
        if writer is not None:
            writer.add_scalar("train/loss", float(train_loss), epoch)
            writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), epoch)
        val_dice = None
        if epoch % cfg["train"]["val_interval"] == 0:
            # For lesion segmentation, exclude background from dice calculation
            # (we care about lesion detection quality, not background)
            include_bg = cfg.get("validation", {}).get("include_background", False)
            val_dice = validate(model, val_loader, device, inferer, post_pred, post_label, include_background=include_bg)
            print(f"  val mean dice: {val_dice:.4f}")
            if writer is not None and val_dice is not None:
                writer.add_scalar("val/dice", float(val_dice), epoch)

        if scheduler is not None:
            scheduler.step()

        # Save last checkpoint only every N epochs (or at final epoch)
        if epoch % save_last_interval == 0 or epoch == epochs:
            # Remove previous last checkpoint if it exists (to save disk space)
            if last_ckpt is not None and os.path.exists(last_ckpt):
                try:
                    os.remove(last_ckpt)
                except OSError:
                    pass  # Ignore errors if file doesn't exist or can't be deleted
            # exp-XXX_last_model_epoch_200.pth
            last_ckpt = os.path.join(run_dir, f"exp-{exp_no:03d}_{last_base}_epoch_{epoch}.pth")
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, last_ckpt)

        # Save best checkpoint only when dice improves (and remove old best)
        if val_dice is not None and val_dice > best_dice:
            # Remove previous best checkpoint if it exists
            if prev_best_ckpt is not None and os.path.exists(prev_best_ckpt):
                try:
                    os.remove(prev_best_ckpt)
                except OSError:
                    pass  # Ignore errors if file doesn't exist or can't be deleted
            
            best_dice = val_dice
            # exp-XXX_best_model_epoch_200_dice_0.2615.pth
            best_ckpt = os.path.join(run_dir, f"exp-{exp_no:03d}_{best_base}_epoch_{epoch}_dice_{best_dice:.4f}.pth")
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "best_dice": best_dice}, best_ckpt)
            prev_best_ckpt = best_ckpt  # Track for deletion on next improvement

        # Log
        append_metrics_csv(
            metrics_csv,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_dice": float(val_dice) if val_dice is not None else "",
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    print("Training complete.")
    if writer is not None:
        writer.flush()
        writer.close()
    return TrainArtifacts(best_ckpt=best_ckpt, last_ckpt=last_ckpt, metrics_csv=metrics_csv)



