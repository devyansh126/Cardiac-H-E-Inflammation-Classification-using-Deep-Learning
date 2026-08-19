"""
train.py
Main training entry point for the cardiac inflammation classifier.

Pipeline (matches the handoff doc's "clean pipeline" decisions):
  1. Load index (dataset.build_index) -- never rescans blindly, keeps cached_path/clahe_path
  2. Patient-stratified split (dataset.get_patient_stratified_datasets) -- GroupShuffleSplit
     on case_id, with an automatic search for a class-balanced split
  3. ResNet50, 4-tier progressive unfreezing, each tier with early stopping
     (patience=3 on val loss, restore best weights) and its own checkpoint
  4. Final evaluation: confusion matrix + classification report, single-pred
     AND with 5-view Test-Time Augmentation (confirmed +1.6pp, essentially free)

Usage:
    python train.py
Or import pieces into a notebook:
    from train import run_pipeline, evaluate_full, evaluate_with_tta
"""

import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

import config
from dataset import (build_index, get_patient_stratified_datasets, get_datasets, get_dataloaders,
                      get_transforms, get_patient_holdout_dataset, split_patient_partial_holdout,
                      InflammationDataset)
from model import get_model, get_device, unfreeze_tier, save_model, freeze_bn_stats, load_model


# ---------------------------------------------------------------------------
# Time estimator -- tracks per-epoch duration and estimates remaining time
# ---------------------------------------------------------------------------
class EpochTimer:
    def __init__(self, stage_name="", total_epochs=1):
        self.stage_name = stage_name
        self.total_epochs = total_epochs
        self.epoch_times = []
        self._epoch_start = None
        self._stage_start = time.time()

    def start(self):
        self._epoch_start = time.time()

    def stop(self, epoch_idx):
        elapsed = time.time() - self._epoch_start
        self.epoch_times.append(elapsed)

        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        epochs_remaining = self.total_epochs - (epoch_idx + 1)
        eta_seconds = avg_epoch_time * epochs_remaining
        stage_elapsed = time.time() - self._stage_start

        print(f"    [{self.stage_name}] epoch time: {self._fmt(elapsed)} | "
              f"avg/epoch: {self._fmt(avg_epoch_time)} | "
              f"stage elapsed: {self._fmt(stage_elapsed)} | "
              f"ETA remaining: {self._fmt(eta_seconds)} "
              f"(finish ~{self._eta_clock(eta_seconds)})")
        return elapsed

    @staticmethod
    def _fmt(seconds):
        return str(datetime.timedelta(seconds=int(seconds)))

    @staticmethod
    def _eta_clock(seconds_remaining):
        finish_time = datetime.datetime.now() + datetime.timedelta(seconds=seconds_remaining)
        return finish_time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Class-weighted loss
# ---------------------------------------------------------------------------
def get_class_weighted_criterion(train_df, device):
    counts = train_df["label"].value_counts().sort_index()  # index 0, 1
    weights = 1.0 / counts.values
    weights = weights / weights.sum() * len(counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    label_smoothing = getattr(config, "LABEL_SMOOTHING", 0.0)
    print(f"Class weights (0=Non-inflammatory, 1=Inflammatory): {weights_tensor.tolist()} "
          f"| label_smoothing={label_smoothing}")
    return nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)


# ---------------------------------------------------------------------------
# AMP helper -- autocast context, on only when enabled and running on CUDA
# ---------------------------------------------------------------------------
def _amp_autocast(device):
    enabled = config.AMP_ENABLED and device.type == "cuda"
    return torch.autocast(device_type="cuda", enabled=enabled)


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    if getattr(config, "FREEZE_BN_STATS", False):
        # Keep BatchNorm running stats fixed (ImageNet-derived) even on
        # unfrozen tiers -- BATCH_SIZE=8 gives noisy per-batch BN estimates
        # otherwise, which is a likely cause of the epoch-to-epoch and
        # tier-to-tier oscillation seen in early training runs.
        freeze_bn_stats(model)

    total_loss, correct, total = 0, 0, 0
    use_amp = scaler is not None and scaler.is_enabled()
    accum_steps = max(1, getattr(config, "GRAD_ACCUM_STEPS", 1))

    mixup_alpha = getattr(config, "MIXUP_ALPHA", 0.0)

    optimizer.zero_grad()
    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Mixup: blend each batch with a randomly-permuted copy of itself,
        # both images and (via the loss) labels. lam close to 1 keeps most
        # batches close to the original image -- Beta(0.2, 0.2) is the
        # standard mixup default and skews toward lam near 0 or 1 rather
        # than always averaging 50/50, which keeps most batches close to
        # a real image rather than a uniform blur of two.
        use_mixup = mixup_alpha > 0.0
        if use_mixup:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(images.size(0), device=device)
            images = lam * images + (1 - lam) * images[perm]
            labels_b = labels[perm]

        if use_amp:
            with _amp_autocast(device):
                outputs = model(images)
                if use_mixup:
                    loss = (lam * criterion(outputs, labels)
                            + (1 - lam) * criterion(outputs, labels_b)) / accum_steps
                else:
                    loss = criterion(outputs, labels) / accum_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(images)
            if use_mixup:
                loss = (lam * criterion(outputs, labels)
                        + (1 - lam) * criterion(outputs, labels_b)) / accum_steps
            else:
                loss = criterion(outputs, labels) / accum_steps
            loss.backward()

        is_last_batch = (step + 1) == len(loader)
        if (step + 1) % accum_steps == 0 or is_last_batch:
            if use_amp:
                scaler.unscale_(optimizer)  # so grad clipping operates on real-scale grads
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=config.GRAD_CLIP_MAX_NORM
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=config.GRAD_CLIP_MAX_NORM
                )
                optimizer.step()
            optimizer.zero_grad()

        # loss was divided by accum_steps for backward -- undo that for the
        # reported running-average loss so it stays comparable across
        # different GRAD_ACCUM_STEPS settings.
        total_loss += loss.item() * accum_steps * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Validation (loss + accuracy -- quick per-epoch check)
# ---------------------------------------------------------------------------
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with _amp_autocast(device):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Full evaluation -- confusion matrix + precision/recall
# ---------------------------------------------------------------------------
def evaluate_full(model, loader, device, label=""):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            with _amp_autocast(device):
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES)

    print(f"\nConfusion Matrix{f' ({label})' if label else ''}:")
    print(f"{'':>20} {'Pred Non-inflam':>18} {'Pred Inflam':>14}")
    print(f"{'True Non-inflam':>20} {cm[0][0]:>18} {cm[0][1]:>14}")
    print(f"{'True Inflam':>20} {cm[1][0]:>18} {cm[1][1]:>14}")
    print("\nClassification Report:")
    print(report)

    return cm, report


# ---------------------------------------------------------------------------
# Per-patient breakdown -- direct monitor for the patient/stain-color
# confound found by analyze_images.py (25/33 patients are label-pure, so a
# good AGGREGATE val accuracy can hide the model just recognizing easy,
# label-pure patients rather than real biology). Run this every time,
# not just when something already looks wrong.
# ---------------------------------------------------------------------------
def evaluate_per_patient(model, dataset, device, label="", batch_size=None):
    """
    dataset: an InflammationDataset (has .df with case_id/label, .image_source).
    Reports accuracy and prediction bias for EACH patient in val/holdout, so
    a couple of easy label-pure patients can't hide behind a good aggregate
    number.
    """
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    model.eval()
    df = dataset.df

    print(f"\n{'-'*70}\nPer-patient breakdown{f' ({label})' if label else ''}\n{'-'*70}")
    rows = []
    for case_id, group_df in df.groupby("case_id"):
        sub_ds = InflammationDataset(group_df, transform=dataset.transform, image_source=dataset.image_source)
        loader = DataLoader(sub_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                with _amp_autocast(device):
                    outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        acc = (all_preds == all_labels).mean()
        true_label_ratio = all_labels.mean()  # 0 = pure non-inflam, 1 = pure inflam
        pred_inflam_ratio = all_preds.mean()
        rows.append({
            "case_id": case_id, "n": len(all_labels), "true_label_name": (
                config.CLASS_NAMES[1] if true_label_ratio == 1 else
                config.CLASS_NAMES[0] if true_label_ratio == 0 else "mixed"),
            "accuracy": acc, "pred_inflam_ratio": pred_inflam_ratio,
        })

    result = pd.DataFrame(rows).sort_values("accuracy")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    worst = result[result["accuracy"] < 0.5]
    if len(worst) > 0:
        print(f"\n  WARNING: {len(worst)} patient(s) below 50% accuracy -- "
              f"{worst['case_id'].tolist()}. Check whether these look like "
              f"stain/scanner outliers relative to the rest of the training set.")
    return result


# ---------------------------------------------------------------------------
# Test-Time Augmentation -- 5-view averaging (original + hflip + vflip +
# +-10deg rotation). Confirmed +1.6pp (88.8% -> 90.4%) on the full-dataset
# ResNet50 Tier-4 model. Operates directly on a DataFrame + image_source so
# it can build its own no-augmentation-base transform regardless of which
# transform the val_loader happened to use.
# ---------------------------------------------------------------------------
def evaluate_with_tta(model, dataset, device, batch_size=None):
    """
    dataset: an InflammationDataset (uses its .df and .image_source, but
    ignores its .transform -- TTA builds its own 5 deterministic views).
    """
    from dataset import InflammationDataset

    batch_size = batch_size or config.BATCH_SIZE
    base_transform = get_transforms()[1]  # val_transform: resize + tensor + normalize, no randomness

    degrees = config.TTA_ROTATION_DEGREES
    view_transforms = {
        "original": base_transform,
        "hflip": _compose_with(base_transform, lambda img: img.transpose(0)),  # placeholder, replaced below
    }

    # Build the 5 views explicitly with torchvision so each is deterministic
    import torchvision.transforms as T
    from PIL import Image

    resize = T.Resize((config.IMG_SIZE, config.IMG_SIZE))
    to_tensor_norm = T.Compose([T.ToTensor(), T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)])

    def make_view(img, hflip=False, vflip=False, rotate=0):
        v = resize(img)
        if hflip:
            v = v.transpose(Image.FLIP_LEFT_RIGHT)
        if vflip:
            v = v.transpose(Image.FLIP_TOP_BOTTOM)
        if rotate:
            v = v.rotate(rotate)
        return to_tensor_norm(v)

    views_spec = [
        dict(hflip=False, vflip=False, rotate=0),
        dict(hflip=True, vflip=False, rotate=0),
        dict(hflip=False, vflip=True, rotate=0),
        dict(hflip=False, vflip=False, rotate=degrees),
        dict(hflip=False, vflip=False, rotate=-degrees),
    ]

    model.eval()
    all_preds, all_labels = [], []
    df = dataset.df
    image_source = dataset.image_source

    from dataset import _resolve_image_path

    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_rows = df.iloc[start:start + batch_size]
            batch_logit_sum = None
            labels_batch = torch.tensor(batch_rows["label"].values, dtype=torch.long)

            for spec in views_spec:
                tensors = []
                for _, row in batch_rows.iterrows():
                    img = Image.open(_resolve_image_path(row, image_source)).convert("RGB")
                    tensors.append(make_view(img, **spec))
                batch_tensor = torch.stack(tensors).to(device)
                outputs = model(batch_tensor)
                probs = F.softmax(outputs, dim=1)
                batch_logit_sum = probs if batch_logit_sum is None else batch_logit_sum + probs

            avg_probs = batch_logit_sum / len(views_spec)
            _, predicted = torch.max(avg_probs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    print(f"\n[TTA, {len(views_spec)}-view] Accuracy: {acc:.4f}")
    print("Confusion Matrix (TTA):")
    print(f"{'':>20} {'Pred Non-inflam':>18} {'Pred Inflam':>14}")
    print(f"{'True Non-inflam':>20} {cm[0][0]:>18} {cm[0][1]:>14}")
    print(f"{'True Inflam':>20} {cm[1][0]:>18} {cm[1][1]:>14}")
    print("\nClassification Report (TTA):")
    print(report)

    return acc, cm, report


def _compose_with(base_transform, fn):
    # unused placeholder kept out of the hot path; real views are built in evaluate_with_tta
    return base_transform


# ---------------------------------------------------------------------------
# Training stage with early stopping
# ---------------------------------------------------------------------------
def train_stage(model, train_loader, val_loader, criterion, optimizer, device,
                 stage_name, max_epochs, patience):
    timer = EpochTimer(stage_name=stage_name, total_epochs=max_epochs)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Checkpoint selection is now tracked SEPARATELY from early stopping.
    # Previously this function saved/restored the state with the lowest
    # val LOSS, which silently discarded higher-accuracy epochs whenever
    # loss and accuracy disagreed (e.g. a confidently-wrong-on-a-few-cases
    # epoch can have worse loss but better accuracy than a hedging one).
    # Early stopping itself is UNCHANGED below -- still driven by val loss
    # plateauing, exactly as before. Only which weights get restored at
    # the end of the stage has changed: now the best-val-ACCURACY epoch,
    # ties broken by lower val loss.
    best_val_acc = -1.0
    best_acc_loss_tiebreak = float("inf")
    best_acc_state = None
    best_acc_epoch = None

    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP_ENABLED and device.type == "cuda")

    scheduler = None
    if getattr(config, "LR_SCHEDULER_ENABLED", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.LR_SCHEDULER_MIN_LR,
        )

    for epoch in range(max_epochs):
        timer.start()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{max_epochs} -- "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} -- "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f} -- LR: {current_lr:.2e}")
        timer.stop(epoch)

        if scheduler is not None:
            prev_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < prev_lr:
                print(f"  LR scheduler: val loss plateaued -- LR {prev_lr:.2e} -> {new_lr:.2e}")

        # --- Checkpoint tracking (by accuracy, ties broken by loss) ---
        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_acc_loss_tiebreak):
            best_val_acc = val_acc
            best_acc_loss_tiebreak = val_loss
            best_acc_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_acc_epoch = epoch + 1

        # --- Early stopping (by loss plateau) -- UNCHANGED from before ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping -- val loss hasn't improved for {patience} epochs "
                      f"(best val_loss={best_val_loss:.4f}).")
                break

    if best_acc_state is not None:
        model.load_state_dict(best_acc_state)
        print(f"  Restoring BEST-VAL-ACCURACY checkpoint: epoch {best_acc_epoch} "
              f"(val_acc={best_val_acc:.3f}, val_loss={best_acc_loss_tiebreak:.4f}).")
    return model


# ---------------------------------------------------------------------------
# Full pipeline: patient-stratified split, all 4 tiers, per-tier checkpoint,
# final evaluation with and without TTA
# ---------------------------------------------------------------------------
def run_pipeline(architecture=None, patience=None, start_tier=1, stop_after_tier=None, checkpoint_tag=""):
    """
    stop_after_tier: if set (e.g. 3), stops right after that tier finishes
    training+eval and returns (model, metrics) immediately -- skips later
    tiers and the "FINAL EVALUATION" block entirely. Used for best-of-N
    candidate generation: run Tiers 1-3 many times cheaply, keep the
    best-scoring Tier 3 checkpoint, then run Tier 4 once from that winner
    instead of from an arbitrary/mediocre Tier 3.

    checkpoint_tag: appended to every saved checkpoint's filename (e.g.
    "_cand3" -> "resnet50_patientsplit_tier3_cand3.pth") so multiple
    candidate runs don't overwrite each other or the "real"
    resnet50_patientsplit_tierN.pth files that start_tier=4 and
    ensemble_eval.py expect by default (tag="").
    """
    architecture = architecture or config.ARCHITECTURE
    patience = patience if patience is not None else config.TIER_PATIENCE

    device = get_device()
    df = build_index()

    resolved_image_source = config.DEFAULT_IMAGE_SOURCE
    split_strategy = getattr(config, "SPLIT_STRATEGY", "patient_wise")
    print(f"Image source for this run: '{resolved_image_source}' "
          f"(config.DEFAULT_IMAGE_SOURCE) | FREEZE_BN_STATS={getattr(config, 'FREEZE_BN_STATS', False)} "
          f"| GRAD_ACCUM_STEPS={getattr(config, 'GRAD_ACCUM_STEPS', 1)} "
          f"(effective batch size {config.BATCH_SIZE * getattr(config, 'GRAD_ACCUM_STEPS', 1)})")
    print(f"Split strategy for this run: '{split_strategy}' (config.SPLIT_STRATEGY)")
    if split_strategy == "image_wise":
        print("  NOTE: image-wise split means a patient's images CAN appear in both "
              "train and val below -- val accuracy here is not a clean generalization "
              "estimate for that reason. The TRUE HOLDOUT numbers further down (a "
              "patient excluded from train+val entirely) are the leakage-free check.")

    partial_holdout_patients = getattr(config, "PARTIAL_HOLDOUT_PATIENTS", {})
    partial_holdout_ids = set(partial_holdout_patients.keys()) & set(df[config.GROUP_COL])

    # Partial takes precedence over full exclusion for any patient listed in both.
    holdout_patients = [p for p in config.ALWAYS_HOLDOUT_PATIENTS
                        if p in set(df[config.GROUP_COL]) and p not in partial_holdout_ids]
    trainval_df = df[~df[config.GROUP_COL].isin(holdout_patients) & ~df[config.GROUP_COL].isin(partial_holdout_ids)].reset_index(drop=True)

    if holdout_patients:
        print(f"Excluding {holdout_patients} from train+val entirely -- "
              f"evaluated separately as a dedicated holdout below, every tier.")

    if partial_holdout_ids:
        print(f"Partial-holdout patients {sorted(partial_holdout_ids)}: splitting each "
              f"patient's OWN images into train/val per config.PARTIAL_HOLDOUT_PATIENTS "
              f"-- NOT a generalization check, see config.py comment for why.")

    if split_strategy == "image_wise":
        train_ds, val_ds = get_datasets(trainval_df)
    else:
        train_ds, val_ds = get_patient_stratified_datasets(trainval_df)

    for case_id in sorted(partial_holdout_ids):
        train_fraction = partial_holdout_patients[case_id]
        p_train_df, p_val_df = split_patient_partial_holdout(
            df, case_id, train_fraction, seed=getattr(config, "PARTIAL_HOLDOUT_SEED", config.RANDOM_SEED)
        )
        train_ds = InflammationDataset(
            pd.concat([train_ds.df, p_train_df], ignore_index=True),
            transform=train_ds.transform, image_source=train_ds.image_source
        )
        val_ds = InflammationDataset(
            pd.concat([val_ds.df, p_val_df], ignore_index=True),
            transform=val_ds.transform, image_source=val_ds.image_source
        )

    train_loader, val_loader = get_dataloaders(train_ds, val_ds)
    criterion = get_class_weighted_criterion(train_ds.df, device)

    val_manifest_path = f"{config.MODEL_DIR}/{architecture}_{split_strategy}_tier4{checkpoint_tag}_val_manifest.csv"
    val_ds.df.to_csv(val_manifest_path, index=False)
    print(f"Saved val manifest ({len(val_ds.df)} images, {val_ds.df[config.GROUP_COL].nunique()} patients) "
          f"to {val_manifest_path}")

    holdout_loader = None
    if holdout_patients:
        _, val_transform = get_transforms()
        holdout_ds = get_patient_holdout_dataset(df, holdout_patients[0])
        # if more than one holdout patient is configured, combine them
        for extra in holdout_patients[1:]:
            more = get_patient_holdout_dataset(df, extra)
            holdout_ds.df = pd_concat_dfs(holdout_ds.df, more.df)
        holdout_loader, _ = get_dataloaders(holdout_ds, holdout_ds, batch_size=config.BATCH_SIZE)

    model = get_model(architecture, device)

    if start_tier > 1:
        prior_tier = start_tier - 1
        ckpt_path = f"{config.MODEL_DIR}/{architecture}_{split_strategy}_tier{prior_tier}.pth"
        print(f"\nstart_tier={start_tier}: loading existing Tier {prior_tier} checkpoint "
              f"from {ckpt_path} and skipping Tiers 1-{prior_tier} entirely.")
        model = load_model(model, ckpt_path, device)

    for tier in range(start_tier, 5):
        print(f"\n{'='*70}\nTIER {tier} ({architecture}, patient-stratified split, "
              f"holdout patient(s) excluded from training)\n{'='*70}")
        model = unfreeze_tier(model, architecture, tier)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TIER_LR[tier],
            weight_decay=getattr(config, "WEIGHT_DECAY", 0.0)
        )
        model = train_stage(
            model, train_loader, val_loader, criterion, optimizer, device,
            stage_name=f"Tier {tier}", max_epochs=config.TIER_MAX_EPOCHS[tier],
            patience=patience
        )
        tier_cm, _ = evaluate_full(model, val_loader, device, label=f"Tier {tier} val (diverse, {val_ds.df['case_id'].nunique()} patients)")
        tier_val_acc = np.trace(tier_cm) / tier_cm.sum()
        if holdout_loader is not None:
            evaluate_full(model, holdout_loader, device,
                          label=f"Tier {tier} TRUE HOLDOUT ({', '.join(holdout_patients)}, never trained on)")
        ckpt_path = f"{config.MODEL_DIR}/{architecture}_{split_strategy}_tier{tier}{checkpoint_tag}.pth"
        save_model(model, ckpt_path)

        if stop_after_tier is not None and tier == stop_after_tier:
            print(f"\nstop_after_tier={stop_after_tier}: returning early after Tier {tier} "
                  f"(val_acc={tier_val_acc:.4f}). Checkpoint saved to {ckpt_path}.")
            return model, {"stopped_after_tier": tier, "tier_val_acc": tier_val_acc,
                            "checkpoint_path": ckpt_path}

    # --- Final evaluation: single-pred vs TTA, on both val and the true holdout ---
    print(f"\n{'='*70}\nFINAL EVALUATION\n{'='*70}")
    metrics = {}

    final_cm, _ = evaluate_full(model, val_loader, device, label="final, single-pred, diverse val")
    metrics["val_acc"] = np.trace(final_cm) / final_cm.sum()

    if holdout_loader is not None:
        holdout_cm, _ = evaluate_full(model, holdout_loader, device,
                      label=f"final, TRUE HOLDOUT ({', '.join(holdout_patients)})")
        metrics["holdout_acc"] = np.trace(holdout_cm) / holdout_cm.sum()

    if config.TTA_ENABLED:
        print("\n--- TTA on diverse val ---")
        tta_acc, _, _ = evaluate_with_tta(model, val_ds, device)
        metrics["val_tta_acc"] = tta_acc
        if holdout_patients:
            print(f"\n--- TTA on TRUE HOLDOUT ({', '.join(holdout_patients)}) ---")
            holdout_tta_acc, _, _ = evaluate_with_tta(model, holdout_ds, device)
            metrics["holdout_tta_acc"] = holdout_tta_acc

    # Per-patient breakdown -- direct check that good aggregate accuracy
    # isn't just a handful of easy, label-pure patients (the confound
    # analyze_images.py flagged). Run every time, not only when something
    # already looks off.
    evaluate_per_patient(model, val_ds, device, label="diverse val")
    if holdout_patients:
        evaluate_per_patient(model, holdout_ds, device, label="TRUE HOLDOUT")

    return model, metrics


def pd_concat_dfs(df1, df2):
    import pandas as pd
    return pd.concat([df1, df2]).reset_index(drop=True)


if __name__ == "__main__":
    run_pipeline()