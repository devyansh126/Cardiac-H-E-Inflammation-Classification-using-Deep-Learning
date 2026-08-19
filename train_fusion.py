"""
train_fusion.py
Feature-fusion variant of train.py: same 4-tier progressive-unfreezing
pipeline, same split/holdout logic, but the model also takes engineered
features (nuclei density, H/E stain stats, GLCM texture -- see
engineered_features.py) concatenated before the classifier head.

Kept as a separate script rather than editing train.py in place:
  - train.py stays exactly as-is, so anything that currently depends on it
    (best_of_n.py, list_misclassified.py, compare_segmentation_sources.py)
    is completely unaffected.
  - You can run both and directly compare fusion vs. image-only on the same
    split/holdout setup.

Prerequisite:
    python stain_normalize_cache.py     (if not already run)
    python engineered_features.py       (adds feat_* columns to file_index.csv)

Usage:
    python train_fusion.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

import config
from dataset import (build_index, get_patient_stratified_datasets, get_datasets,
                      get_transforms, get_patient_holdout_dataset, split_patient_partial_holdout,
                      InflammationDataset)
from model import get_fusion_model, get_device, unfreeze_tier_fusion, save_model, load_model
from engineered_features import FEATURE_COLUMNS
from train import EpochTimer, get_class_weighted_criterion, _amp_autocast, pd_concat_dfs


# ---------------------------------------------------------------------------
# One epoch of training -- identical to train.train_one_epoch except the
# loader yields (images, features, labels) and the model takes both inputs.
# Mixup is intentionally NOT ported here: it was already tested and
# reverted in the image-only pipeline (broke Tier 1's ability to fit the
# data), and blending engineered feature vectors the same way has no clear
# justification, so it's left out rather than guessed at.
# ---------------------------------------------------------------------------
def train_one_epoch_fusion(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    use_amp = scaler is not None and scaler.is_enabled()

    optimizer.zero_grad()
    for images, feats, labels in loader:
        images, feats, labels = images.to(device), feats.to(device), labels.to(device)

        if use_amp:
            with _amp_autocast(device):
                outputs = model(images, feats)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=config.GRAD_CLIP_MAX_NORM
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, feats)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=config.GRAD_CLIP_MAX_NORM
            )
            optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def validate_fusion(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, feats, labels in loader:
            images, feats, labels = images.to(device), feats.to(device), labels.to(device)
            with _amp_autocast(device):
                outputs = model(images, feats)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def evaluate_full_fusion(model, loader, device, label=""):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, feats, labels in loader:
            images, feats = images.to(device), feats.to(device)
            with _amp_autocast(device):
                outputs = model(images, feats)
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


def evaluate_per_patient_fusion(model, dataset, device, label="", batch_size=None):
    batch_size = batch_size or config.BATCH_SIZE
    model.eval()
    df = dataset.df
    print(f"\n{'-'*70}\nPer-patient breakdown{f' ({label})' if label else ''}\n{'-'*70}")
    rows = []
    for case_id, group_df in df.groupby("case_id"):
        sub_ds = InflammationDataset(group_df, transform=dataset.transform,
                                      image_source=dataset.image_source,
                                      feature_columns=dataset.feature_columns)
        loader = DataLoader(sub_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, feats, labels in loader:
                images, feats = images.to(device), feats.to(device)
                with _amp_autocast(device):
                    outputs = model(images, feats)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        acc = (all_preds == all_labels).mean()
        true_label_ratio = all_labels.mean()
        rows.append({
            "case_id": case_id, "n": len(all_labels),
            "true_label_name": (config.CLASS_NAMES[1] if true_label_ratio == 1 else
                                 config.CLASS_NAMES[0] if true_label_ratio == 0 else "mixed"),
            "accuracy": acc, "pred_inflam_ratio": all_preds.mean(),
        })
    result = pd.DataFrame(rows).sort_values("accuracy")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return result


# ---------------------------------------------------------------------------
# TTA -- rebuilds the 5 deterministic views directly from the dataframe
# (same approach as train.evaluate_with_tta), plus the engineered feature
# vector for each row, reused unchanged across that row's 5 views (the
# nuclei/stain/texture features describe the tissue itself, not a specific
# augmented view of it).
# ---------------------------------------------------------------------------
def evaluate_with_tta_fusion(model, dataset, device, batch_size=None):
    from dataset import _resolve_image_path
    import torchvision.transforms as T
    from PIL import Image
    import torch.nn.functional as F

    batch_size = batch_size or config.BATCH_SIZE
    df = dataset.df
    image_source = dataset.image_source
    feature_columns = dataset.feature_columns

    degrees = config.TTA_ROTATION_DEGREES
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
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_rows = df.iloc[start:start + batch_size]
            feats_batch = torch.tensor(batch_rows[feature_columns].values.astype("float32")).to(device)
            labels_batch = torch.tensor(batch_rows["label"].values, dtype=torch.long)

            batch_prob_sum = None
            for spec in views_spec:
                tensors = [make_view(Image.open(_resolve_image_path(row, image_source)).convert("RGB"), **spec)
                           for _, row in batch_rows.iterrows()]
                batch_tensor = torch.stack(tensors).to(device)
                outputs = model(batch_tensor, feats_batch)
                probs = F.softmax(outputs, dim=1)
                batch_prob_sum = probs if batch_prob_sum is None else batch_prob_sum + probs

            avg_probs = batch_prob_sum / len(views_spec)
            _, predicted = torch.max(avg_probs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    print(f"\n[Fusion TTA, {len(views_spec)}-view] Accuracy: {acc:.4f}")
    print("\nClassification Report (Fusion TTA):")
    print(report)
    return acc, cm, report


def train_stage_fusion(model, train_loader, val_loader, criterion, optimizer, device,
                        stage_name, max_epochs, patience):
    timer = EpochTimer(stage_name=stage_name, total_epochs=max_epochs)
    best_val_loss = float("inf")
    epochs_without_improvement = 0
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
        train_loss, train_acc = train_one_epoch_fusion(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_loss, val_acc = validate_fusion(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{max_epochs} -- "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} -- "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f} -- LR: {current_lr:.2e}")
        timer.stop(epoch)

        if scheduler is not None:
            scheduler.step(val_loss)

        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_acc_loss_tiebreak):
            best_val_acc = val_acc
            best_acc_loss_tiebreak = val_loss
            best_acc_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_acc_epoch = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping (best val_loss={best_val_loss:.4f}).")
                break

    if best_acc_state is not None:
        model.load_state_dict(best_acc_state)
        print(f"  Restoring BEST-VAL-ACCURACY checkpoint: epoch {best_acc_epoch} "
              f"(val_acc={best_val_acc:.3f}, val_loss={best_acc_loss_tiebreak:.4f}).")
    return model


def run_fusion_pipeline(patience=None, checkpoint_tag="_fusion"):
    patience = patience if patience is not None else config.TIER_PATIENCE
    device = get_device()
    df = build_index()

    missing_feats = [c for c in FEATURE_COLUMNS if c not in df.columns or df[c].isna().any()]
    if missing_feats:
        raise RuntimeError(
            f"Missing/incomplete engineered feature columns {missing_feats} -- "
            f"run engineered_features.py first."
        )

    split_strategy = getattr(config, "SPLIT_STRATEGY", "patient_wise")
    print(f"Split strategy for this run: '{split_strategy}' (config.SPLIT_STRATEGY)")

    partial_holdout_patients = getattr(config, "PARTIAL_HOLDOUT_PATIENTS", {})
    partial_holdout_ids = set(partial_holdout_patients.keys()) & set(df[config.GROUP_COL])
    holdout_patients = [p for p in config.ALWAYS_HOLDOUT_PATIENTS
                        if p in set(df[config.GROUP_COL]) and p not in partial_holdout_ids]
    trainval_df = df[~df[config.GROUP_COL].isin(holdout_patients) &
                      ~df[config.GROUP_COL].isin(partial_holdout_ids)].reset_index(drop=True)

    if split_strategy == "image_wise":
        train_ds, val_ds = get_datasets(trainval_df)
    else:
        train_ds, val_ds = get_patient_stratified_datasets(trainval_df)

    # re-wrap with feature_columns set (get_datasets/get_patient_stratified_datasets
    # build plain InflammationDataset instances without it)
    train_ds = InflammationDataset(train_ds.df, transform=train_ds.transform,
                                    image_source=train_ds.image_source, feature_columns=FEATURE_COLUMNS)
    val_ds = InflammationDataset(val_ds.df, transform=val_ds.transform,
                                  image_source=val_ds.image_source, feature_columns=FEATURE_COLUMNS)

    for case_id in sorted(partial_holdout_ids):
        train_fraction = partial_holdout_patients[case_id]
        p_train_df, p_val_df = split_patient_partial_holdout(
            df, case_id, train_fraction, seed=getattr(config, "PARTIAL_HOLDOUT_SEED", config.RANDOM_SEED)
        )
        train_ds = InflammationDataset(pd.concat([train_ds.df, p_train_df], ignore_index=True),
                                        transform=train_ds.transform, image_source=train_ds.image_source,
                                        feature_columns=FEATURE_COLUMNS)
        val_ds = InflammationDataset(pd.concat([val_ds.df, p_val_df], ignore_index=True),
                                      transform=val_ds.transform, image_source=val_ds.image_source,
                                      feature_columns=FEATURE_COLUMNS)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    criterion = get_class_weighted_criterion(train_ds.df, device)

    val_manifest_path = f"{config.MODEL_DIR}/resnet50_fusion_tier4_val_manifest.csv"
    val_ds.df.to_csv(val_manifest_path, index=False)
    print(f"Saved val manifest ({len(val_ds.df)} images, {val_ds.df[config.GROUP_COL].nunique()} patients) "
          f"to {val_manifest_path}")

    holdout_loader = None
    holdout_ds = None
    if holdout_patients:
        holdout_ds = get_patient_holdout_dataset(df, holdout_patients[0])
        for extra in holdout_patients[1:]:
            more = get_patient_holdout_dataset(df, extra)
            holdout_ds.df = pd_concat_dfs(holdout_ds.df, more.df)
        holdout_ds = InflammationDataset(holdout_ds.df, transform=holdout_ds.transform,
                                          image_source=holdout_ds.image_source, feature_columns=FEATURE_COLUMNS)
        holdout_loader = DataLoader(holdout_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model = get_fusion_model(len(FEATURE_COLUMNS), device)

    for tier in range(1, 5):
        print(f"\n{'='*70}\nFUSION TIER {tier}\n{'='*70}")
        model = unfreeze_tier_fusion(model, tier)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TIER_LR[tier],
            weight_decay=getattr(config, "WEIGHT_DECAY", 0.0)
        )
        model = train_stage_fusion(model, train_loader, val_loader, criterion, optimizer, device,
                                    stage_name=f"Fusion Tier {tier}", max_epochs=config.TIER_MAX_EPOCHS[tier],
                                    patience=patience)
        evaluate_full_fusion(model, val_loader, device, label=f"Fusion Tier {tier} val")
        if holdout_loader is not None:
            evaluate_full_fusion(model, holdout_loader, device,
                                  label=f"Fusion Tier {tier} TRUE HOLDOUT ({', '.join(holdout_patients)})")
        ckpt_path = f"{config.MODEL_DIR}/resnet50_fusion_tier{tier}{checkpoint_tag}.pth"
        save_model(model, ckpt_path)

    print(f"\n{'='*70}\nFUSION FINAL EVALUATION\n{'='*70}")
    metrics = {}
    final_cm, _ = evaluate_full_fusion(model, val_loader, device, label="final, single-pred, val")
    metrics["val_acc"] = float(np.trace(final_cm) / final_cm.sum())

    if holdout_loader is not None:
        holdout_cm, _ = evaluate_full_fusion(model, holdout_loader, device,
                                              label=f"final, TRUE HOLDOUT ({', '.join(holdout_patients)})")
        metrics["holdout_acc"] = float(np.trace(holdout_cm) / holdout_cm.sum())

    if config.TTA_ENABLED:
        tta_acc, _, _ = evaluate_with_tta_fusion(model, val_ds, device)
        metrics["val_tta_acc"] = tta_acc
        if holdout_ds is not None:
            holdout_tta_acc, _, _ = evaluate_with_tta_fusion(model, holdout_ds, device)
            metrics["holdout_tta_acc"] = holdout_tta_acc

    evaluate_per_patient_fusion(model, val_ds, device, label="val")
    if holdout_ds is not None:
        evaluate_per_patient_fusion(model, holdout_ds, device, label="TRUE HOLDOUT")

    print("\nFusion pipeline metrics:", metrics)
    return model, metrics


if __name__ == "__main__":
    run_fusion_pipeline()