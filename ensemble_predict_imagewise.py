"""
ensemble_predict_imagewise.py
Image-wise counterpart to ensemble_predict.py (which is locked to
patient-wise). Combines independently-trained IMAGE-WISE models -- can mix
plain image-only checkpoints and the fusion (image+engineered-features)
checkpoint, since both were trained under the same split policy.

Still manifest-verified, not reconstructed: each checkpoint must be paired
with its own val_manifest.csv (saved at training time) so a split mismatch
is caught loudly instead of silently producing a wrong number (see the
patient-wise version's history for exactly why this matters).

>>> EDIT CHECKPOINTS BELOW. kind="image" uses model.get_model; kind="fusion"
>>> uses model.get_fusion_model + engineered_features.FEATURE_COLUMNS.

Usage:
    python ensemble_predict_imagewise.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, f1_score

import config
from dataset import _resolve_image_path
from model import get_model, get_fusion_model, load_model, get_device


# ---------------------------------------------------------------------------
# EDIT THIS -- each entry needs its own manifest (from its own training run).
# ---------------------------------------------------------------------------
CHECKPOINTS = [
    dict(kind="image", architecture="resnet50",
         path=r"D:\BioInformatics\IITRoorkeProject\models\resnet50_image_wise_tier4.pth",
         manifest=r"D:\BioInformatics\IITRoorkeProject\models\resnet50_image_wise_tier4_val_manifest.csv"),
    dict(kind="fusion",
         path=r"D:\BioInformatics\IITRoorkeProject\models\resnet50_fusion_tier4_fusion.pth",
         manifest=r"D:\BioInformatics\IITRoorkeProject\models\resnet50_fusion_tier4_val_manifest.csv"),
]

THRESHOLD_SWEEP = np.arange(0.30, 0.71, 0.02)
WEIGHT_SWEEP = np.arange(0.0, 1.01, 0.05)  # weight on CHECKPOINTS[0]; only used when exactly 2 models


def load_val_manifest(manifest_path):
    val_df = pd.read_csv(manifest_path)
    print(f"Loaded val manifest: {manifest_path}")
    print(f"  {len(val_df)} images, {val_df[config.GROUP_COL].nunique()} patients.")
    return val_df


def assert_manifests_match(reference_df, other_df, ref_name, other_name):
    ref_key = set(reference_df[config.GROUP_COL].astype(str) + "::" + reference_df["path"].astype(str))
    other_key = set(other_df[config.GROUP_COL].astype(str) + "::" + other_df["path"].astype(str))
    if ref_key != other_key:
        only_in_ref = list(ref_key - other_key)[:5]
        only_in_other = list(other_key - ref_key)[:5]
        raise ValueError(
            f"Manifest mismatch between '{ref_name}' and '{other_name}' -- these two runs used "
            f"DIFFERENT val sets and cannot be ensembled together.\n"
            f"  In {ref_name} but not {other_name} (sample): {only_in_ref}\n"
            f"  In {other_name} but not {ref_name} (sample): {only_in_other}"
        )


def _make_views(img, degrees):
    resize = T.Resize((config.IMG_SIZE, config.IMG_SIZE))
    to_tensor_norm = T.Compose([T.ToTensor(), T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)])
    specs = [
        dict(hflip=False, vflip=False, rotate=0),
        dict(hflip=True, vflip=False, rotate=0),
        dict(hflip=False, vflip=True, rotate=0),
        dict(hflip=False, vflip=False, rotate=degrees),
        dict(hflip=False, vflip=False, rotate=-degrees),
    ]
    views = []
    for spec in specs:
        v = resize(img)
        if spec["hflip"]:
            v = v.transpose(Image.FLIP_LEFT_RIGHT)
        if spec["vflip"]:
            v = v.transpose(Image.FLIP_TOP_BOTTOM)
        if spec["rotate"]:
            v = v.rotate(spec["rotate"])
        views.append(to_tensor_norm(v))
    return views


def get_tta_probs_for_model(model, kind, val_df, device, image_source, feature_columns=None, batch_size=None):
    batch_size = batch_size or config.BATCH_SIZE
    model.eval()
    all_probs = []

    with torch.no_grad():
        for start in range(0, len(val_df), batch_size):
            batch_rows = val_df.iloc[start:start + batch_size]
            imgs = [Image.open(_resolve_image_path(row, image_source)).convert("RGB")
                    for _, row in batch_rows.iterrows()]
            per_view_tensors = list(zip(*[_make_views(img, config.TTA_ROTATION_DEGREES) for img in imgs]))

            feats_batch = None
            if kind == "fusion":
                feats_batch = torch.tensor(batch_rows[feature_columns].values.astype("float32")).to(device)

            prob_sum = None
            for view_tensors in per_view_tensors:
                batch_tensor = torch.stack(view_tensors).to(device)
                outputs = model(batch_tensor, feats_batch) if kind == "fusion" else model(batch_tensor)
                probs = F.softmax(outputs, dim=1)
                prob_sum = probs if prob_sum is None else prob_sum + probs
            avg_probs = (prob_sum / len(per_view_tensors)).cpu().numpy()
            all_probs.append(avg_probs)

    return np.concatenate(all_probs, axis=0)


def sweep_threshold(probs_inflam, labels):
    best = None
    for t in THRESHOLD_SWEEP:
        preds = (probs_inflam >= t).astype(int)
        f1_macro = f1_score(labels, preds, average="macro")
        f1_per_class = f1_score(labels, preds, average=None)
        if best is None or f1_macro > best["f1_macro"]:
            best = dict(threshold=float(t), f1_macro=f1_macro,
                        f1_non_inflam=f1_per_class[0], f1_inflam=f1_per_class[1])
    return best


def sweep_weight_and_threshold(probs_inflam_a, probs_inflam_b, labels):
    """
    Jointly searches the blend weight (alpha*model_A + (1-alpha)*model_B)
    and the decision threshold, maximizing macro F1. A straight 50/50
    average ignores that the two models have different solo strength --
    this lets the weaker model's contribution shrink instead of dragging
    the stronger one down equally.

    Honest caveat: searching TWO knobs (weight + threshold) against the
    same val set is more prone to overfitting that val set than searching
    one (threshold alone, as the rest of this script already does). With
    only 232 val images, treat any improvement here as suggestive, not
    as strong evidence -- if it barely beats the equal-weight result,
    that itself is useful information (the models are too correlated for
    reweighting to matter much), not a failure.
    """
    best = None
    for alpha in WEIGHT_SWEEP:
        blended = alpha * probs_inflam_a + (1 - alpha) * probs_inflam_b
        for t in THRESHOLD_SWEEP:
            preds = (blended >= t).astype(int)
            f1_macro = f1_score(labels, preds, average="macro")
            f1_per_class = f1_score(labels, preds, average=None)
            acc = (preds == labels).mean()
            if best is None or f1_macro > best["f1_macro"]:
                best = dict(alpha=float(alpha), threshold=float(t), f1_macro=f1_macro,
                            f1_non_inflam=f1_per_class[0], f1_inflam=f1_per_class[1], acc=acc)
    return best


def main():
    from engineered_features import FEATURE_COLUMNS
    device = get_device()

    reference_df = load_val_manifest(CHECKPOINTS[0]["manifest"])
    for ckpt in CHECKPOINTS[1:]:
        other_df = load_val_manifest(ckpt["manifest"])
        assert_manifests_match(reference_df, other_df, CHECKPOINTS[0]["path"], ckpt["path"])
    print("All manifests agree on the same val images -- safe to ensemble.\n")

    val_df = reference_df
    labels = val_df["label"].values
    image_source = getattr(config, "DEFAULT_IMAGE_SOURCE", "stain_norm")

    all_model_probs = []
    for ckpt in CHECKPOINTS:
        print(f"\nLoading {ckpt['kind']} model: {ckpt['path']}")
        if ckpt["kind"] == "image":
            model = get_model(ckpt["architecture"], device)
            model = load_model(model, ckpt["path"], device)
        else:
            model = get_fusion_model(len(FEATURE_COLUMNS), device)
            model = load_model(model, ckpt["path"], device)

        probs = get_tta_probs_for_model(model, ckpt["kind"], val_df, device, image_source,
                                         feature_columns=FEATURE_COLUMNS)
        preds = probs.argmax(axis=1)
        acc = (preds == labels).mean()
        print(f"  Solo TTA accuracy: {acc:.4f}")
        print(classification_report(labels, preds, target_names=config.CLASS_NAMES))
        all_model_probs.append(probs)

    ensemble_probs = np.mean(all_model_probs, axis=0)
    probs_inflam = ensemble_probs[:, 1]

    print(f"\n{'='*70}\nENSEMBLE OF {len(CHECKPOINTS)} IMAGE-WISE MODELS (EQUAL WEIGHT)\n{'='*70}")
    default_preds = ensemble_probs.argmax(axis=1)
    acc_default = (default_preds == labels).mean()
    print(f"\nEnsemble accuracy @ threshold 0.50: {acc_default:.4f}")
    print(classification_report(labels, default_preds, target_names=config.CLASS_NAMES))

    best = sweep_threshold(probs_inflam, labels)
    tuned_preds = (probs_inflam >= best["threshold"]).astype(int)
    acc_tuned = (tuned_preds == labels).mean()

    print(f"\nBest threshold from sweep: {best['threshold']:.2f} "
          f"(F1 non-inflam={best['f1_non_inflam']:.3f}, F1 inflam={best['f1_inflam']:.3f})")
    print(f"Ensemble accuracy @ tuned threshold: {acc_tuned:.4f}")
    cm = confusion_matrix(labels, tuned_preds)
    print(f"\nConfusion Matrix (tuned threshold, equal weight):")
    print(f"{'':>20} {'Pred Non-inflam':>18} {'Pred Inflam':>14}")
    print(f"{'True Non-inflam':>20} {cm[0][0]:>18} {cm[0][1]:>14}")
    print(f"{'True Inflam':>20} {cm[1][0]:>18} {cm[1][1]:>14}")
    print("\nClassification Report (tuned threshold, equal weight):")
    print(classification_report(labels, tuned_preds, target_names=config.CLASS_NAMES))

    if len(CHECKPOINTS) == 2:
        print(f"\n{'='*70}\nWEIGHTED BLEND + THRESHOLD SWEEP (2 models only)\n{'='*70}")
        probs_a = all_model_probs[0][:, 1]
        probs_b = all_model_probs[1][:, 1]
        wbest = sweep_weight_and_threshold(probs_a, probs_b, labels)
        w_preds = ((wbest["alpha"] * probs_a + (1 - wbest["alpha"]) * probs_b) >= wbest["threshold"]).astype(int)

        print(f"Best weight (on model 0 = {CHECKPOINTS[0]['path'].split(chr(92))[-1]}): {wbest['alpha']:.2f} "
              f"| threshold: {wbest['threshold']:.2f}")
        print(f"Accuracy: {wbest['acc']:.4f} | F1 non-inflam={wbest['f1_non_inflam']:.3f}, "
              f"F1 inflam={wbest['f1_inflam']:.3f}")
        print("NOTE: this searched weight AND threshold against the same 232-image val set -- "
              "treat any gain over the equal-weight result above as suggestive, not conclusive, "
              "given the small sample. If it's close to equal weight, that itself tells you the "
              "two models are too correlated for reweighting to matter.")
        print(classification_report(labels, w_preds, target_names=config.CLASS_NAMES))


if __name__ == "__main__":
    main()