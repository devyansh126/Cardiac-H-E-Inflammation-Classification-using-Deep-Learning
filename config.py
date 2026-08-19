"""
config.py
Shared configuration for the cardiac H&E inflammation classification project.

This version reflects the "clean pipeline" decisions from the project handoff:
  - ResNet50 only (won the fair architecture comparison)
  - Patient-stratified splitting is the primary split strategy (see dataset.py:
    get_patient_stratified_datasets)
  - 4-tier progressive unfreezing, one LR per tier
  - class-weighted loss, early stopping, TTA at inference

Edit values here rather than scattering hardcoded paths/constants across files.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = r"D:\BioInformatics\IITRoorkeProject"
INFLAM_DIR = os.path.join(BASE_DIR, "Inflammatory")
NONINFLAM_DIR = os.path.join(BASE_DIR, "Non-inflammatory")

# Existing index — dataset.py loads this rather than rescanning folders, so it
# never loses the cached_path / clahe_path columns already populated on it.
INDEX_CSV = os.path.join(BASE_DIR, "file_index.csv")

# Resized-image cache (built by preprocess_cache.py). ALWAYS train off this
# (or stain_norm_path/clahe_path), never the raw 2748x2750 'path' column.
#
# Cache dir name includes CACHE_SIZE on purpose: preprocess_cache.py is
# idempotent by basename, so if you change CACHE_SIZE without changing the
# dir name, old-resolution files at those basenames would just get silently
# reused (stale cache) instead of rebuilt at the new size. Keying the dir by
# size means changing CACHE_SIZE always produces a fresh cache automatically.
CACHE_SIZE = 512  # bumped from 256 -- see resolution-loss diagnostic:
                    # 256px retained only ~9% of native high-frequency detail
                    # and shrank a lymphocyte nucleus to ~1.5-3px (unresolvable).
                    # 512px cuts the downsize factor from 10.7x to ~5.4x.
CACHE_DIR = os.path.join(BASE_DIR, f"cache_resized_{CACHE_SIZE}")

# Macenko stain-normalized cache (built by stain_normalize_cache.py). See
# that script's docstring: analyze_images.py found 25/33 patients are
# label-pure and between-patient stain variability is ~98% as large as
# within-class variability -- i.e. color/stain is a usable shortcut for
# "which patient" that a classical model (and almost certainly the CNN)
# was exploiting instead of real biology. This maps every image onto one
# fixed reference stain matrix to remove that shortcut.
STAIN_NORM_DIR = os.path.join(BASE_DIR, "cache_stain_norm")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------
LABEL_MAP = {"Inflammatory": 1, "Non-inflammatory": 0}
CLASS_NAMES = ["Non-inflammatory", "Inflammatory"]  # index 0, index 1

# ---------------------------------------------------------------------------
# Architecture — ResNet50 only in the main pipeline (won the capacity-matched
# comparison; DenseNet121 showed a persistent inflammatory-recall collapse
# and was deprioritized). compare_architectures.py can still request other
# architectures from model.py directly for one-off comparisons.
# ---------------------------------------------------------------------------
ARCHITECTURE = "resnet50"

# ---------------------------------------------------------------------------
# Image / dataloader settings
# ---------------------------------------------------------------------------
IMG_SIZE = 384          # bumped from 224 alongside CACHE_SIZE -- keeps native downsize
                          # to ~5.4x total instead of ~10.7x/12.3x
BATCH_SIZE = 8          # reduced from 16 -- 384px activations use ~3x the memory of
                          # 224px on this 4GB card. If you hit CUDA out-of-memory,
                          # this is the first thing to drop further (try 4).
NUM_WORKERS = 0        # MUST stay 0 on this Windows/Jupyter setup — DataLoader hangs otherwise

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Automatic Mixed Precision -- roughly halves activation memory and speeds up
# training on this GPU (RTX 2050 / Ampere has fp16 tensor cores), which buys
# back some of the batch-size headroom lost to the larger images. Safe
# default; disable only if you hit numerical instability (NaN losses).
AMP_ENABLED = True

# Which image column InflammationDataset reads by default.
#   "stain_norm" -- Macenko stain-normalized (see stain_normalize_cache.py).
#                   NEW DEFAULT. The 88.3%/90.4%-TTA numbers were gotten on
#                   "cached" (no stain normalization), but analyze_images.py
#                   showed those numbers were very likely inflated by a
#                   patient/stain-color confound (25/33 patients are label-
#                   pure; a classical model on color features alone got
#                   ~50% val acc but 8-32% on the true holdout patient).
#                   Requires running stain_normalize_cache.py first.
#   "cached"     -- resized, no stain norm, no CLAHE. The OLD baseline.
#                   Kept for direct comparison against the new pipeline --
#                   if you want to reproduce the original 88.3%/90.4%
#                   numbers or re-run the confound diagnostic, set this.
#   "clahe"      -- forces clahe_path (local contrast only, no color fix --
#                   does NOT address the stain confound above).
#   "raw"        -- forces the original huge 'path' column. Avoid for training.
#   "auto"       -- stain_norm_path > clahe_path > cached_path > path, first
#                   that exists. Opt-in only, not the default.
DEFAULT_IMAGE_SOURCE = "stain_norm"

# Color-jitter augmentation, applied on top of whichever image_source is in
# use (train split only). This is a second, independent line of defense
# against the same stain/patient confound: even after Macenko normalization
# reduces the STATIC color difference between patients, jitter forces the
# model not to over-rely on absolute color/brightness for any single
# training batch, rather than only removing it once at cache-build time.
COLOR_JITTER_ENABLED = True
COLOR_JITTER_BRIGHTNESS = 0.15
COLOR_JITTER_CONTRAST = 0.15
COLOR_JITTER_SATURATION = 0.10
COLOR_JITTER_HUE = 0.02  # kept small -- hue swings can flip H&E's meaning (H vs E)

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Split settings
# ---------------------------------------------------------------------------
# "image_wise": plain stratified split by label, ignoring case_id. Patients
#   CAN appear in both train and val (except whichever are pinned into
#   ALWAYS_HOLDOUT_PATIENTS below, which are excluded from train+val
#   regardless of this setting and evaluated as a separate true holdout).
#   This is the split your supervisor asked for. Be aware of what it does:
#   with 25/33 patients label-pure, val accuracy under this setting is not
#   a clean estimate of generalization to an unseen patient -- the
#   dedicated ALWAYS_HOLDOUT_PATIENTS evaluation below is what tells you
#   that number instead, and is worth reporting alongside the image-wise
#   val accuracy for that reason.
# "patient_wise": GroupShuffleSplit by case_id -- no patient's images
#   appear in both train and val. Gives a lower, but leakage-free, estimate.
SPLIT_STRATEGY = "image_wise"

VAL_SPLIT = 0.2
GROUP_COL = "case_id"          # patient-stratified splitting groups by this column
SPLIT_BALANCE_MAX_TRIES = 30    # GroupShuffleSplit doesn't guarantee label ratio;
                                  # try this many seeds and keep the best-balanced one
SPLIT_BALANCE_TOLERANCE = 0.08  # acceptable |val class ratio - overall class ratio|
                                  # before we stop searching and just take the best found

# Known problem patient: dominates ~43% of the Non-inflammatory class even
# after dedup. Not excluded by default — patient_holdout_check.py uses this
# to characterize the generalization gap deliberately. Documented here so
# no script has to hardcode the ID from scratch.
KNOWN_DOMINANT_PATIENT = "19-29311"

# Patients ALWAYS excluded from train+val entirely and evaluated separately
# as a dedicated holdout (see train.py). Two earlier approaches were tried
# and rejected before this one:
#   1. Leaving the split-balance search unconstrained: it tends to tuck
#      large lopsided patients like KNOWN_DOMINANT_PATIENT into TRAIN,
#      since that's an easy way to hit good val class-ratio balance — but
#      it means the pipeline never validates against your hardest patient.
#   2. Forcing the patient into val instead: fails differently. Patient
#      19-29311 alone (205 images) already exceeds the ~190-image val
#      budget, so "forcing into val" doesn't add it to a diverse val set —
#      it REPLACES val entirely with that one 97.5%-one-class patient,
#      which would make early-stopping/tier decisions swing on a single
#      lopsided patient instead of measuring general performance.
# This setting instead removes the patient from the train/val pool
# completely (val stays diverse across the remaining patients, stable for
# early stopping), and train.py evaluates it as a separate always-on
# holdout metric — the same design patient_holdout_check.py already used,
# now built into the default pipeline instead of a one-off script.
#
# NOTE: any patient listed in PARTIAL_HOLDOUT_PATIENTS below is handled
# there INSTEAD of here, even if also listed in this list -- partial
# takes precedence.
ALWAYS_HOLDOUT_PATIENTS = [KNOWN_DOMINANT_PATIENT]

# PARTIAL_HOLDOUT_PATIENTS: patient_id -> fraction of THAT PATIENT's OWN
# images to put in train (remainder goes to val), instead of excluding the
# patient entirely. Splits within the patient are stratified by label where
# possible.
#
# IMPORTANT -- read before enabling: this trades away your only
# leakage-free generalization check. Once a patient's images are split
# train/val this way, "val accuracy on this patient" reflects the model
# partly recognizing THIS patient's specific stain/tissue appearance
# (same tissue block, same staining batch, adjacent tiles) -- it is NOT a
# test of generalization to an unseen patient, for the same reason
# image-wise splitting isn't for everyone else. train.py labels this
# patient's results accordingly rather than calling them "TRUE HOLDOUT".
#
# Rationale for using it anyway on KNOWN_DOMINANT_PATIENT specifically:
# it's ~43% of the entire Non-inflammatory class -- excluding it entirely
# starves the model of a large, legitimately different-looking chunk of
# that class, which is arguably why it was misclassified so heavily as
# Inflammatory. Splitting it 60/40 lets the model learn from most of that
# variation while still keeping ~80 of its images out of training for a
# rough same-patient consistency check.
#
# Set to {} to disable and fall back to full exclusion via
# ALWAYS_HOLDOUT_PATIENTS above.
PARTIAL_HOLDOUT_PATIENTS = {
    KNOWN_DOMINANT_PATIENT: 0.6,  # 60% train / 40% val, within this patient only
}
PARTIAL_HOLDOUT_SEED = 42

# ---------------------------------------------------------------------------
# 4-tier progressive unfreezing hyperparameters
# ---------------------------------------------------------------------------
# Raised from the original values -- train acc was plateauing at 0.85-0.90
# with train LOSS still visibly decreasing when early-stopping cut in
# (e.g. Tier 3/4 stopped at epoch 7-9/10 while train loss was still
# dropping every epoch) -- that's the optimizer not being given enough
# room, not the model lacking capacity. Higher LR + more epochs/patience
# below address that directly; LR_SCHEDULER (further down) gives the
# larger initial LR room to decay once val loss plateaus, instead of
# needing one fixed compromise LR for the whole tier.
TIER_LR = {
    1: 1e-3,   # frozen backbone, classifier head only (was 5e-4)
    2: 5e-5,   # last block unfrozen (was 1e-5)
    3: 3e-5,   # last two blocks unfrozen (was 1e-5)
    4: 1e-5,   # fully unfrozen -- smallest LR, protects early ImageNet features (was 5e-6)
}

TIER_MAX_EPOCHS = {
    1: 20,
    2: 20,
    3: 20,
    4: 20,
}

TIER_PATIENCE = 6  # early-stopping patience (on val loss) for every tier -- was 3,
                     # too aggressive given loss was often still trending down

# Per-tier LR scheduler: decays LR when val loss plateaus, so each tier can
# start at the higher TIER_LR above (for real progress early) and still
# settle down for fine convergence late, instead of needing one fixed LR
# for the whole tier's epoch budget.
LR_SCHEDULER_ENABLED = True
LR_SCHEDULER_FACTOR = 0.5     # halve LR on plateau
LR_SCHEDULER_PATIENCE = 2     # epochs of no val-loss improvement before decaying
LR_SCHEDULER_MIN_LR = 1e-7

GRAD_CLIP_MAX_NORM = 1.0

# ---------------------------------------------------------------------------
# Regularization -- added after the first full run with the fixed
# best-val-accuracy checkpointing showed a real overfitting gap emerging by
# Tier 3/4: train loss dropped to ~0.08-0.19 while val loss sat at
# ~0.43-0.64 (train acc 95-97% vs val acc 88-90%). Two standard, cheap
# fixes for exactly this pattern (large-capacity ResNet50 on a few hundred
# effective training images):
#
# DROPOUT_P / WEIGHT_DECAY: TESTED AND REVERTED. Two runs tried (0.4/1e-4,
# then 0.2/5e-5) and BOTH underperformed the no-regularization baseline
# (90.1% Tier-4 val acc) -- 87.0% and 89.2% respectively -- and the second,
# gentler attempt barely closed the train/val gap at all (6.9pp vs the
# baseline's 5.5pp). Two data points in the same direction means this
# dataset's train/val gap isn't primarily a dropout/weight-decay-fixable
# overfitting problem -- it's more likely dataset-size/domain-difficulty
# related. Disabling both; the best real model remains the
# checkpoint-fix-only 90.1% run. If you want to revisit this, try much
# smaller values (e.g. DROPOUT_P=0.05-0.1, WEIGHT_DECAY=0) rather than
# continuing to halve from here, or invest effort in feature fusion /
# more data instead -- see chat for that discussion.
DROPOUT_P = 0.0
WEIGHT_DECAY = 0.0

# LABEL_SMOOTHING: a DIFFERENT regularization mechanism than the two above
# -- instead of reducing model capacity, it caps how confident the loss
# function ever rewards the model for being (target becomes e.g. 0.95/0.05
# instead of 1.0/0.0). Worth trying specifically because the observed
# problem isn't "model can't fit the data" (train acc is 96%+) -- it's
# "model is extremely overconfident on train and much less sure on val"
# (train loss 0.08 vs val loss 0.53, a 6x gap). That's exactly the failure
# mode label smoothing targets; dropout/weight-decay target a different
# failure mode (excess capacity) that testing showed isn't the issue here.
LABEL_SMOOTHING = 0.1

# MIXUP_ALPHA: blends pairs of training images and their labels. TESTED AND
# REVERTED alongside RandomResizedCrop -- with it on, Tier 1's frozen-
# backbone linear probe couldn't even fit blended inputs (train acc stuck
# at 0.62-0.69 vs baseline's climb to 0.80+), and every downstream tier
# inherited that damage (Tier4 val 87.9%/TTA 86.6%, both below baseline).
# Left in the code (0.0 = fully disabled) in case a future experiment with
# a longer epoch budget wants to revisit it, but don't re-enable without
# re-testing tier-by-tier train accuracy, not just the final number.
MIXUP_ALPHA = 0.0

# ---------------------------------------------------------------------------
# Tiny-batch stabilization knobs (RTX 2050, 4GB -> BATCH_SIZE=8). BOTH ARE
# OFF BY DEFAULT -- tested on this dataset and made things WORSE, not
# better, so don't re-enable without re-testing:
#
# FREEZE_BN_STATS=True regressed val AND holdout accuracy at every tier,
# including Tier 1 (where no BN params are even trainable). Reason: H&E
# tissue's color/intensity distribution is very different from ImageNet's
# natural photos, and letting BatchNorm's running mean/var drift toward
# the new domain during forward passes was doing useful, "free" domain
# adaptation. Freezing them to ImageNet statistics removed that benefit --
# the batch-noise problem it was meant to fix turned out to matter less
# than this domain-shift adaptation.
FREEZE_BN_STATS = False

# GRAD_ACCUM_STEPS=4 cut optimizer steps per epoch by 4x. With only 10 max
# epochs and patience=3, that meant early stopping triggered before the
# model had enough weight UPDATES (not epochs) to converge -- undertrained,
# not more stable. Set to 1 to disable (each mini-batch steps the optimizer
# immediately, as before).
GRAD_ACCUM_STEPS = 1

# ---------------------------------------------------------------------------
# Test-Time Augmentation — confirmed +1.6pp (88.8% -> 90.4%) on the full-data
# ResNet50 Tier-4 model. Cheap, keep it on by default at final evaluation.
# ---------------------------------------------------------------------------
TTA_ENABLED = True
TTA_ROTATION_DEGREES = 10  # the "+-10 deg" views

# ---------------------------------------------------------------------------
# Sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("BASE_DIR exists:", os.path.exists(BASE_DIR))
    print("INDEX_CSV exists:", os.path.exists(INDEX_CSV))
    print("CACHE_DIR:", CACHE_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    print("Architecture:", ARCHITECTURE, "| Default image source:", DEFAULT_IMAGE_SOURCE)