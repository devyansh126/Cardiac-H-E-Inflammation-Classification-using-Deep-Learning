"""
dataset.py
Builds/loads the file index, defines the PyTorch Dataset class, transforms,
and BOTH split strategies:

  - get_patient_stratified_datasets()  <- PRIMARY, use this one
        Groups by case_id (GroupShuffleSplit) so no single patient's images
        can appear in both train and val. Checks and reports the resulting
        class balance (group splitting doesn't guarantee label ratios), and
        retries a handful of seeds to find a well-balanced split.

  - get_datasets()                     <- LEGACY / reference only
        Plain image-wise stratified split. This is the split behind the
        original 88.3%/90.4%-TTA numbers, and is also what exposed the
        95%-collapses-to-35% generalization gap on the dominant patient.
        Kept only for reproducing/comparing against those numbers — do not
        use it for the "clean" final model.

Usage:
    from dataset import build_index, get_patient_stratified_datasets, get_dataloaders
    df = build_index()
    train_ds, val_ds = get_patient_stratified_datasets(df)
    train_loader, val_loader = get_dataloaders(train_ds, val_ds)

Image source selection (image_source param, also see config.DEFAULT_IMAGE_SOURCE)
-----------------------------------------------------------------------------
    "stain_norm" (default) -- Macenko stain-normalized cache. This is now
                the main pipeline -- see config.py's DEFAULT_IMAGE_SOURCE
                comment and stain_normalize_cache.py's docstring for why:
                analyze_images.py found the un-normalized pipeline's
                accuracy was very likely riding a patient/stain-color
                confound (25/33 patients are label-pure). Requires running
                stain_normalize_cache.py first.
    "cached" -- resized, no stain norm, no CLAHE. The OLD baseline pipeline
                (88.3%/90.4%-TTA numbers) -- kept for direct comparison.
    "clahe"  -- forces clahe_path. Raises if missing, so a CLAHE experiment
                can never silently fall back to un-preprocessed images.
    "raw"    -- forces the original huge 'path' column. Avoid for training.
    "auto"   -- stain_norm_path > clahe_path > cached_path > path, first
                that exists. NOT the default: with multiple cache columns
                populated, "auto" would silently pick whichever pipeline
                happens to exist first instead of the one you intended.
                Only use "auto" if you explicitly want that fallback
                behavior and understand the implication.
"""

import os
import glob
import re
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import config


# ---------------------------------------------------------------------------
# Step 1: Build / load the file index
# ---------------------------------------------------------------------------
def extract_case_id(filename):
    """Extracts the leading accession/case number, e.g. '13-21062' from a filename."""
    name = os.path.basename(filename)
    match = re.match(r"^(\d+-\d+)", name)
    return match.group(1) if match else "UNKNOWN"


def build_index(force_rebuild=False):
    """
    Loads config.INDEX_CSV if it exists (preserving cached_path / clahe_path
    columns added by preprocess_cache.py / clahe_cache.py). Only rescans the
    raw folders from scratch if the index is missing or force_rebuild=True —
    a blind rescan would throw away those cache columns and every image
    would need re-caching.
    """
    if os.path.exists(config.INDEX_CSV) and not force_rebuild:
        print(f"Loading existing index from {config.INDEX_CSV}")
        df = pd.read_csv(config.INDEX_CSV)
        missing_cache_cols = [c for c in ("cached_path", "stain_norm_path", "clahe_path") if c not in df.columns]
        if missing_cache_cols:
            print(f"  Note: index is missing columns {missing_cache_cols} "
                  f"(run preprocess_cache.py / stain_normalize_cache.py / clahe_cache.py to populate them).")
        return df

    print("No existing index found (or force_rebuild=True) -- scanning raw folders.")
    records = []
    for folder, label_name in [(config.INFLAM_DIR, "Inflammatory"),
                                 (config.NONINFLAM_DIR, "Non-inflammatory")]:
        files = glob.glob(os.path.join(folder, "*"))
        for f in files:
            records.append({
                "path": f,
                "label": config.LABEL_MAP[label_name],
                "label_name": label_name,
                "case_id": extract_case_id(f),
            })

    df = pd.DataFrame(records)
    df.to_csv(config.INDEX_CSV, index=False)
    print(f"Built fresh index with {len(df)} images, saved to {config.INDEX_CSV}")
    print("  cached_path / clahe_path are NOT set yet -- run preprocess_cache.py before training.")
    return df


# ---------------------------------------------------------------------------
# Step 2: PyTorch Dataset class
# ---------------------------------------------------------------------------
VALID_IMAGE_SOURCES = ("stain_norm", "cached", "clahe", "raw", "auto")


def _resolve_image_path(row, image_source):
    if image_source == "raw":
        return row["path"]

    if image_source == "cached":
        if "cached_path" not in row or pd.isna(row["cached_path"]):
            raise ValueError(
                "image_source='cached' but 'cached_path' is missing for this row. "
                "Run preprocess_cache.py first."
            )
        return row["cached_path"]

    if image_source == "stain_norm":
        if "stain_norm_path" not in row or pd.isna(row["stain_norm_path"]):
            raise ValueError(
                "image_source='stain_norm' but 'stain_norm_path' is missing for this row. "
                "Run stain_normalize_cache.py first (it adds the 'stain_norm_path' column)."
            )
        return row["stain_norm_path"]

    if image_source == "clahe":
        if "clahe_path" not in row or pd.isna(row["clahe_path"]):
            raise ValueError(
                "image_source='clahe' but 'clahe_path' is missing for this row. "
                "Run clahe_cache.py first (it adds the 'clahe_path' column)."
            )
        return row["clahe_path"]

    # "auto": stain_norm_path > clahe_path > cached_path > path -- opt-in
    # only, see module docstring.
    if "stain_norm_path" in row and pd.notna(row["stain_norm_path"]):
        return row["stain_norm_path"]
    if "clahe_path" in row and pd.notna(row["clahe_path"]):
        return row["clahe_path"]
    if "cached_path" in row and pd.notna(row["cached_path"]):
        return row["cached_path"]
    return row["path"]


class InflammationDataset(Dataset):
    def __init__(self, dataframe, transform=None, image_source=None, feature_columns=None):
        """
        feature_columns: optional list of column names (e.g.
        engineered_features.FEATURE_COLUMNS) to also return as a float32
        tensor for feature-fusion models (see model.FusionResNet50 /
        train_fusion.py). Default None preserves the original behavior
        (__getitem__ returns (img, label)) so every existing caller that
        doesn't ask for features is completely unaffected.
        """
        image_source = image_source or config.DEFAULT_IMAGE_SOURCE
        if image_source not in VALID_IMAGE_SOURCES:
            raise ValueError(f"image_source must be one of {VALID_IMAGE_SOURCES}, got '{image_source}'")
        if feature_columns is not None:
            missing = [c for c in feature_columns if c not in dataframe.columns]
            if missing:
                raise ValueError(
                    f"feature_columns {missing} not found in dataframe -- "
                    f"run engineered_features.py first."
                )
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_source = image_source
        self.feature_columns = feature_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = _resolve_image_path(row, self.image_source)
        img = Image.open(img_path).convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        if self.feature_columns is not None:
            feat = torch.tensor(row[self.feature_columns].values.astype("float32"))
            return img, feat, label
        return img, label


# ---------------------------------------------------------------------------
# Step 3: Transforms
# ---------------------------------------------------------------------------
def get_transforms():
    # RandomResizedCrop was tested here and reverted: an isolated test
    # (crop only, mixup off) still came in below the plain-Resize baseline
    # (Tier4 val 86.6%/TTA 87.1% vs baseline 90.1%/88.8%). On these tiles,
    # cropping out 0-25% of the frame apparently removes informative tissue
    # context more often than it adds a useful new view, and 597 training
    # images over ~20 epochs isn't enough to average that out. Back to
    # plain Resize -- the checkpoint-fix-only run remains the best result.
    train_ops = [
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
    ]

    # Second, independent defense against the patient/stain-color confound
    # (see stain_normalize_cache.py): stain normalization fixes the STATIC
    # color difference between patients at cache-build time; jitter forces
    # the model to not lean on absolute color/brightness within any given
    # training batch either. Hue range is kept small on purpose -- large
    # hue swings can flip Hematoxylin-like and Eosin-like hues into each
    # other, which would corrupt the actual biological signal.
    if getattr(config, "COLOR_JITTER_ENABLED", False):
        train_ops.append(transforms.ColorJitter(
            brightness=config.COLOR_JITTER_BRIGHTNESS,
            contrast=config.COLOR_JITTER_CONTRAST,
            saturation=config.COLOR_JITTER_SATURATION,
            hue=config.COLOR_JITTER_HUE,
        ))

    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]
    train_transform = transforms.Compose(train_ops)

    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])

    return train_transform, val_transform


# ---------------------------------------------------------------------------
# Step 4a: PRIMARY split -- patient-stratified (GroupShuffleSplit on case_id)
# ---------------------------------------------------------------------------
def _class_ratio(df):
    """Fraction of rows that are label==1 (Inflammatory)."""
    return (df["label"] == 1).mean()


def _find_balanced_group_split(df, val_split, group_col, base_seed,
                                 max_tries, tolerance, forced_val_patients=None):
    """
    GroupShuffleSplit doesn't stratify by label, so with case_id groups of
    very uneven size (esp. the dominant Non-inflammatory patient) a single
    split can land badly imbalanced. Try several seeds, score each by
    |val_ratio - overall_ratio|, keep the best. Returns (train_df, val_df,
    seed_used, achieved_gap).

    forced_val_patients: case_ids that MUST end up in val on every attempt.
    Without this, the balance search will happily learn to keep a large,
    lopsided patient (like the known dominant Non-inflammatory patient) in
    TRAIN instead, because that's an easier way to hit a low class-ratio
    gap -- which means you'd never actually validate against your hardest,
    most important patient. Forcing it into val trades a bit of class
    balance for the split actually testing what you need it to test.
    """
    overall_ratio = _class_ratio(df)
    forced_val_patients = forced_val_patients or []

    forced_mask = df[group_col].isin(forced_val_patients)
    forced_df = df[forced_mask]
    rest_df = df[~forced_mask]

    if len(forced_df) == 0:
        # nothing forced -- original unconstrained search
        best = None
        for attempt in range(max_tries):
            seed = base_seed + attempt
            splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
            train_idx, val_idx = next(splitter.split(df, groups=df[group_col]))
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
            gap = abs(_class_ratio(val_df) - overall_ratio)
            if best is None or gap < best[0]:
                best = (gap, seed, train_df, val_df)
            if gap <= tolerance:
                break
        gap, seed, train_df, val_df = best
        return train_df, val_df, seed, gap

    # forced patient(s) present -- they always land in val; search seeds only
    # over how the REST of the patients split, so the total val class ratio
    # (forced + sampled rest) lands as close to overall as we can manage.
    target_val_n = val_split * len(df)
    remaining_val_n = target_val_n - len(forced_df)

    if remaining_val_n <= 0:
        # forced patient(s) alone already meet/exceed the val quota -- put
        # everything else in train, no further split needed.
        gap = abs(_class_ratio(forced_df) - overall_ratio)
        return rest_df, forced_df, None, gap

    rest_val_split = min(max(remaining_val_n / len(rest_df), 0.05), 0.95)

    best = None
    for attempt in range(max_tries):
        seed = base_seed + attempt
        splitter = GroupShuffleSplit(n_splits=1, test_size=rest_val_split, random_state=seed)
        rest_train_idx, rest_val_idx = next(splitter.split(rest_df, groups=rest_df[group_col]))
        rest_train_df, rest_val_df = rest_df.iloc[rest_train_idx], rest_df.iloc[rest_val_idx]

        train_df = rest_train_df
        val_df = pd.concat([forced_df, rest_val_df])

        gap = abs(_class_ratio(val_df) - overall_ratio)
        if best is None or gap < best[0]:
            best = (gap, seed, train_df, val_df)
        if gap <= tolerance:
            break

    gap, seed, train_df, val_df = best
    return train_df, val_df, seed, gap


def get_patient_stratified_datasets(df, val_split=None, random_seed=None,
                                      image_source=None, group_col=None,
                                      max_tries=None, tolerance=None,
                                      forced_val_patients=None):
    """
    PRIMARY split for the clean pipeline. Groups by case_id so no patient's
    images leak across train/val, then reports (and actively searches for)
    a reasonably class-balanced split, since group-based splitting doesn't
    preserve label ratios automatically.

    forced_val_patients: opt-in only (default: none). Forcing a patient into
    val is USUALLY THE WRONG TOOL if that patient is large relative to your
    val budget — it can replace val entirely rather than diversify it (see
    the comment on config.ALWAYS_HOLDOUT_PATIENTS for why). For known
    hard/lopsided patients, exclude them from `df` before calling this
    function and evaluate them as a separate holdout instead — that's what
    train.py does by default via config.ALWAYS_HOLDOUT_PATIENTS.
    """
    val_split = val_split if val_split is not None else config.VAL_SPLIT
    random_seed = random_seed if random_seed is not None else config.RANDOM_SEED
    group_col = group_col or config.GROUP_COL
    max_tries = max_tries if max_tries is not None else config.SPLIT_BALANCE_MAX_TRIES
    tolerance = tolerance if tolerance is not None else config.SPLIT_BALANCE_TOLERANCE
    forced_val_patients = forced_val_patients or []

    train_df, val_df, seed_used, gap = _find_balanced_group_split(
        df, val_split, group_col, random_seed, max_tries, tolerance, forced_val_patients
    )

    overall_ratio = _class_ratio(df)
    train_patients = train_df[group_col].nunique()
    val_patients = val_df[group_col].nunique()
    overlap = set(train_df[group_col]) & set(val_df[group_col])

    if forced_val_patients:
        print(f"Forced into val (never left to the balance search): {forced_val_patients}")
    print(f"Patient-stratified split (GroupShuffleSplit, seed={seed_used}, "
          f"searched up to {max_tries} seeds):")
    print(f"  Train: {len(train_df)} images, {train_patients} patients "
          f"({train_df['label_name'].value_counts().to_dict()})")
    print(f"  Val:   {len(val_df)} images, {val_patients} patients "
          f"({val_df['label_name'].value_counts().to_dict()})")
    print(f"  Overall Inflammatory ratio: {overall_ratio:.3f} | "
          f"Val Inflammatory ratio: {_class_ratio(val_df):.3f} | gap: {gap:.3f} "
          f"(tolerance: {tolerance})")
    if gap > tolerance:
        print(f"  WARNING: could not find a split within tolerance after {max_tries} tries -- "
              f"using the best one found (gap={gap:.3f}). This is expected if one patient "
              f"dominates a class; consider it a reportable limitation, not a bug.")
    assert not overlap, f"Patient leakage between train/val: {overlap}"
    print(f"  Patient leakage check: OK (0 overlapping patients)")

    train_transform, val_transform = get_transforms()
    train_ds = InflammationDataset(train_df, transform=train_transform, image_source=image_source)
    val_ds = InflammationDataset(val_df, transform=val_transform, image_source=image_source)
    return train_ds, val_ds


def get_patient_holdout_dataset(df, holdout_case_id, image_source=None):
    """Builds a dataset from ONE specific patient, entirely held out for a true generalization test."""
    holdout_df = df[df["case_id"] == holdout_case_id]
    print(f"Holdout patient {holdout_case_id}: {len(holdout_df)} images")
    _, val_transform = get_transforms()
    return InflammationDataset(holdout_df, transform=val_transform, image_source=image_source)


def split_patient_partial_holdout(df, case_id, train_fraction, seed=None):
    """
    Splits ONE patient's own images into a train portion and a val portion,
    instead of excluding the patient entirely (see config.PARTIAL_HOLDOUT_PATIENTS
    for why/when to use this over full exclusion). Stratifies by label
    within the patient where both classes have >=2 images; falls back to a
    plain random split otherwise (e.g. a label-pure patient).

    Returns (patient_train_df, patient_val_df). NOT a generalization test --
    same tissue block / staining batch can appear on both sides, exactly
    like the rest of the image-wise split.
    """
    seed = seed if seed is not None else config.RANDOM_SEED
    patient_df = df[df["case_id"] == case_id]
    if len(patient_df) == 0:
        print(f"  WARNING: partial-holdout patient {case_id} not found in data -- skipping.")
        return patient_df, patient_df

    can_stratify = patient_df["label"].nunique() > 1 and patient_df["label"].value_counts().min() >= 2
    try:
        p_train, p_val = train_test_split(
            patient_df, train_size=train_fraction, random_state=seed,
            stratify=patient_df["label"] if can_stratify else None,
        )
    except ValueError:
        # e.g. too few images for the requested split ratio -- fall back to unstratified
        p_train, p_val = train_test_split(patient_df, train_size=train_fraction, random_state=seed)

    print(f"  Partial-holdout patient {case_id}: {len(patient_df)} images -> "
          f"{len(p_train)} train / {len(p_val)} val ({train_fraction:.0%}/{1-train_fraction:.0%}) "
          f"-- NOT a generalization test, same patient on both sides.")
    return p_train, p_val


# ---------------------------------------------------------------------------
# Step 4b: LEGACY split -- plain image-wise stratified (reference only)
# ---------------------------------------------------------------------------
def get_datasets(df, val_split=None, random_seed=None, image_source=None):
    """
    Image-wise stratified split (ignores case_id). Selected via
    config.SPLIT_STRATEGY = "image_wise". Patients CAN appear in both train
    and val under this split -- pair with the ALWAYS_HOLDOUT_PATIENTS true
    holdout in train.py for a leakage-free generalization check, since val
    accuracy alone under this split isn't one (see config.py's
    SPLIT_STRATEGY comment for why).
    """
    val_split = val_split if val_split is not None else config.VAL_SPLIT
    random_seed = random_seed if random_seed is not None else config.RANDOM_SEED

    train_df, val_df = train_test_split(
        df, test_size=val_split, stratify=df["label"], random_state=random_seed
    )

    print(f"[image-wise split] Train: {len(train_df)} images "
          f"({train_df['label_name'].value_counts().to_dict()})")
    print(f"[image-wise split] Val:   {len(val_df)} images "
          f"({val_df['label_name'].value_counts().to_dict()})")
    train_patients, val_patients = set(train_df["case_id"]), set(val_df["case_id"])
    overlap = train_patients & val_patients
    print(f"[image-wise split] Patients in train: {len(train_patients)} | val: {len(val_patients)} "
          f"| overlapping (expected under image-wise split): {len(overlap)}")

    train_transform, val_transform = get_transforms()
    train_ds = InflammationDataset(train_df, transform=train_transform, image_source=image_source)
    val_ds = InflammationDataset(val_df, transform=val_transform, image_source=image_source)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Step 5: DataLoaders
# ---------------------------------------------------------------------------
def get_dataloaders(train_ds, val_ds, batch_size=None, num_workers=None):
    batch_size = batch_size if batch_size is not None else config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = build_index()
    print("\nLabel distribution:\n", df["label_name"].value_counts())
    print("\nUnique patients:", df["case_id"].nunique())

    train_ds, val_ds = get_patient_stratified_datasets(df)
    train_loader, val_loader = get_dataloaders(train_ds, val_ds)

    images, labels = next(iter(train_loader))
    print("\nBatch shape:", images.shape)
    print("Batch labels:", labels)