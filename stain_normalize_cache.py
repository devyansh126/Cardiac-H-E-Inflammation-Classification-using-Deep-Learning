"""
stain_normalize_cache.py
Macenko stain normalization -- maps every image's H&E stain colors onto a
single fixed reference stain matrix, so patient-to-patient staining/scanner
differences stop being a usable shortcut feature.

WHY THIS EXISTS (see analyze_images.py results):
  - 25/33 patients are label-PURE (all-Inflammatory or all-Non-inflammatory),
    so case_id and label are almost the same variable.
  - Between-patient stain/color variability is ~98% as large as within-class
    variability (analyze_images.py part B) -- color/stain features barely
    separate "class" from "which patient".
  - A classical model trained on color/stain features got ~50% val accuracy
    but collapsed to 8-32% accuracy on the true holdout patient (19-29311)
    -- i.e. those features memorize patient stain identity and do NOT
    generalize. The CNN's known 95%->35% holdout collapse is almost
    certainly the same confound.
Macenko normalization (NOT Reinhard -- Reinhard was tried before and hurt
accuracy 90%->81%, see clahe_cache.py's docstring) removes stain-intensity/
color-balance differences between patients while preserving structural
(nuclei, texture) content, which is what should carry the real biological
signal.

Method (per image):
  1. Convert RGB -> optical density (OD) space.
  2. Drop near-white/background pixels (OD below `beta`).
  3. Find the plane spanned by the top-2 OD eigenvectors; project pixels
     onto it and take the extreme angular percentiles as the two stain
     vectors (Hematoxylin, Eosin) for THIS image.
  4. Solve for stain concentrations, re-express them against a FIXED
     reference stain matrix + fixed reference max concentrations (standard
     Macenko reference values), reconstruct the image.
  5. If the image is background-dominated (e.g. a mostly-white tile) and
     the SVD/percentile step is degenerate, fall back to the original
     image untouched rather than producing garbage -- logged, not silent.

Run ONCE before retraining (after preprocess_cache.py has built cached_path):
    python stain_normalize_cache.py

Adds a 'stain_norm_path' column to file_index.csv. dataset.py's
image_source="stain_norm" reads from it; config.DEFAULT_IMAGE_SOURCE now
defaults to "stain_norm" -- see config.py comments.
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import config

STAIN_NORM_DIR = config.STAIN_NORM_DIR

# ---------------------------------------------------------------------------
# Standard Macenko reference stain matrix + max concentrations (H&E).
# These are the widely-used reference values from Macenko et al. 2009 --
# every image gets remapped onto THIS fixed target, which is what actually
# kills the patient-specific color confound (a per-batch "average target"
# would just re-introduce a milder version of the same problem).
# ---------------------------------------------------------------------------
HE_REF = np.array([[0.5626, 0.2159],
                    [0.7201, 0.8012],
                    [0.4062, 0.5581]])
MAX_C_REF = np.array([1.9705, 1.0308])

Io = 240      # background transmitted light intensity
ALPHA = 1     # percentile cutoff for pseudo-min/max stain vectors
BETA = 0.15   # OD threshold below which a pixel is treated as background


def macenko_normalize(img_rgb, Io=Io, alpha=ALPHA, beta=BETA,
                       he_ref=HE_REF, max_c_ref=MAX_C_REF):
    """
    img_rgb: HxWx3 uint8 RGB image.
    Returns a normalized HxWx3 uint8 RGB image, or None if the image is too
    background-dominated to fit a stain matrix (caller should fall back).
    """
    h, w, _ = img_rgb.shape
    img = img_rgb.reshape(-1, 3).astype(np.float64)

    # RGB -> optical density
    od = -np.log10((img + 1.0) / Io)

    # keep only pixels with enough stain signal
    od_hat = od[~np.any(od < beta, axis=1)]
    if od_hat.shape[0] < 50:
        return None  # essentially all background -- nothing to normalize against

    # eigenvectors of the covariance of the OD tuples
    cov = np.cov(od_hat.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # top 2 eigenvectors (largest eigenvalues)
    eigvecs = eigvecs[:, [-1, -2]]

    # project onto the plane, get angles
    that = od_hat.dot(eigvecs)
    phi = np.arctan2(that[:, 1], that[:, 0])

    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    v_min = eigvecs.dot(np.array([np.cos(min_phi), np.sin(min_phi)]))
    v_max = eigvecs.dot(np.array([np.cos(max_phi), np.sin(max_phi)]))

    # Heuristically order as (Hematoxylin, Eosin): H has larger R-channel OD
    if v_min[0] > v_max[0]:
        he = np.array([v_min, v_max]).T
    else:
        he = np.array([v_max, v_min]).T

    # solve for concentrations of this image's own stain vectors
    y = od.T
    try:
        c = np.linalg.lstsq(he, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    max_c = np.percentile(c, 99, axis=1)
    max_c[max_c == 0] = 1e-6  # guard against divide-by-zero on degenerate tiles

    # rescale concentrations onto the fixed reference's concentration scale
    c_norm = c / max_c[:, None] * max_c_ref[:, None]

    # reconstruct using the FIXED reference stain matrix.
    # NOTE: forward transform above is base-10 (-log10), so the inverse
    # MUST be base-10 exponentiation (10**-x), not natural exp -- using
    # np.exp here was a real bug in an earlier version of this file: it
    # silently over-brightened/washed out every cached image by ~2x at
    # typical stain density (10**-0.5=0.32 vs exp(-0.5)=0.61), which would
    # show up exactly as "the model can't fit the training data well"
    # since the images had lost real contrast/detail before training ever
    # started.
    od_norm = he_ref.dot(c_norm)
    img_norm = Io * np.power(10.0, -od_norm) - 1.0
    img_norm = np.clip(img_norm, 0, 255).astype(np.uint8)
    img_norm = img_norm.T.reshape(h, w, 3)
    return img_norm


def build_stain_norm_cache(force=False):
    os.makedirs(STAIN_NORM_DIR, exist_ok=True)
    df = pd.read_csv(config.INDEX_CSV)

    # Source: the resized cache (fast to read), never the raw originals.
    source_col = "cached_path" if "cached_path" in df.columns and df["cached_path"].notna().all() else "path"
    if source_col == "path":
        print("WARNING: 'cached_path' not fully populated -- run preprocess_cache.py "
              "first. Falling back to raw 'path' (slow).")
    print(f"Using '{source_col}' as source for stain normalization -> {STAIN_NORM_DIR}")

    stain_norm_paths = []
    n_built, n_skipped, n_fallback = 0, 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        src_path = row[source_col]
        out_path = os.path.join(STAIN_NORM_DIR, os.path.basename(src_path))

        if os.path.exists(out_path) and not force:
            n_skipped += 1
        else:
            img_bgr = cv2.imread(src_path)
            if img_bgr is None:
                stain_norm_paths.append(None)
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            norm_rgb = macenko_normalize(img_rgb)
            if norm_rgb is None:
                # background-dominated / degenerate tile -- keep original
                # rather than write a broken image, and say so.
                norm_bgr = img_bgr
                n_fallback += 1
            else:
                norm_bgr = cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_path, norm_bgr)
            n_built += 1

        stain_norm_paths.append(out_path)

    df["stain_norm_path"] = stain_norm_paths
    df.to_csv(config.INDEX_CSV, index=False)

    print(f"\nDone. Built: {n_built} (of which fell back to original: {n_fallback}) "
          f"| Already cached (skipped): {n_skipped}")
    print(f"Updated {config.INDEX_CSV} with 'stain_norm_path' column.")


if __name__ == "__main__":
    build_stain_norm_cache()