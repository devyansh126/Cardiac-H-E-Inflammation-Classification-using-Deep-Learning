"""
preprocess_cache.py
One-time preprocessing step: resizes every source image (2748x2750) down to
a small cached copy at config.CACHE_SIZE, so training doesn't re-decode/
re-resize huge PNGs from disk every epoch.

Run this once (safe to re-run any time after manual cleanup / re-syncing
the index -- it skips anything already cached):
    python preprocess_cache.py

Idempotency note (fixed vs. earlier version): cache filenames are now keyed
off the SOURCE basename, not the DataFrame row index. The old idx-based
naming broke silently after manual file deletions reshuffled row indices
(sync_index_after_manual_cleanup.py renumbers rows) -- a file could end up
"cached" under a stale index number, or get needlessly re-cached. Filenames
in this dataset already encode patient/case/timestamp and are unique, so the
basename alone is a safe, stable cache key.
"""

import os
from PIL import Image
import pandas as pd
from tqdm import tqdm

import config


def build_cache(force=False):
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    df = pd.read_csv(config.INDEX_CSV)
    cached_paths = []
    n_skipped, n_built, n_failed = 0, 0, 0

    print(f"Resizing up to {len(df)} images to "
          f"{config.CACHE_SIZE}x{config.CACHE_SIZE} -> {config.CACHE_DIR}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        src_path = row["path"]
        cache_filename = os.path.basename(src_path)  # stable key, independent of row order
        cache_path = os.path.join(config.CACHE_DIR, cache_filename)

        if os.path.exists(cache_path) and not force:
            n_skipped += 1
        else:
            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize((config.CACHE_SIZE, config.CACHE_SIZE), Image.BILINEAR)
                img.save(cache_path, "PNG")
                n_built += 1
            except Exception as e:
                print(f"  FAILED to cache {src_path}: {e}")
                cache_path = None
                n_failed += 1

        cached_paths.append(cache_path)

    df["cached_path"] = cached_paths
    df.to_csv(config.INDEX_CSV, index=False)

    print(f"\nDone. Built: {n_built} | Already cached (skipped): {n_skipped} | Failed: {n_failed}")
    print(f"Updated {config.INDEX_CSV} with 'cached_path' column.")

    actual_files = len(os.listdir(config.CACHE_DIR))
    print(f"Cache folder contains {actual_files} files "
          f"(index has {df['cached_path'].notna().sum()} valid cached_path entries).")
    if actual_files < df['cached_path'].notna().sum():
        print("  WARNING: fewer files on disk than entries in the index -- "
              "some cached images may have been deleted outside this script.")


if __name__ == "__main__":
    build_cache()