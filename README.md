# Cardiac H&E Inflammation Classification

Binary classification of cardiac H&E histopathology tiles as **Inflammatory** vs **Non-inflammatory**, using transfer learning on CNNs with progressive layer unfreezing, stain normalization, engineered-feature fusion, and model ensembling.

Developed during a research internship at the Computational Biology and Translational Bioinformatics Laboratory, IIT Roorkee.

## Overview

The pipeline classifies histopathology tiles as the first stage of a larger workflow that also performs nuclei instance segmentation on tiles flagged as Inflammatory. Key components:

- **Stain normalization** — Macenko normalization removes patient/scanner color variation that would otherwise let a model shortcut on staining identity rather than tissue biology.
- **CNN classification** — ResNet50 (primary) and DenseNet121 (comparison), both ImageNet-pretrained, fine-tuned with a 4-tier progressive-unfreezing schedule.
- **Feature fusion** — a variant that concatenates CNN features with hand-engineered features (nuclei density, stain-channel statistics, GLCM texture) before classification.
- **Ensembling** — independently trained models combined via probability averaging with a tuned decision threshold.

## Results

| Model | Accuracy | F1 (Non-inflam.) | F1 (Inflam.) |
|---|---|---|---|
| ResNet50, Tier 4 (final) | 91% | 0.93 | 0.90 |
| DenseNet121, Tier 4 | 86.6% | 0.88 | 0.85 |
| ResNet50 + Feature Fusion, Tier 4 | 88% | 0.89 | 0.86 |
| Ensemble (Baseline + Fusion) | 90.5% | 0.92 | 0.89 |

## Project Structure

```
config.py                     # paths, hyperparameters, split/holdout settings
dataset.py                    # dataset indexing, PyTorch Dataset, train/val split logic
preprocess_cache.py           # resizes and caches source images
stain_normalize_cache.py      # Macenko stain normalization
engineered_features.py        # nuclei/stain/texture feature extraction
model.py                      # ResNet50/DenseNet121 loading, tiered unfreezing, fusion model
train.py                      # main training pipeline (image-only)
train_fusion.py               # training pipeline for the feature-fusion model
predict.py                    # inference on new raw images (loads both trained models)
```

## Trained Weights

Trained model checkpoints (`.pth`) are not included in this repository due to file size. Download them from the [Releases](../../releases) page and place them in a `models/` folder before running `predict.py`.

## Setup

```bash
pip install torch torchvision opencv-python scikit-image scipy pandas pillow tqdm scikit-learn
```

## Usage

Run in order:

```bash
python preprocess_cache.py        # resize + cache source images
python stain_normalize_cache.py   # Macenko normalization
python engineered_features.py     # engineered feature extraction (for fusion model)
python train.py                   # train the baseline CNN
python train_fusion.py            # train the feature-fusion model
python predict.py path/to/image_or_folder   # run inference on new images
```

Configure architecture, split strategy, and hyperparameters in `config.py` before training.

## Methodology Notes

- **Split strategy**: image-wise (tile-level) train/validation split, stratified by class.
- **Training**: class-weighted cross-entropy with label smoothing, per-tier learning rates with plateau-based LR scheduling and early stopping, best-validation-accuracy checkpoint restoration, 5-view test-time augmentation (TTA) at inference.
- **Progressive unfreezing**: Tier 1 (classifier head only) → Tier 2 (last block) → Tier 3 (last two blocks) → Tier 4 (full network).

## Limitations

- Dataset is limited in size (952 tiles, 33 patients), which bounds achievable model performance regardless of training recipe.
- Several patients are label-pure and vary widely in image count, which affects both training and evaluation stability.
- Run-to-run variance under identical configuration was observed, attributable to random initialization and batch ordering in early training tiers.

## Acknowledgements

Developed under the supervision of Dr. Deepak Sharma, Computational Biology and Translational Bioinformatics Laboratory, Indian Institute of Technology Roorkee.
