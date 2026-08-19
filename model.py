"""
model.py
Loads pretrained ResNet50 (main pipeline) or other architectures (for
compare_architectures.py), replaces the final classifier layer for binary
classification, and provides freeze/unfreeze helpers for the 4-tier
progressive-unfreezing workflow.

Main pipeline usage (train.py):
    from model import get_model, get_device, unfreeze_tier
    device = get_device()
    model = get_model(config.ARCHITECTURE, device)   # tier 1: frozen backbone
    model = unfreeze_tier(model, config.ARCHITECTURE, tier=2)
    model = unfreeze_tier(model, config.ARCHITECTURE, tier=3)
    model = unfreeze_tier(model, config.ARCHITECTURE, tier=4)

compare_architectures.py can still call unfreeze_last_block / unfreeze_deeper
/ unfreeze_all directly, or go through unfreeze_tier -- both are equivalent.
"""

import torch
import torch.nn as nn
from torchvision import models

import config


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    return device


# ---------------------------------------------------------------------------
# Model loading -- frozen backbone, custom binary head
# ---------------------------------------------------------------------------
def get_model(architecture, device, num_classes=2, pretrained=True):
    """
    architecture: "resnet50", "resnet34", or "densenet121"
    Returns a model with all pretrained layers frozen except the new final
    classification layer (Tier 1 / feature-extraction mode).
    """
    architecture = architecture.lower()
    dropout_p = getattr(config, "DROPOUT_P", 0.0)

    if architecture == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(num_features, num_classes))

    elif architecture == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(num_features, num_classes))

    elif architecture == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(num_features, num_classes))

    else:
        raise ValueError(f"Unknown architecture '{architecture}'. "
                          f"Use 'resnet50', 'resnet34', or 'densenet121'.")

    model = model.to(device)
    _report_trainable(model, architecture, "frozen backbone (Tier 1)")
    return model


def _report_trainable(model, architecture, mode_label):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{architecture}] Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%) -- {mode_label}")


# ---------------------------------------------------------------------------
# BatchNorm stats freezing -- see config.FREEZE_BN_STATS docstring. Call
# this AFTER model.train() every epoch (train.py does this in
# train_one_epoch) so BN layers stay in eval mode -- using their existing
# running mean/var -- while their affine weight/bias (and everything else
# unfrozen) still trains normally via backprop. This does NOT touch
# requires_grad, only the module's train/eval behavior for running stats.
# ---------------------------------------------------------------------------
def freeze_bn_stats(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
    return model


# ---------------------------------------------------------------------------
# Tier 2: unfreeze the last block only
# ---------------------------------------------------------------------------
def unfreeze_last_block(model, architecture):
    architecture = architecture.lower()

    if architecture in ("resnet50", "resnet34"):
        for param in model.layer4.parameters():
            param.requires_grad = True

    elif architecture == "densenet121":
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown architecture '{architecture}'.")

    _report_trainable(model, architecture, "Tier 2: last block unfrozen")
    return model


# ---------------------------------------------------------------------------
# Tier 3: unfreeze the last two blocks (deeper, capacity-matched)
# ---------------------------------------------------------------------------
def unfreeze_deeper(model, architecture):
    architecture = architecture.lower()

    if architecture in ("resnet50", "resnet34"):
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True

    elif architecture == "densenet121":
        for param in model.features.denseblock3.parameters():
            param.requires_grad = True
        for param in model.features.transition3.parameters():
            param.requires_grad = True
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown architecture '{architecture}'.")

    _report_trainable(model, architecture, "Tier 3: last two blocks unfrozen")
    return model


# ---------------------------------------------------------------------------
# Tier 4: unfreeze everything, including earliest ImageNet features
# ---------------------------------------------------------------------------
def unfreeze_all(model, architecture):
    for param in model.parameters():
        param.requires_grad = True

    _report_trainable(model, architecture, "Tier 4: fully unfrozen")
    return model


# ---------------------------------------------------------------------------
# Single dispatcher used by train.py / patient_holdout_check.py so the tier
# progression lives in one place instead of being hand-rolled per script.
# ---------------------------------------------------------------------------
def unfreeze_tier(model, architecture, tier):
    """
    tier=1 is a no-op here (get_model already returns the frozen Tier-1
    model) -- included so callers can loop `for tier in (1,2,3,4)` cleanly.
    """
    if tier == 1:
        return model
    elif tier == 2:
        return unfreeze_last_block(model, architecture)
    elif tier == 3:
        return unfreeze_deeper(model, architecture)
    elif tier == 4:
        return unfreeze_all(model, architecture)
    else:
        raise ValueError(f"tier must be 1, 2, 3, or 4, got {tier}")


# ---------------------------------------------------------------------------
# Feature-fusion model -- ResNet50 CNN branch + engineered features
# (nuclei density, H/E stain stats, GLCM texture; see engineered_features.py)
# concatenated before the final classifier layer.
#
# Kept as a fully separate class/pathway (get_fusion_model /
# unfreeze_tier_fusion) rather than modifying get_model/unfreeze_tier, so
# the existing image-only pipeline is completely untouched -- run.py /
# best_of_n.py / list_misclassified.py etc. keep working exactly as before
# if they never call these.
# ---------------------------------------------------------------------------
class FusionResNet50(nn.Module):
    """
    engineered_features are fed through a BatchNorm1d before concatenation
    instead of a manually-fit StandardScaler. This is deliberate: BatchNorm1d
    only updates its running mean/var during model.train() forward passes,
    so as long as you only call model.train() on TRAIN batches (as
    train_stage already does), the normalization statistics are fit on
    train data only -- the same leakage guarantee a StandardScaler.fit on
    train-only would give you, without needing a separate fit/save/load
    step.
    """
    def __init__(self, num_engineered_features, num_classes=2, pretrained=True,
                 dropout_p=0.3, fusion_hidden=32):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        num_cnn_features = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()  # expose the pooled 2048-d feature, not class logits
        for param in backbone.parameters():
            param.requires_grad = False
        self.backbone = backbone

        self.feature_norm = nn.BatchNorm1d(num_engineered_features)
        self.head = nn.Sequential(
            nn.Linear(num_cnn_features + num_engineered_features, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, images, engineered_features):
        cnn_feat = self.backbone(images)                   # (B, 2048)
        eng_feat = self.feature_norm(engineered_features)   # (B, F)
        combined = torch.cat([cnn_feat, eng_feat], dim=1)
        return self.head(combined)


def get_fusion_model(num_engineered_features, device, num_classes=2, pretrained=True):
    # Deliberately NOT config.DROPOUT_P -- that was tested and set to 0.0 for
    # the CNN backbone specifically (see config.py's comment: dropout there
    # underfit every tier). The fusion head is a separate, freshly-initialized
    # ~2060->H->2 MLP that's fully trainable from Tier 1 onward regardless of
    # backbone tier, so it needs its own regularization knob -- reusing
    # DROPOUT_P=0.0 silently gave the head zero dropout, which is the likely
    # cause of it memorizing train fast (528k params at fusion_hidden=256,
    # zero dropout, on 597 images).
    dropout_p = getattr(config, "FUSION_HEAD_DROPOUT", 0.3)
    fusion_hidden = getattr(config, "FUSION_HIDDEN", 32)
    model = FusionResNet50(num_engineered_features, num_classes=num_classes,
                            pretrained=pretrained, dropout_p=dropout_p, fusion_hidden=fusion_hidden)
    model = model.to(device)
    _report_trainable(model, "resnet50-fusion", "frozen backbone (Tier 1) + trainable fusion head")
    return model


def unfreeze_tier_fusion(model, tier):
    """
    Same tier semantics as unfreeze_tier, applied to model.backbone. The
    fusion head (feature_norm + head) has no pretrained weights to protect
    and is trainable at every tier, including Tier 1.
    """
    if tier == 1:
        pass
    elif tier == 2:
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
    elif tier == 3:
        for param in model.backbone.layer3.parameters():
            param.requires_grad = True
        for param in model.backbone.layer4.parameters():
            param.requires_grad = True
    elif tier == 4:
        for param in model.backbone.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"tier must be 1, 2, 3, or 4, got {tier}")

    _report_trainable(model, "resnet50-fusion", f"Tier {tier}")
    return model


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved model weights to {path}")


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    print(f"Loaded model weights from {path}")
    return model


# ---------------------------------------------------------------------------
# Sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = get_device()

    print("\n--- ResNet50, all 4 tiers ---")
    model = get_model("resnet50", device)
    for tier in (2, 3, 4):
        model = unfreeze_tier(model, "resnet50", tier)

    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    print("\nDummy forward pass output shape:", output.shape)  # should be [2, 2]