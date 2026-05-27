"""
cnn_features.py
───────────────
Stage 2: CNN Feature Extraction
ResNet-18 operates on the ORIGINAL image (matching Kaggle training).
Statistical features operate on the preprocessed array (fallback).
"""

import logging
import numpy as np
from PIL import Image as _PilImage

logger = logging.getLogger(__name__)

USE_TORCH          = False
_feature_extractor = None
_transform         = None
_DEVICE            = None

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _resnet = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )
    _feature_extractor = torch.nn.Sequential(
        *list(_resnet.children())[:-1]
    )
    _feature_extractor = _feature_extractor.to(_DEVICE)
    _feature_extractor.eval()

    _transform = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])

    USE_TORCH = True
    logger.info("ResNet-18 backbone loaded on %s", _DEVICE)

except Exception as e:
    USE_TORCH = False
    logger.warning("PyTorch not available (%s) — using statistical features", e)


def get_feature_dim() -> int:
    return 512 if USE_TORCH else 64


def extract_cnn_features(arr: np.ndarray,
                         original_img: _PilImage.Image = None) -> np.ndarray:
    """
    Extract feature vector from OCT image.

    Parameters
    ----------
    arr          : np.ndarray float32 [0,1] (H,W) — preprocessed array
    original_img : PIL.Image — original image BEFORE preprocessing.
                   MUST be passed when PyTorch is available so that
                   ResNet sees the same pixel distribution as during
                   Kaggle training.

    Returns
    -------
    np.ndarray float32 shape (512,) or (64,)
    """
    if USE_TORCH:
        # Use original image if available (matches training pipeline exactly)
        # Fall back to reconstructing from arr if not provided
        src = original_img if original_img is not None else \
              _PilImage.fromarray((arr * 255).astype(np.uint8))
        return _resnet_features(src)
    return _statistical_features(arr)


def _resnet_features(img: _PilImage.Image) -> np.ndarray:
    """Run ResNet-18 on a PIL image → 512-dim vector."""
    import torch
    tensor = _transform(img.convert("RGB")).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        feat = _feature_extractor(tensor)
    return feat.squeeze().cpu().numpy().astype(np.float32)


def _statistical_features(arr: np.ndarray) -> np.ndarray:
    H, W  = arr.shape
    feats = []

    zh = H // 6
    for z in range(6):
        zone = arr[z * zh:(z + 1) * zh, :]
        feats.extend([
            float(np.mean(zone)),
            float(np.std(zone)),
            float(np.percentile(zone, 25)),
            float(np.percentile(zone, 75)),
        ])

    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)
    gm = np.sqrt(
        np.pad(gx, ((0, 0), (0, 1))) ** 2 +
        np.pad(gy, ((0, 1), (0, 0))) ** 2
    )
    feats.extend([
        float(np.mean(gm)), float(np.max(gm)),
        float(np.std(gm)),  float(np.sum(gm > 0.1) / gm.size),
    ])

    pv = [
        float(np.var(arr[py:py + 3, px:px + 3]))
        for py in range(0, H - 3, 8)
        for px in range(0, W - 3, 16)
    ]
    feats.extend([
        float(np.mean(pv)), float(np.std(pv)),
        float(np.percentile(pv, 90)), float(np.max(pv)),
    ])

    row_mean = np.mean(arr, axis=1)
    peaks = sorted(
        [(i, float(row_mean[i])) for i in range(1, len(row_mean) - 1)
         if row_mean[i] > row_mean[i - 1] and row_mean[i] > row_mean[i + 1]],
        key=lambda x: -x[1]
    )
    for py, pv2 in (peaks[:4] + [(0, 0.0)] * 4)[:4]:
        feats.extend([py / H, pv2])

    lo = arr[H // 2:, W // 4:3 * W // 4]
    mi = arr[H // 3:2 * H // 3, W // 4:3 * W // 4]
    feats.extend([
        float(np.sum(lo < 0.15) / lo.size),
        float(np.sum(mi < 0.10) / mi.size),
    ])

    ct = arr[:, W // 2 - 20:W // 2 + 20]
    feats.extend([
        float(np.mean(ct)), float(np.std(ct)),
        float(np.min(np.mean(ct, axis=1))),
    ])

    feats = (feats + [0.0] * 64)[:64]
    return np.array(feats, dtype=np.float32)