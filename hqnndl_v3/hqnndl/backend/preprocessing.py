"""
preprocessing.py
────────────────
Stage 1: OCT Image Preprocessing
  • Grayscale conversion
  • Resize to 512×256 (standard OCT resolution)
  • Gaussian denoising  (σ = 0.8)
  • Median filter       (3×3)
  • Contrast enhancement (CLAHE-like, factor 2.2)
  • Unsharp sharpening
"""

import logging
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

logger = logging.getLogger(__name__)

TARGET_W = 512
TARGET_H = 256


def preprocess_oct(img: Image.Image):
    """
    Preprocess a raw OCT PIL Image.

    Returns
    -------
    arr  : np.ndarray  float32 [0,1]  shape (H, W)
    pil  : PIL.Image   uint8          preprocessed grayscale image
    """
    logger.debug("Preprocessing OCT image (size=%s mode=%s)", img.size, img.mode)

    # 1. Grayscale
    gray = img.convert("L")

    # 2. Resize
    resized = gray.resize((TARGET_W, TARGET_H), Image.LANCZOS)

    # 3. Gaussian denoising
    denoised = resized.filter(ImageFilter.GaussianBlur(radius=0.8))

    # 4. Median filter (speckle removal)
    denoised = denoised.filter(ImageFilter.MedianFilter(size=3))

    # 5. Contrast enhancement (CLAHE-like)
    enhanced = ImageEnhance.Contrast(denoised).enhance(2.2)

    # 6. Sharpening
    sharp = enhanced.filter(ImageFilter.SHARPEN)

    arr = np.array(sharp, dtype=np.float32) / 255.0
    logger.debug("Preprocessing done — array shape %s", arr.shape)
    return arr, sharp
