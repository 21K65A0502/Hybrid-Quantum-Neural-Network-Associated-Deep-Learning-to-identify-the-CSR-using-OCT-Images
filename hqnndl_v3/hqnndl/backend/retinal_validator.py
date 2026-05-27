"""
retinal_validator.py
────────────────────
Stricter retinal OCT gatekeeper.
Rejects faces, X-rays, natural photos, and other non-OCT inputs.
"""

import numpy as np
from PIL import Image

RETINAL_THRESHOLD = 0.42


def _arr(img, w=256, h=128):
    return np.array(
        img.convert("L").resize((w, h)),
        dtype=np.float32
    ) / 255.0


def _color_variance_score(img) -> float:
    """
    OCT images are truly grayscale — R=G=B channels.
    Colour photos have high inter-channel variance.
    This is the strongest single signal.
    """
    rgb    = np.array(img.convert("RGB"), dtype=np.float32)
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    rg_diff = float(np.mean(np.abs(r - g)))
    rb_diff = float(np.mean(np.abs(r - b)))
    gb_diff = float(np.mean(np.abs(g - b)))
    avg_diff = (rg_diff + rb_diff + gb_diff) / 3.0
    # OCT: avg_diff < 3  |  Colour photo: avg_diff > 15
    score = max(0.0, 1.0 - avg_diff / 10.0)
    return round(score, 3)


def _horizontal_layer_score(arr) -> float:
    """OCT scans have strong horizontal banding from retinal layers."""
    gy    = np.abs(np.diff(arr, axis=0))
    gx    = np.abs(np.diff(arr, axis=1))
    ratio = float(np.mean(gy)) / (float(np.mean(gx)) + 1e-8)
    return min(1.0, max(0.0, (ratio - 0.85) / 1.8))


def _dark_region_score(arr) -> float:
    """OCT images have large dark regions (vitreous, background)."""
    dark_ratio = float(np.mean(arr < 0.15))
    if dark_ratio < 0.08:
        return 0.0
    if dark_ratio > 0.85:
        return 0.1
    return min(1.0, dark_ratio / 0.45)


def _intensity_bimodal_score(arr) -> float:
    """OCT has bimodal distribution: dark fluid + bright tissue."""
    dark_ratio   = float(np.mean(arr < 0.20))
    bright_ratio = float(np.mean(arr > 0.60))
    mid_ratio    = float(np.mean((arr >= 0.20) & (arr <= 0.60)))
    score = 0.0
    if 0.08 < dark_ratio   < 0.80: score += 0.35
    if 0.03 < bright_ratio < 0.60: score += 0.35
    if 0.08 < mid_ratio    < 0.70: score += 0.30
    return round(score, 3)


def validate_retinal_image(img: Image.Image) -> dict:
    """
    Returns
    -------
    dict:
        is_retinal : bool
        confidence : float [0,1]
        scores     : dict of individual signal scores
        reason     : str
    """
    arr = _arr(img)

    s_color   = _color_variance_score(img)
    s_layer   = _horizontal_layer_score(arr)
    s_dark    = _dark_region_score(arr)
    s_bimodal = _intensity_bimodal_score(arr)

    # Weighted combination — no aspect ratio (OCT can be any shape)
    combined = (
        0.40 * s_color   +   # strongest — grayscale check
        0.25 * s_layer   +   # horizontal banding
        0.20 * s_dark    +   # dark region presence
        0.15 * s_bimodal     # bimodal intensity
    )

    # Hard reject — ONLY colour check (most reliable signal)
    hard_reject = None
    if s_color < 0.35:
        hard_reject = (
            "Image appears to be in colour — OCT scans are greyscale. "
            "Please upload a greyscale OCT retinal scan."
        )

    is_retinal = (hard_reject is None) and (combined >= RETINAL_THRESHOLD)

    if hard_reject:
        reason = hard_reject
    elif not is_retinal:
        weakest = min(
            [("colour check",      s_color),
             ("horizontal layers", s_layer),
             ("dark regions",      s_dark),
             ("intensity profile", s_bimodal)],
            key=lambda x: x[1]
        )
        reason = (
            f"Low score on {weakest[0]} ({weakest[1]:.2f}) "
            f"— image does not match OCT pattern"
        )
    else:
        reason = "OK"

    return {
        "is_retinal":  is_retinal,
        "confidence":  round(float(combined), 3),
        "threshold":   RETINAL_THRESHOLD,
        "scores": {
            "colour_grayscale":  s_color,
            "horizontal_layers": round(s_layer,   3),
            "dark_regions":      round(s_dark,    3),
            "intensity_bimodal": round(s_bimodal, 3),
        },
        "reason": reason,
    }