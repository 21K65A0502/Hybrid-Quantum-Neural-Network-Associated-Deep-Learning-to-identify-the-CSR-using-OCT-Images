"""
biomarkers.py
─────────────
Stage 6: Retinal Biomarker Analysis

Computes 6 quantitative biomarker scores from the preprocessed OCT array.
Each score is an integer in [1, 99] derived from pixel-level statistics
that correlate with clinically observed OCT features.

Biomarkers
──────────
  1. Subretinal Fluid    — hyporeflective dome in mid-retinal zone
  2. RPE Detachment      — irregularity in the RPE / Bruch's membrane layer
  3. Foveal Thickness    — central column bright-pixel count proxy
  4. Choroidal Volume    — choroidal layer mean intensity
  5. IS/OS Integrity     — inner/outer segment junction integrity
  6. Ellipsoid Zone      — ellipsoid zone reflectivity variance
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

SEVERITY_THRESHOLDS = (25, 50, 75)


def _clamp(v: float) -> int:
    return min(99, max(1, int(v)))


def _severity(score: int, thresholds=SEVERITY_THRESHOLDS) -> str:
    a, b, c = thresholds
    if score < a: return "Absent"
    if score < b: return "Mild"
    if score < c: return "Moderate"
    return "Severe"


def analyze_biomarkers(arr: np.ndarray, diagnosis: str) -> list:
    """
    Compute retinal biomarker scores from a preprocessed OCT array.

    Parameters
    ----------
    arr       : np.ndarray  float32 [0,1]  shape (H, W)
    diagnosis : str  — classification label (used for context logging)

    Returns
    -------
    List of 6 biomarker dicts, each with:
        name, score, severity, color, normal
    """
    H, W = arr.shape
    logger.debug("Computing biomarkers for diagnosis=%s", diagnosis)

    # 1. Subretinal Fluid
    r   = arr[H // 3:2 * H // 3, W // 3:2 * W // 3]
    fs  = _clamp(float(np.sum(r < 0.12) / r.size) * 100 * 6)

    # 2. RPE Detachment
    rp  = arr[2 * H // 3:, :]
    rs  = _clamp(float(np.std(np.mean(rp, axis=1))) * 100 * 8)

    # 3. Foveal Thickness
    ct  = arr[:, W // 2 - 30:W // 2 + 30]
    ts  = _clamp(float(np.sum(np.mean(ct, axis=1) > 0.3)) / H * 100 * 1.5)

    # 4. Choroidal Volume
    ch  = arr[3 * H // 4:, :]
    cs  = _clamp(float(np.mean(ch)) * 100 * 1.8)

    # 5. IS/OS Integrity
    io_ = arr[H // 4:H // 2, :]
    is_ = _clamp(float(np.mean(io_)) * 100 * 1.5)

    # 6. Ellipsoid Zone
    ez  = arr[H // 3:H // 2, W // 4:3 * W // 4]
    es  = _clamp(float(np.std(ez)) * 200)

    biomarkers = [
        {"name": "Subretinal Fluid", "score": fs, "severity": _severity(fs),
         "color": "#06b6d4", "normal": "0-15"},
        {"name": "RPE Detachment",   "score": rs, "severity": _severity(rs),
         "color": "#ef4444", "normal": "0-20"},
        {"name": "Foveal Thickness", "score": ts, "severity": _severity(ts),
         "color": "#8b5cf6", "normal": "0-35"},
        {"name": "Choroidal Volume", "score": cs, "severity": _severity(cs),
         "color": "#f59e0b", "normal": "0-40"},
        {"name": "IS/OS Integrity",  "score": is_,"severity": _severity(is_),
         "color": "#22c55e", "normal": "50+"},
        {"name": "Ellipsoid Zone",   "score": es, "severity": _severity(es),
         "color": "#38bdf8", "normal": "0-30"},
    ]

    logger.debug(
        "Biomarkers: %s",
        {b["name"]: f"{b['score']}/99 ({b['severity']})" for b in biomarkers}
    )
    return biomarkers
