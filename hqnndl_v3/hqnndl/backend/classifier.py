"""
classifier.py
─────────────
Stage 5: Hybrid Fusion + Final Classification
Loads trained weights from trained_weights.npz
Architecture: ResNet18(512) → 128 → 32 → 4
"""

import os
import logging
import numpy as np
from quantum_model import run_vqc

logger = logging.getLogger(__name__)

LABELS = ["Normal", "CSR Grade I", "CSR Grade II", "Other Pathology"]
COLORS = {
    "Normal":          "#22c55e",
    "CSR Grade I":     "#f59e0b",
    "CSR Grade II":    "#ef4444",
    "Other Pathology": "#8b5cf6",
}

# ── Load trained weights ───────────────────────────────────────────
_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "trained_weights.npz"
)

_W1 = _b1 = _W2 = _b2 = _W3 = _b3 = None
_X_mean = _X_std = None
_WEIGHTS_LOADED = False
_FEATURE_DIM    = None

def _load_weights():
    global _W1, _b1, _W2, _b2, _W3, _b3
    global _X_mean, _X_std, _WEIGHTS_LOADED, _FEATURE_DIM

    if not os.path.exists(_WEIGHTS_PATH):
        logger.warning(
            "trained_weights.npz not found at %s — "
            "predictions will be unreliable", _WEIGHTS_PATH
        )
        return

    try:
        ckpt = np.load(_WEIGHTS_PATH)

        # Detect architecture from weight shapes
        _W1 = ckpt["W1"].astype(np.float32)
        _b1 = ckpt["b1"].astype(np.float32)
        _W2 = ckpt["W2"].astype(np.float32)
        _b2 = ckpt["b2"].astype(np.float32)
        _W3 = ckpt["W3"].astype(np.float32)
        _b3 = ckpt["b3"].astype(np.float32)

        _X_mean = ckpt["X_mean"].astype(np.float32)
        _X_std  = ckpt["X_std"].astype(np.float32)

        _FEATURE_DIM    = _W1.shape[1]
        _WEIGHTS_LOADED = True

        logger.info(
            "Loaded trained weights: arch=%s → %s → %s → %s  "
            "feature_dim=%d",
            _W1.shape[1], _W1.shape[0],
            _W2.shape[0], _W3.shape[0],
            _FEATURE_DIM
        )

    except Exception as e:
        logger.error("Failed to load weights: %s", e)

_load_weights()


# ── Network functions ─────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def _forward(x: np.ndarray) -> np.ndarray:
    """Single sample forward pass → 4-class probabilities.
    Architecture: 512 → 256 → 64 → 4
    """
    h1     = _relu(x  @ _W1.T + _b1)   # 512 → 256
    h2     = _relu(h1 @ _W2.T + _b2)   # 256 → 64
    logits = h2 @ _W3.T + _b3          #  64 → 4
    return _softmax(logits)


# ── Main classify function ────────────────────────────────────────

def classify(features: np.ndarray, arr: np.ndarray) -> dict:
    """
    Classify an OCT scan.

    Parameters
    ----------
    features : np.ndarray  — CNN feature vector (512-dim or 64-dim)
    arr      : np.ndarray  — preprocessed image array (H, W)

    Returns
    -------
    dict with label, confidence, probabilities, measurements,
         fluid_signal, grad_energy
    """

    # ── Quantum measurements (kept for hybrid output reporting) ────
    top8  = features[np.argsort(np.abs(features))[-8:]]
    q_out = run_vqc(top8)

    # ── Image signals (for reporting only, not classification) ─────
    H, W = arr.shape
    lo   = arr[H // 2:, W // 4:3 * W // 4]
    mi   = arr[H // 3:2 * H // 3, W // 4:3 * W // 4]
    fluid_signal = (
        float(np.sum(lo < 0.15) / lo.size) * 3.0 +
        float(np.sum(mi < 0.10) / mi.size) * 2.0
    )
    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)
    grad_energy = float(np.mean(
        np.sqrt(np.pad(gx, ((0,0),(0,1)))**2 +
                np.pad(gy, ((0,1),(0,0)))**2)
    ))

    # ── Classification ─────────────────────────────────────────────
    if _WEIGHTS_LOADED:
        # Normalise using training statistics
        feat_norm = (features - _X_mean) / (_X_std + 1e-8)

        # Truncate or pad to match trained feature dim
        if len(feat_norm) > _FEATURE_DIM:
            feat_norm = feat_norm[:_FEATURE_DIM]
        elif len(feat_norm) < _FEATURE_DIM:
            feat_norm = np.pad(
                feat_norm, (0, _FEATURE_DIM - len(feat_norm))
            )

        probs = _forward(feat_norm)
        logger.debug("Trained weights used — probs=%s", probs)

    else:
        # Fallback: heuristic (random weights, unreliable)
        logger.warning("Using fallback heuristic — no trained weights")
        fs_norm = min(1.0, fluid_signal / 0.25)
        ge_norm = min(1.0, grad_energy  / 0.06)

        logits = np.array([
            (1.0 - fs_norm) * 1.8 + (0.5 - abs(ge_norm - 0.5)) * 0.8,
             fs_norm * 0.8 * (1.0 - fs_norm) * 4.0,
             fs_norm * 1.5 - (1.0 - fs_norm) * 1.2,
             ge_norm * 1.5 - fs_norm * 0.5,
        ], dtype=np.float32)
        probs = _softmax(logits)

    pred = int(np.argmax(probs))

    logger.info(
        "Classification → %s  (conf=%.1f%%  fluid=%.4f  grad=%.4f  "
        "weights=%s)",
        LABELS[pred], probs[pred] * 100,
        fluid_signal, grad_energy,
        "trained" if _WEIGHTS_LOADED else "fallback"
    )

    return {
        "label":      LABELS[pred],
        "confidence": round(float(probs[pred] * 100), 1),
        "probabilities": [
            {
                "label":       LABELS[i],
                "probability": float(round(probs[i] * 100, 2)),
                "color":       COLORS[LABELS[i]],
            }
            for i in range(4)
        ],
        "measurements":  q_out.tolist(),
        "fluid_signal":  round(fluid_signal, 4),
        "grad_energy":   round(grad_energy,  4),
        "weights_source": "trained" if _WEIGHTS_LOADED else "fallback",
    }