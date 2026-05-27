import os
import numpy as np
from PIL import Image

print("\n=== HQNNDL Local Verification ===\n")

# ── 1. Check weights file ──────────────────────────────────────────
weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_weights.npz")

if not os.path.exists(weights_path):
    print("✗ trained_weights.npz NOT found in backend/")
    print("  → Download it from Kaggle Output tab and place it here")
else:
    ckpt = np.load(weights_path)
    print("✓ trained_weights.npz found")
    print(f"  W1 : {ckpt['W1'].shape}   (expected: (128, 512))")
    print(f"  W2 : {ckpt['W2'].shape}   (expected: (32, 128))")
    print(f"  W3 : {ckpt['W3'].shape}   (expected: (4, 32))")

# ── 2. Check PyTorch ───────────────────────────────────────────────
print()
try:
    import torch
    import torchvision
    print(f"✓ PyTorch     : {torch.__version__}")
    print(f"✓ Torchvision : {torchvision.__version__}")
except ImportError:
    print("✗ PyTorch not installed")
    print("  → Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

# ── 3. Check feature backend ───────────────────────────────────────
print()
try:
    from cnn_features import USE_TORCH, get_feature_dim
    backend = "ResNet-18 (512-dim)" if USE_TORCH else "Statistical (64-dim fallback)"
    print(f"✓ CNN backend : {backend}")
    print(f"  Feature dim : {get_feature_dim()}")
except Exception as e:
    print(f"✗ cnn_features import failed: {e}")

# ── 4. Check classifier loads weights ─────────────────────────────
print()
try:
    from classifier import _WEIGHTS_LOADED
    mark   = "✓" if _WEIGHTS_LOADED else "✗"
    status = "trained weights" if _WEIGHTS_LOADED else "fallback heuristic (unreliable)"
    print(f"{mark} Classifier   : {status}")
except Exception as e:
    print(f"✗ classifier import failed: {e}")

# ── 5. Run a test prediction ───────────────────────────────────────
print()
try:
    from preprocessing import preprocess_oct
    from cnn_features import extract_cnn_features
    from classifier import classify

    img = Image.fromarray(
        np.random.randint(50, 200, (256, 512), dtype=np.uint8)
    )
    arr, _ = preprocess_oct(img)
    feats  = extract_cnn_features(arr)
    result = classify(feats, arr)

    print(f"✓ Test prediction : {result['label']}  ({result['confidence']}%)")
    print(f"  Weights source  : {result['weights_source']}")
    print(f"  All probs       :")
    for p in result["probabilities"]:
        print(f"    {p['label']:<20} {p['probability']:.1f}%")

except Exception as e:
    import traceback
    print(f"✗ Test prediction failed: {e}")
    traceback.print_exc()

print("\n=== Done ===\n")

# Add this to the bottom of verify.py temporarily to check architecture
import numpy as np, os
ckpt = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_weights.npz"))
print("\n=== Architecture check ===")
print(f"  W1 : {ckpt['W1'].shape}  → expecting (256, 512)")
print(f"  W2 : {ckpt['W2'].shape}  → expecting (64, 256)")
print(f"  W3 : {ckpt['W3'].shape}  → expecting (4, 64)")
arch_ok = (ckpt['W1'].shape == (256,512) and
           ckpt['W2'].shape == (64,256)  and
           ckpt['W3'].shape == (4,64))
print(f"\n  {'✓ Architecture correct' if arch_ok else '✗ Architecture mismatch'}")