import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n=== FILE CHECK ===")

# Check which app.py is actually being used
app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
with open(app_path, "r") as f:
    content = f.read()

has_validator_import = "from retinal_validator import" in content
has_validator_call   = "validate_retinal_image" in content
has_original_img     = "original_img=img" in content

print(f"  app.py has validator import  : {'✓' if has_validator_import else '✗ MISSING'}")
print(f"  app.py calls validator       : {'✓' if has_validator_call   else '✗ MISSING'}")
print(f"  app.py passes original_img   : {'✓' if has_original_img     else '✗ MISSING'}")

# Check retinal_validator.py exists and has strict checks
val_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retinal_validator.py")
if os.path.exists(val_path):
    with open(val_path, "r") as f:
        val_content = f.read()
    has_color_check  = "_color_variance_score" in val_content
    has_aspect_check = "_aspect_ratio_score"   in val_content
    has_hard_reject  = "hard_reject"           in val_content
    print(f"\n  retinal_validator.py found   : ✓")
    print(f"  has colour check             : {'✓' if has_color_check  else '✗ OLD VERSION'}")
    print(f"  has aspect ratio check       : {'✓' if has_aspect_check else '✗ OLD VERSION'}")
    print(f"  has hard reject rules        : {'✓' if has_hard_reject  else '✗ OLD VERSION'}")
else:
    print(f"\n  retinal_validator.py         : ✗ FILE NOT FOUND")

# Check cnn_features.py has original_img parameter
cnn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_features.py")
with open(cnn_path, "r") as f:
    cnn_content = f.read()
has_original_param = "original_img" in cnn_content
print(f"\n  cnn_features.py original_img : {'✓' if has_original_param else '✗ OLD VERSION'}")

# Check trained weights architecture
weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_weights.npz")
if os.path.exists(weights_path):
    import numpy as np
    ckpt = np.load(weights_path)
    print(f"\n  trained_weights.npz          : ✓")
    print(f"  W1 shape : {ckpt['W1'].shape}  (need (256,512))")
    print(f"  W2 shape : {ckpt['W2'].shape}  (need (64,256))")
    print(f"  W3 shape : {ckpt['W3'].shape}  (need (4,64))")
    arch_ok = (ckpt['W1'].shape == (256,512) and
               ckpt['W2'].shape == (64,256)  and
               ckpt['W3'].shape == (4,64))
    print(f"  Architecture correct         : {'✓' if arch_ok else '✗ WRONG WEIGHTS'}")

print("\n=== LIVE VALIDATOR TEST ===")
try:
    from retinal_validator import validate_retinal_image, RETINAL_THRESHOLD
    from PIL import Image
    import numpy as np

    print(f"  Threshold in use : {RETINAL_THRESHOLD}")

    # Simulate a colour face photo (high R-G-B variance)
    face_arr = np.zeros((256,256,3), dtype=np.uint8)
    face_arr[:,:,0] = 180   # R high  (skin tone)
    face_arr[:,:,1] = 140   # G mid
    face_arr[:,:,2] = 120   # B low
    face_img = Image.fromarray(face_arr)
    face_result = validate_retinal_image(face_img)
    print(f"\n  Fake face photo test:")
    print(f"    is_retinal : {face_result['is_retinal']}  (should be False)")
    print(f"    confidence : {face_result['confidence']}")
    print(f"    reason     : {face_result['reason']}")
    print(f"    scores     : {face_result['scores']}")

    # Simulate a grayscale landscape OCT
    oct_arr = np.zeros((128,256,3), dtype=np.uint8)
    oct_arr[:20,  :] = 40    # dark vitreous
    oct_arr[20:30,:] = 200   # bright RNFL
    oct_arr[30:70,:] = 20    # dark fluid zone
    oct_arr[70:90,:] = 160   # bright RPE
    oct_arr[90:, :] = 10    # dark choroid
    oct_img = Image.fromarray(oct_arr)
    oct_result = validate_retinal_image(oct_img)
    print(f"\n  Fake OCT image test:")
    print(f"    is_retinal : {oct_result['is_retinal']}  (should be True)")
    print(f"    confidence : {oct_result['confidence']}")
    print(f"    scores     : {oct_result['scores']}")

except Exception as e:
    import traceback
    print(f"  ✗ Validator test failed: {e}")
    traceback.print_exc()

print("\n=== DONE ===\n")