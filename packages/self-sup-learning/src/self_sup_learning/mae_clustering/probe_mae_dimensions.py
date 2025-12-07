"""
Script to probe MAE ViT model output dimensions
"""

import sys

import torch

sys.path.insert(0, "/home/jin/Desktop/mm/mmpretrain")

# Monkey-patch torch.load to use weights_only=False
# This is safe since we're loading a trusted checkpoint from OpenMMLab
_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

from mmpretrain import get_model

# Load the model with pretrained weights
config = "/home/jin/Desktop/mm/mmpretrain/configs/mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py"
checkpoint = "/home/jin/Desktop/mm/pretrained/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth"

print("Loading model...")
model = get_model(config, pretrained=checkpoint, device="cpu")
model.eval()

print(f"Model type: {type(model)}")
print(f"Model structure:\n{model}")

# Test with a dummy input
dummy_input = torch.randn(2, 3, 224, 224)
print(f"\nInput shape: {dummy_input.shape}")

# Try to extract features from backbone
print("\n--- Testing backbone feature extraction ---")
try:
    with torch.no_grad():
        # Process through data preprocessor first
        processed = model.data_preprocessor({"inputs": dummy_input}, training=False)
        inputs = processed["inputs"]
        print(f"Preprocessed input shape: {inputs.shape}")

        # Extract features from backbone
        feats = model.backbone(inputs)
        if isinstance(feats, (list, tuple)):
            print(f"Backbone outputs {len(feats)} feature maps:")
            for i, feat in enumerate(feats):
                print(f"  Feature {i}: shape={feat.shape}, dtype={feat.dtype}")
                # Flatten to see final dimension
                flat = feat.view(feat.shape[0], -1)
                print(f"    Flattened: {flat.shape}")
        else:
            print(f"Backbone output shape: {feats.shape}")
            flat = feats.view(feats.shape[0], -1)
            print(f"  Flattened: {flat.shape}")
except Exception as e:
    print(f"Error with backbone: {e}")
    import traceback

    traceback.print_exc()

# Try extract_feat method
print("\n--- Testing model.extract_feat() ---")
try:
    with torch.no_grad():
        feats = model.extract_feat(dummy_input)
        if isinstance(feats, (list, tuple)):
            print(f"extract_feat outputs {len(feats)} tensors:")
            for i, feat in enumerate(feats):
                print(f"  Feature {i}: shape={feat.shape}, dtype={feat.dtype}")
                flat = feat.view(feat.shape[0], -1)
                print(f"    Flattened: {flat.shape}")
        else:
            print(f"extract_feat output shape: {feats.shape}")
            flat = feats.view(feats.shape[0], -1)
            print(f"  Flattened: {flat.shape}")
except Exception as e:
    print(f"Error with extract_feat: {e}")
    import traceback

    traceback.print_exc()

print("\n--- Summary ---")
print("Use the flattened dimension shown above for LanceDB embedding storage.")
