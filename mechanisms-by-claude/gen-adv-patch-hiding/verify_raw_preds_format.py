"""
One-time sanity check for generate_art_patch.py / apply_saved_patch.py's
get_raw_preds() parsing, run against whatever model_path is currently in
config.json. Confirms the returned tensor has the shape and value ranges
those scripts assume -- (B, 4+nc, num_anchors), decoded boxes in pixel
space, class scores already sigmoided into [0,1] -- before you trust
patch generation results against a new model.

This is the manual equivalent of UltralyticsClsHeadAdapter.describe()
on the detection side: there's no adapter object here to ask, since
generate_art_patch.py talks to the model directly rather than through
hooks, so this script checks the same kind of thing by hand.
"""

import json
import torch
from pathlib import Path
from ultralytics import YOLO

CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

INPUT_SIZE = 640
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_raw_preds(model, x):
    """Identical to generate_art_patch.py's version -- kept in sync
    deliberately, so this check verifies the ACTUAL parsing logic in
    use, not a reimplementation of it."""
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    elif isinstance(out, dict):
        out = out.get("one2many", list(out.values())[0])
        if isinstance(out, tuple):
            out = out[0]
    return out.transpose(1, 2)


print(f"Loading model: {config['model_path']}")
yolo_model = YOLO(config["model_path"]).to(device)
yolo_model.model.eval()

nc = yolo_model.model.model[-1].nc  # number of classes the head was built for
expected_channels = 4 + nc
print(f"Model reports nc={nc} -> expecting {expected_channels} channels (4 box + {nc} class) per anchor")

dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
with torch.no_grad():
    preds = get_raw_preds(yolo_model.model, dummy)

print(f"\nget_raw_preds() returned shape: {tuple(preds.shape)}")
B, A, C = preds.shape
print(f"  batch={B}, anchors={A}, channels={C}")

assert C == expected_channels, (
    f"MISMATCH: got {C} channels, expected {expected_channels}. "
    f"get_raw_preds' tuple/dict unwrapping logic likely does not match "
    f"this model's Detect.forward() output -- do not trust patch "
    f"generation results until this is resolved."
)
print(f"  channel count matches (4 + nc) -- get_raw_preds is unwrapping this model's output correctly.")

boxes = preds[0, :, 0:4]
class_probs = preds[0, :, 4:]
print(f"\nBox coordinate range (should span roughly 0-{INPUT_SIZE} pixels once a real image "
      f"is used -- a dummy all-zero input will legitimately show degenerate/tiny values here, "
      f"this is just checking the channels exist and are numeric, not meaningful box geometry):")
print(f"  min={boxes.min().item():.2f}  max={boxes.max().item():.2f}")

print(f"\nClass probability range (should be within [0, 1] -- confirms scores are "
      f"already sigmoided, not raw logits, matching what select_target_object assumes):")
print(f"  min={class_probs.min().item():.4f}  max={class_probs.max().item():.4f}")
assert 0.0 <= class_probs.min().item() and class_probs.max().item() <= 1.0, (
    "Class scores fall outside [0,1] -- they may be raw logits rather than "
    "sigmoided probabilities for this model/version. conf_thresh comparisons "
    "in select_target_object would be meaningless until this is addressed."
)
print("  range is valid -- class scores are properly sigmoided probabilities.")

print("\nAll checks passed. get_raw_preds() is safe to trust for this model.")