"""
Step 1 of 2 for Mech-7: build the per-class (+ background) Mahalanobis
profiles from the images in `profile_images_dir`.

Same procedure as Mech-1 Option B (cells inside the detector's boxes feed that
class's profile; a capped random sample of uncovered cells feeds the
background profile), with the tap scale/stride read from config. Thresholds are
NOT learned here -- they come from a different, disjoint image set in
learn_thresholds.py, so rare-class distances aren't measured on the same cells
the profile was fitted on.
"""

import argparse
import pickle
import random
from collections import defaultdict
from pathlib import Path

import torch
from ultralytics import YOLO

from mech7_core import load_config, validated_settings, list_images, load_image_tensor
from ultralytics_cls_head_adapter import UltralyticsClsHeadAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-images", type=int, default=None, help="override detector_settings.max_calibration_images")
    args = ap.parse_args()

    config = load_config(args.config)
    s = validated_settings(config)
    scale, stride = s.get("tap_scale", "P3"), int(s["stride"])
    min_samples = s["min_calibration_samples"]
    bg_id = s.get("background_class_id", -1)
    bg_per_image = s.get("background_samples_per_image", 250)
    max_images = args.max_images or s.get("max_calibration_images")

    profile_dir = Path(config["profiles_dir"])
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / config["calibration_profile_filename"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(config["model_path"]).to(device)
    adapter = UltralyticsClsHeadAdapter(model.model, scales=(scale,))
    print(adapter.describe())

    images = list_images(config["profile_images_dir"], max_images)
    print(f"Building profiles from {len(images)} images in {config['profile_images_dir']} (tap={scale}, stride={stride})")

    per_class = defaultdict(list)
    for n, img_path in enumerate(images, 1):
        _, tensor = load_image_tensor(img_path, device)
        results = model.predict(source=tensor, verbose=False)[0]
        feat = adapter.get_activations(tensor)[scale]          # [1, C, H, W]
        H, W = feat.shape[2], feat.shape[3]
        covered = torch.zeros((H, W), dtype=torch.bool)

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = (box / stride).int().tolist()
            x1, x2 = min(max(x1, 0), W - 1), min(max(x2, 0), W)
            y1, y2 = min(max(y1, 0), H - 1), min(max(y2, 0), H)
            if x2 <= x1 or y2 <= y1:
                continue
            covered[y1:y2, x1:x2] = True
            cells = feat[0, :, y1:y2, x1:x2].reshape(feat.shape[1], -1).T.cpu()   # [cells, C]
            per_class[int(cls.item())].extend(cells.unbind(0))

        bg_coords = (~covered).nonzero(as_tuple=False).tolist()
        random.shuffle(bg_coords)
        for gy, gx in bg_coords[:bg_per_image]:
            per_class[bg_id].append(feat[0, :, gy, gx].cpu())

        if n % 200 == 0:
            print(f"  {n}/{len(images)} images")

    stats = {}
    for cid, feats in sorted(per_class.items()):
        label = "background" if cid == bg_id else f"class {cid}"
        if len(feats) <= min_samples:
            print(f"  Skipping {label}: only {len(feats)} samples (<= {min_samples})")
            continue
        x = torch.stack(feats).double()
        cov = torch.cov(x.T) + 1e-6 * torch.eye(x.shape[1], dtype=torch.double)
        stats[cid] = {"mean": x.mean(0).float(), "inv_cov": torch.linalg.inv(cov).float()}
        print(f"  {label}: {len(feats)} samples")

    with open(profile_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"\nSaved {len(stats)} profiles (incl. background) to {profile_path}")
    print("Next: python learn_thresholds.py  (uses threshold_images_dir)")


if __name__ == "__main__":
    main()
