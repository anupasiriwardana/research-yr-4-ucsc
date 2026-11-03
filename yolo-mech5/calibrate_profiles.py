"""
Step 1 of 2 for Mech-5: build the per-class (+ background) Mahalanobis
profiles of EVERY configured scale from the images in `profile_images_dir`.

Same procedure as Mech-1 Option B at each scale (cells inside the detector's
boxes feed that class's profile; a capped random sample of uncovered cells
feeds the background profile), done in one pass over the images with all
scales tapped at once. Each scale's own stride projects the boxes onto its own
grid. Thresholds are NOT learned here -- they come from a different, disjoint
image set in learn_thresholds.py.
"""

import argparse
import pickle
import random
from collections import defaultdict
from pathlib import Path

import torch
from ultralytics import YOLO

from mech5_core import load_config, validated_settings, ordered_scales, STRIDE_BY_SCALE, list_images, load_image_tensor
from ultralytics_cls_head_adapter import UltralyticsClsHeadAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-images", type=int, default=None, help="override detector_settings.max_calibration_images")
    args = ap.parse_args()

    config = load_config(args.config)
    s = validated_settings(config)
    scales = ordered_scales(config)
    min_samples = s["min_calibration_samples"]
    bg_id = s.get("background_class_id", -1)
    bg_per_image = s.get("background_samples_per_image", 250)
    max_images = args.max_images or s.get("max_calibration_images")

    profile_dir = Path(config["profiles_dir"])
    profile_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(config["model_path"]).to(device)
    adapter = UltralyticsClsHeadAdapter(model.model, scales=tuple(scales))
    print(adapter.describe())

    images = list_images(config["profile_images_dir"], max_images)
    print(f"Building profiles from {len(images)} images in {config['profile_images_dir']} (scales={scales})")

    per_class = {sc: defaultdict(list) for sc in scales}
    for n, img_path in enumerate(images, 1):
        _, tensor = load_image_tensor(img_path, device)
        results = model.predict(source=tensor, verbose=False)[0]
        acts = adapter.get_activations(tensor)
        for sc in scales:
            stride = STRIDE_BY_SCALE[sc]
            feat = acts[sc]                                    # [1, C, H, W]
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
                per_class[sc][int(cls.item())].extend(cells.unbind(0))
            bg_coords = (~covered).nonzero(as_tuple=False).tolist()
            random.shuffle(bg_coords)
            for gy, gx in bg_coords[:bg_per_image]:
                per_class[sc][bg_id].append(feat[0, :, gy, gx].cpu())
        if n % 200 == 0:
            print(f"  {n}/{len(images)} images")

    for sc in scales:
        stats = {}
        for cid, feats in sorted(per_class[sc].items()):
            label = "background" if cid == bg_id else f"class {cid}"
            if len(feats) <= min_samples:
                print(f"  [{sc}] Skipping {label}: only {len(feats)} samples (<= {min_samples})")
                continue
            x = torch.stack(feats).double()
            cov = torch.cov(x.T) + 1e-6 * torch.eye(x.shape[1], dtype=torch.double)
            stats[cid] = {"mean": x.mean(0).float(), "inv_cov": torch.linalg.inv(cov).float()}
            print(f"  [{sc}] {label}: {len(feats)} samples")
        path = profile_dir / config["scales"][sc]["profile_filename"]
        with open(path, "wb") as f:
            pickle.dump(stats, f)
        print(f"[{sc}] saved {len(stats)} profiles (incl. background) to {path}")
    print("\nNext: python learn_thresholds.py  (uses threshold_images_dir)")


if __name__ == "__main__":
    main()
