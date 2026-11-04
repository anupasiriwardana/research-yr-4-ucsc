"""
Check on many images at once whether a patch actually hides the target objects.

For every patched image in <paths.patched_dir> whose name starts with `verify.patched_prefix` (default: the
prefix used by apply_patch_batch.py), the prefix is removed to find the ORIGINAL image in <paths.clean_dir>.
Both are run through the model from the config (same 640x640 pipeline as the patch training) and compared.

Per image:
  * a side-by-side picture  <paths.efficacy_dir>/<verify.output_prefix><original stem>.jpg
    (left = clean detections, right = patched detections, cyan boxes = where the patches were pasted
     when the coordinate file exists);
  * a row in efficacy_summary.csv.
Overall: how many of the clean target objects were hidden, and how many NEW detections appeared.

  hidden object  = a clean detection of the target class with no patched detection of the same class
                   at IoU >= verify.match_iou
  new detection  = a patched detection with no clean detection of the same class at IoU >= match_iou

Usage:
    python verify_patch_efficacy_batch.py --config config_yolov8.json
    python verify_patch_efficacy_batch.py --config config_yolov8.json --prefix multi-patched_ --num-images 20
"""

import argparse
import csv
import json

import cv2
import numpy as np
import torch

from patch_common import (DetModel, INPUT_SIZE, list_images, load_config, load_image, read_coordinate_file, resolve,
                          section)

CLEAN_COLOR, PATCHED_COLOR, GT_COLOR = (0, 200, 0), (0, 0, 255), (255, 255, 0)     # BGR


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def unmatched(src, other, match_iou):
    """Detections in `src` with no same-class detection in `other` at IoU >= match_iou."""
    return [d for d in src if not any(o[5] == d[5] and iou(d, o) >= match_iou for o in other)]


def draw(tensor, dets, names, color, title, patch_boxes=()):
    img = cv2.cvtColor((tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    for x1, y1, x2, y2 in patch_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 1)
    for x1, y1, x2, y2, conf, cls in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2)
        label = f"{names.get(cls, cls)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], max(p1[1] - th - 6, 0)), (p1[0] + tw + 4, max(p1[1], th + 6)), color, -1)
        cv2.putText(img, label, (p1[0] + 2, max(p1[1] - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prefix", default=None, help="override verify.patched_prefix (prefix to strip from patched names)")
    ap.add_argument("--num-images", type=int, default=None, help="override verify.num_images")
    ap.add_argument("--start-index", type=int, default=None, help="override verify.start_index")
    ap.add_argument("--patched-dir", default=None, help="override paths.patched_dir")
    ap.add_argument("--out-dir", default=None, help="override paths.efficacy_dir")
    ap.add_argument("--save-visualizations", action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths, v = section(cfg, "paths"), section(cfg, "verify")
    prefix = v.get("patched_prefix") if args.prefix is None else args.prefix
    if prefix is None:
        prefix = cfg.get("apply", {}).get("prefix", "")
    num_images = args.num_images or v.get("num_images")
    start = v.get("start_index", 0) if args.start_index is None else args.start_index
    conf, iou_nms, match_iou = v.get("conf_thresh", 0.25), v.get("iou_thresh", 0.7), v.get("match_iou", 0.5)
    target_class = v.get("target_class")
    save_viz = v.get("save_visualizations", True) if args.save_visualizations is None else args.save_visualizations
    out_prefix = v.get("output_prefix", "efficacy_")

    patched_dir = resolve(cfg, args.patched_dir or paths["patched_dir"])
    clean_dir = resolve(cfg, paths["clean_dir"])
    out_dir = resolve(cfg, args.out_dir or paths["efficacy_dir"])
    coord_dir = resolve(cfg, paths["coordinates_dir"]) if "coordinates_dir" in paths else None
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = v.get("image_extensions")
    patched = [p for p in list_images(patched_dir, 0, None, exts) if p.name.startswith(prefix)]
    patched = patched[start:start + num_images] if num_images else patched[start:]
    if not patched:
        raise FileNotFoundError(f"No patched images starting with '{prefix}' in {patched_dir}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetModel(cfg, device)
    print(f"Model {cfg['model']['weights']}; {len(patched)} patched image(s) with prefix '{prefix}' from {patched_dir}")

    rows, skipped = [], []
    for n, ppath in enumerate(patched, 1):
        original = ppath.name[len(prefix):]
        cpath = clean_dir / original
        if not cpath.exists():
            alt = [p for p in clean_dir.glob(f"{cpath.stem}.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            cpath = alt[0] if alt else None
        if cpath is None:
            skipped.append(ppath.name)
            print(f"[{n}/{len(patched)}] SKIP {ppath.name}: no original '{original}' in {clean_dir}")
            continue
        clean_t, patched_t = load_image(cpath, device), load_image(ppath, device)
        if clean_t is None or patched_t is None:
            skipped.append(ppath.name)
            print(f"[{n}/{len(patched)}] SKIP {ppath.name}: unreadable image")
            continue
        cdet = model.detect(clean_t.unsqueeze(0), conf, iou_nms)
        pdet = model.detect(patched_t.unsqueeze(0), conf, iou_nms)
        ctar = [d for d in cdet if target_class is None or d[5] == target_class]
        ptar = [d for d in pdet if target_class is None or d[5] == target_class]
        hidden = unmatched(ctar, pdet, match_iou)
        new = unmatched(pdet, cdet, match_iou)
        coord_name = cpath.stem + ".txt"
        patch_boxes = read_coordinate_file(coord_dir / coord_name) if coord_dir else []
        if not patch_boxes and coord_dir:
            patch_boxes = read_coordinate_file(coord_dir / (ppath.stem + ".txt"))
        rows.append({"image": cpath.name, "patched_image": ppath.name, "patches": len(patch_boxes),
                     "clean_detections": len(cdet), "patched_detections": len(pdet),
                     "clean_target": len(ctar), "patched_target": len(ptar),
                     "hidden_target": len(hidden), "new_detections": len(new)})
        if save_viz:
            left = draw(clean_t, cdet, model.names, CLEAN_COLOR, "CLEAN")
            right = draw(patched_t, pdet, model.names, PATCHED_COLOR, "PATCHED", patch_boxes)
            cv2.imwrite(str(out_dir / f"{out_prefix}{cpath.stem}.jpg"), cv2.hconcat([left, right]))
        print(f"[{n}/{len(patched)}] {cpath.name}: target {len(ctar)} -> {len(ptar)}, hidden {len(hidden)}/{len(ctar)}, new {len(new)}")

    if not rows:
        raise RuntimeError("Nothing was compared.")
    with (out_dir / "efficacy_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tot_t = sum(r["clean_target"] for r in rows)
    tot_h = sum(r["hidden_target"] for r in rows)
    with_t = [r for r in rows if r["clean_target"] > 0]
    all_hidden = sum(1 for r in with_t if r["hidden_target"] == r["clean_target"])
    summary = {
        "model": cfg["model"]["weights"], "patch": cfg.get("patch_name"), "images_compared": len(rows),
        "images_skipped": len(skipped), "target_class": target_class,
        "clean_target_objects": tot_t, "hidden_target_objects": tot_h,
        "hiding_rate": round(tot_h / tot_t, 4) if tot_t else None,
        "images_with_targets": len(with_t), "images_all_targets_hidden": all_hidden,
        "new_detections_total": sum(r["new_detections"] for r in rows),
        "settings": {"conf_thresh": conf, "iou_thresh": iou_nms, "match_iou": match_iou, "prefix": prefix},
    }
    with (out_dir / "efficacy_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Summary ===")
    for k, val in summary.items():
        if k != "settings":
            print(f"  {k}: {val}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
