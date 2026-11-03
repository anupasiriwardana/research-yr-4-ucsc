"""
Run Mech-7 on a folder of images (patched OR clean).

Per image it writes `<stem>.txt` in the same format the Mech-1 batch script
uses, so calc-metrics/compare_patch_detection_metrics.py works unchanged:

    latency_ms=...
    is_attack=true|false
    score=...
    box=(x1, y1, x2, y2) score=...

It also writes `run_summary.json` in the output folder: images with >=1
detection, total / mean boxes, latency stats. On a CLEAN folder,
`images_with_detection / images` is the image-level false-positive rate.

Overrides let you point at another folder without editing config.json:
    python detect_mech7_multi_img.py --input-dir <clean_dir> --coordinates-dir <out_dir> --num-images 300
"""

import argparse
import json
from pathlib import Path

from mech7_core import load_config, list_images
from mech7_detector import Mech7Detector


def write_detection_file(path, result, latency_ms):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"latency_ms={latency_ms:.3f}\n")
        f.write(f"is_attack={str(result['is_attack']).lower()}\n")
        f.write(f"score={result['score']:.6f}\n")
        for e in result["bounding_boxes"]:
            x1, y1, x2, y2 = e["box"]
            f.write(f"box=({x1}, {y1}, {x2}, {y2}) score={e['score']:.6f}\n")


def main():
    cfg = load_config()
    b = cfg["batch_settings"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--input-dir", default=b["input_dir"])
    ap.add_argument("--coordinates-dir", default=b["coordinates_dir"])
    ap.add_argument("--num-images", type=int, default=b.get("num_images", 100))
    ap.add_argument("--start-index", type=int, default=b.get("start_index", 0))
    ap.add_argument("--no-visualizations", action="store_true")
    ap.add_argument("--save-score-maps", action=argparse.BooleanOptionalAction, default=None,
                    help="override batch_settings.save_score_maps")
    ap.add_argument("--score-maps-dir", default=None, help="override batch_settings.score_maps_dir")
    args = ap.parse_args()
    if args.config:
        cfg = load_config(args.config)
        b = cfg["batch_settings"]
    if args.save_score_maps is not None:
        b["save_score_maps"] = args.save_score_maps
    if args.score_maps_dir:
        b["score_maps_dir"] = args.score_maps_dir
    if b.get("save_score_maps") and not b.get("score_maps_dir"):
        b["score_maps_dir"] = str(Path(args.coordinates_dir) / "score_maps")
    if args.num_images <= 0 or args.start_index < 0:
        raise ValueError("num_images must be > 0 and start_index >= 0.")

    out_dir = Path(args.coordinates_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(args.input_dir)[args.start_index:args.start_index + args.num_images]
    if not images:
        raise FileNotFoundError(f"No images selected from {args.input_dir} (start_index={args.start_index}).")

    viz = b.get("save_visualizations", True) and not args.no_visualizations
    detector = Mech7Detector(cfg)

    latencies, box_counts, failures = [], [], []
    for n, img in enumerate(images, 1):
        try:
            result = detector.detect(img, save_visualization=viz)
        except (OSError, RuntimeError, ValueError) as e:
            failures.append(img.name)
            print(f"[{n}/{len(images)}] FAILED {img.name}: {e}")
            continue
        lat = result["detection_latency_ms"]
        write_detection_file(out_dir / f"{img.stem}.txt", result, lat)
        latencies.append(lat)
        box_counts.append(len(result["bounding_boxes"]))
        print(f"[{n}/{len(images)}] {img.name}: {box_counts[-1]} detection(s), latency={lat:.3f} ms")

    s = cfg["detector_settings"]
    summary = {
        "input_dir": str(args.input_dir),
        "settings": {k: s.get(k) for k in ("tap_scale", "stride", "threshold_mode", "threshold_percentile",
                                            "adjust_threshold", "global_threshold",
                                            "min_class_samples_for_own_threshold", "min_region_cells")},
        "images": len(box_counts),
        "images_with_detection": sum(1 for c in box_counts if c > 0),
        "total_boxes": sum(box_counts),
        "mean_boxes_per_image": (sum(box_counts) / len(box_counts)) if box_counts else 0.0,
        "latency_ms": {"mean": sum(latencies) / len(latencies) if latencies else None,
                       "min": min(latencies, default=None), "max": max(latencies, default=None)},
        "failures": failures,
    }
    (out_dir / b.get("summary_filename", "run_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} image(s) failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
