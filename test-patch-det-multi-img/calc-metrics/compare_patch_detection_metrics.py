"""Compare patch-coordinate files and calculate detection metrics.

The comparison is performed in the shared 640x640 pixel coordinate system.
A detected box is a true positive when it has a positive-area intersection
with an unmatched ground-truth box. Exact coordinate equality is not
required: a smaller box inside the patch, a larger box around it, and any
other overlapping box are all accepted. Each box can be matched only once.
Candidate pairs are sorted by IoU before matching so that ambiguous overlaps
are resolved consistently.

Object-level metrics:
    TP: matched ground-truth/detected box pairs.
    FP: detected boxes that could not be matched to a ground-truth box.
    FN: ground-truth boxes with no matching detection.
    Precision: TP / (TP + FP), the fraction of detections that are valid.
    Recall: TP / (TP + FN), the fraction of patches that were detected.
    F1: harmonic mean of precision and recall.

Image-level metrics treat an image as positive when it has at least one
ground-truth patch or detection. This additionally provides a clean-image
false-positive rate (FP / (FP + TN)); TN is a clean image with no detection.
Latency is read from ``latency_ms=...`` in each detection file and averaged
over files that contain a valid latency value.
"""

import argparse
import csv
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "metrics_config.json"
BOX_PATTERN = re.compile(
    r"box\s*=\s*\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)",
    re.IGNORECASE,
)
LATENCY_PATTERN = re.compile(
    r"latency_ms\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE
)


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def parse_boxes(path):
    text = path.read_text(encoding="utf-8")
    boxes = []
    for match in BOX_PATTERN.finditer(text):
        values = tuple(float(value) for value in match.groups())
        x1, y1, x2, y2 = values
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right <= left or bottom <= top:
            raise ValueError(f"Invalid box with non-positive area in {path}: {values}")
        boxes.append((left, top, right, bottom))
    return boxes


def parse_latency(path):
    text = path.read_text(encoding="utf-8")
    match = LATENCY_PATTERN.search(text)
    if match is None:
        return None
    latency = float(match.group(1))
    if latency < 0:
        raise ValueError(f"Latency cannot be negative in {path}: {latency}")
    return latency


def intersection_area(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def iou(first, second):
    intersection = intersection_area(first, second)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def match_boxes(ground_truth, detections):
    candidates = [
        (iou(gt_box, detected_box), gt_index, detected_index)
        for gt_index, gt_box in enumerate(ground_truth)
        for detected_index, detected_box in enumerate(detections)
        if intersection_area(gt_box, detected_box) > 0
    ]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_gt = set()
    matched_detections = set()
    matches = []
    for overlap, gt_index, detected_index in candidates:
        if gt_index in matched_gt or detected_index in matched_detections:
            continue
        matched_gt.add(gt_index)
        matched_detections.add(detected_index)
        matches.append((gt_index, detected_index, overlap))
    return matches


def image_key(path, detection_prefix):
    key = path.stem
    if key.startswith(detection_prefix):
        key = key[len(detection_prefix):]
    return key


def discover_files(directory):
    return {path.stem: path for path in directory.glob("*.txt")}


def discover_detection_files(directory, detection_prefix):
    detection_files = {}
    duplicate_keys = set()
    for path in directory.glob("*.txt"):
        key = image_key(path, detection_prefix)
        if key in detection_files:
            duplicate_keys.add(key)
        else:
            detection_files[key] = path
    if duplicate_keys:
        duplicates = ", ".join(sorted(duplicate_keys))
        raise ValueError(f"Duplicate detection files for image key(s): {duplicates}")
    return detection_files


def metric_summary(tp, fp, fn):
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_ratio(2 * precision * recall, precision + recall),
    }


def compare(config):
    ground_truth_dir = Path(config["ground_truth_dir"])
    detection_dir = Path(config["detection_dir"])
    output_dir = Path(config["output_dir"])
    detection_prefix = config.get("detection_filename_prefix", "multi-patched_")
    strict_matching = config.get("strict_file_matching", True)

    if not ground_truth_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {ground_truth_dir}")
    if not detection_dir.is_dir():
        raise FileNotFoundError(f"Detection directory does not exist: {detection_dir}")

    ground_truth_files = discover_files(ground_truth_dir)
    detection_files = discover_detection_files(detection_dir, detection_prefix)
    all_keys = sorted(set(ground_truth_files) | set(detection_files))
    missing_ground_truth = sorted(set(detection_files) - set(ground_truth_files))
    missing_detections = sorted(set(ground_truth_files) - set(detection_files))
    if strict_matching and (missing_ground_truth or missing_detections):
        raise ValueError(
            "Coordinate-file mismatch. "
            f"Missing ground truth: {missing_ground_truth[:10]}; "
            f"missing detections: {missing_detections[:10]}"
        )

    per_image = []
    latencies = []
    object_tp = object_fp = object_fn = 0
    image_tp = image_fp = image_fn = image_tn = 0
    for key in all_keys:
        ground_truth_path = ground_truth_files.get(key)
        detection_path = detection_files.get(key)
        ground_truth = parse_boxes(ground_truth_path) if ground_truth_path else []
        detections = parse_boxes(detection_path) if detection_path else []
        latency = parse_latency(detection_path) if detection_path else None
        if latency is not None:
            latencies.append(latency)

        matches = match_boxes(ground_truth, detections)
        tp = len(matches)
        fp = len(detections) - tp
        fn = len(ground_truth) - tp
        object_tp += tp
        object_fp += fp
        object_fn += fn

        actual_positive = bool(ground_truth)
        predicted_positive = bool(detections)
        if actual_positive and predicted_positive:
            image_tp += 1
            image_status = "TP"
        elif actual_positive:
            image_fn += 1
            image_status = "FN"
        elif predicted_positive:
            image_fp += 1
            image_status = "FP"
        else:
            image_tn += 1
            image_status = "TN"

        per_image.append({
            "image_key": key,
            "ground_truth_file": ground_truth_path.name if ground_truth_path else None,
            "detection_file": detection_path.name if detection_path else None,
            "ground_truth_boxes": len(ground_truth),
            "detected_boxes": len(detections),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "image_status": image_status,
            "latency_ms": latency,
            "matched_ious": [round(overlap, 6) for _, _, overlap in matches],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    object_metrics = metric_summary(object_tp, object_fp, object_fn)
    image_metrics = {
        **metric_summary(image_tp, image_fp, image_fn),
        "true_negatives": image_tn,
        "false_positive_rate": safe_ratio(image_fp, image_fp + image_tn),
    }
    result = {
        "coordinate_matching": {
            "criterion": "positive-area intersection",
            "one_to_one_matching": True,
            "matching_tie_breaker": "highest IoU first",
            "coordinate_space": "640x640 pixel space",
        },
        "dataset": {
            "images_compared": len(per_image),
            "ground_truth_directory": str(ground_truth_dir),
            "detection_directory": str(detection_dir),
            "missing_ground_truth_files": missing_ground_truth,
            "missing_detection_files": missing_detections,
        },
        "object_metrics": object_metrics,
        "image_metrics": image_metrics,
        "latency": {
            "images_with_latency": len(latencies),
            "average_ms": safe_ratio(sum(latencies), len(latencies)),
            "minimum_ms": min(latencies) if latencies else None,
            "maximum_ms": max(latencies) if latencies else None,
        },
        "per_image": per_image,
    }
    metrics_path = output_dir / config.get("metrics_json", "patch_detection_metrics.json")
    csv_path = output_dir / config.get("per_image_csv", "patch_detection_per_image.csv")
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fieldnames = list(per_image[0]) if per_image else [
        "image_key", "ground_truth_file", "detection_file", "ground_truth_boxes",
        "detected_boxes", "true_positives", "false_positives", "false_negatives",
        "image_status", "latency_ms", "matched_ious",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_image:
            csv_row = dict(row)
            csv_row["matched_ious"] = ";".join(map(str, row["matched_ious"]))
            writer.writerow(csv_row)
    return result, metrics_path, csv_path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare patch detections with patch coordinates.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    result, metrics_path, csv_path = compare(config)
    print(f"Compared {result['dataset']['images_compared']} image(s).")
    print(f"Object metrics: {json.dumps(result['object_metrics'])}")
    print(f"Image metrics: {json.dumps(result['image_metrics'])}")
    print(f"Average latency: {result['latency']['average_ms']:.3f} ms")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Per-image CSV: {csv_path}")


if __name__ == "__main__":
    main()
