"""
Run the cls-head patch detector on a batch of patched images.

For every processed image, this script writes one text file containing:

    latency_ms=...
    is_attack=...
    score=...
    box=(x1, y1, x2, y2) score=...

Boxes use the same 640x640 pixel coordinate system and tuple format returned
by cls_head_detector.py.
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "detection_config.json"

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    config = json.load(config_file)

DETECTOR_DIR = Path(config["detector_dir"]).resolve()
if str(DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTOR_DIR))

from cls_head_detector import ClsHeadMahalanobisDetector  # noqa: E402


BATCH_SETTINGS = config.get("batch_settings", {})
INPUT_DIR = Path(BATCH_SETTINGS["input_dir"])
OUTPUT_DIR = Path(BATCH_SETTINGS["coordinates_dir"])
NUM_IMAGES = BATCH_SETTINGS.get("num_images", 100)
START_INDEX = BATCH_SETTINGS.get("start_index", 0)
SAVE_VISUALIZATIONS = BATCH_SETTINGS.get("save_visualizations", True)
IMAGE_EXTENSIONS = {
    extension.lower()
    for extension in BATCH_SETTINGS.get(
        "image_extensions", [".jpg", ".jpeg", ".png"]
    )
}


def discover_images(start_index, num_images):
    candidates = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    selected = candidates[start_index:start_index + num_images]
    if not selected:
        raise FileNotFoundError(
            f"No images selected from {INPUT_DIR} with extensions "
            f"{sorted(IMAGE_EXTENSIONS)} and start_index={start_index}."
        )
    return selected


def write_detection_file(path, result, latency_ms):
    with path.open("w", encoding="utf-8") as detection_file:
        detection_file.write(f"latency_ms={latency_ms:.3f}\n")
        detection_file.write(f"is_attack={str(result['is_attack']).lower()}\n")
        detection_file.write(f"score={result['score']:.6f}\n")
        for entry in result["bounding_boxes"]:
            x1, y1, x2, y2 = entry["box"]
            detection_file.write(
                f"box=({x1}, {y1}, {x2}, {y2}) "
                f"score={entry['score']:.6f}\n"
            )


def process_image(detector, image_path):
    result = detector.detect(
        image_path, save_visualization=SAVE_VISUALIZATIONS
    )
    latency_ms = result["detection_latency_ms"]

    output_path = OUTPUT_DIR / f"{image_path.stem}.txt"
    write_detection_file(output_path, result, latency_ms)
    return output_path, result, latency_ms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cls-head detection on multiple patched images."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=NUM_IMAGES,
        help=f"Override batch_settings.num_images (default: {NUM_IMAGES}).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=START_INDEX,
        help=f"Override batch_settings.start_index (default: {START_INDEX}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("num_images must be greater than zero.")
    if args.start_index < 0:
        raise ValueError("start_index cannot be negative.")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = discover_images(args.start_index, args.num_images)

    detector_config = dict(config)
    detector_config["detection_output_dir"] = str(
        config.get("detector_output_dir", OUTPUT_DIR)
    )
    print(f"Loading detector from: {DETECTOR_DIR}")
    detector = ClsHeadMahalanobisDetector(detector_config)
    print(
        f"Processing {len(image_paths)} image(s), "
        f"device={detector.device}, threshold={detector.threshold}"
    )

    failures = []
    for position, image_path in enumerate(image_paths, start=1):
        try:
            output_path, result, latency_ms = process_image(
                detector, image_path
            )
            print(
                f"[{position}/{len(image_paths)}] {image_path.name}: "
                f"{len(result['bounding_boxes'])} detection(s), "
                f"latency={latency_ms:.3f} ms, output={output_path.name}"
            )
        except (OSError, RuntimeError, ValueError) as error:
            failures.append((image_path, error))
            print(f"[{position}/{len(image_paths)}] FAILED {image_path.name}: {error}")

    if failures:
        failure_names = ", ".join(path.name for path, _ in failures)
        raise RuntimeError(
            f"{len(failures)} image(s) failed: {failure_names}"
        )

    print(f"Completed successfully. Detection files: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
