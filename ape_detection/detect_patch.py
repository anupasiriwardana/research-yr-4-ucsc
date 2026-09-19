import sys
from pathlib import Path
from ape_detector import ApeDetector

# Resolve paths relative to this script's directory
SCRIPT_DIR = Path(__file__).resolve().parent


def run_benchmark(config_file: str, model_label: str):
    config_path = SCRIPT_DIR / config_file
    print(f"\n{'=' * 60}")
    print(f"🚀 Running Benchmark on {model_label} ({config_file})")
    print(f"{'=' * 60}")

    if not config_path.exists():
        print(f"❌ Error: {config_file} not found. Skipping.")
        return

    # 1. Initialize Detector
    detector = ApeDetector(config_path=str(config_path))

    # 2. Run over all test images
    results = detector.detect_folder(save_visualization=True)

    # 3. Print Results Summary
    total_images = len(results)
    attack_count = sum(1 for r in results if r["is_attack"])

    print(f"\n📊 --- Summary for {model_label} ---")
    print(f"Total Images Evaluated : {total_images}")
    print(f"Attacks / Anomalies    : {attack_count}")
    print(f"Clean Frames           : {total_images - attack_count}")
    print("-" * 60)
    print(f"{'Image':<30} | {'Detections':<12} | {'Is Attack?':<12} | {'Max Z-Score':<10}")
    print("-" * 60)
    
    for r in results:
        max_z = r["defense_stats"].get("max_z_score", 0.0)
        status = "🚨 YES" if r["is_attack"] else "✅ NO"
        print(f"{r['image']:<30} | {r['detections_count']:<12} | {status:<12} | {max_z:<10.2f}")

    # Clean up forward hooks so they don't persist
    detector.close()


def main():
    # Make sure test images directory exists
    Path("ape_detection/test_images").mkdir(parents=True, exist_ok=True)

    # Test YOLOv8
    # run_benchmark(config_file="config_yolov8.json", model_label="YOLOv8")

    # Test YOLOv5
    run_benchmark(config_file="config_yolov5.json", model_label="YOLOv5")


if __name__ == "__main__":
    main()