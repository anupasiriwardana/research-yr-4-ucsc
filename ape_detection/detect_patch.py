import sys
from pathlib import Path
import json
from ape_detector import ApeDetector

SCRIPT_DIR = Path(__file__).resolve().parent
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def run_benchmark(config_file: str, model_label: str):
    config_path = SCRIPT_DIR / config_file
    print(f"\n{'=' * 80}")
    print(f"🚀 Running Benchmark on {model_label} ({config_file})")
    print(f"{'=' * 80}")

    if not config_path.exists():
        print(f"❌ Error: {config_file} not found. Skipping.")
        return

    # 1. Load config to find the test directory
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    test_dir = Path(config_data.get("test_images_dir", "ape_detection/test_images"))
    if not test_dir.exists():
        print(f"❌ Error: Image directory {test_dir} not found.")
        return
        
    image_files = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS])
    print(f"📂 Found {len(image_files)} images in: {test_dir}\n")

    # 2. Initialize Detector
    detector = ApeDetector(config_path=str(config_path))
    results = []

    # 3. Iterate through images
    for img_path in image_files:
        res = detector.detect(str(img_path), save_visualization=True)
        results.append(res)

    # 4. Print Results Summary
    total_images = len(results)
    attack_count = sum(1 for r in results if r["is_attack"])

    print(f"\n📊 --- Summary for {model_label} ---")
    print(f"Total Images Evaluated : {total_images}")
    print(f"Attacks / Anomalies    : {attack_count}")
    print(f"Clean Frames           : {total_images - attack_count}")
    print("-" * 80)
    print(f"{'Image':<30} | {'Detections':<12} | {'Is Attack?':<12} | {'Max Z-Score':<10}")
    print("-" * 80)
    
    for r in results:
        max_z = r.get("max_z_score", 0.0)
        status = "🚨 YES" if r["is_attack"] else "✅ NO"
        print(f"{r['image']:<30} | {r['detections_count']:<12} | {status:<12} | {max_z:<10.2f}")

    detector.close()

if __name__ == "__main__":
    Path("ape_detection/test_images/original").mkdir(parents=True, exist_ok=True)
    run_benchmark(config_file="config_yolov5.json", model_label="YOLOv5")
