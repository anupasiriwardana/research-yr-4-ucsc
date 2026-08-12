import cv2
import json
import torch
from pathlib import Path
from ultralytics import YOLO

# 1. Load Configuration
CONFIG_PATH = Path(__file__).parent / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

CLEAN_DIR = Path(config["clean_dir"])
PATCHED_DIR = Path(config["output_dir"])
PATCH_DETECTION_DIR = Path(config["patch_detection_dir"])

# 2. Resolve Patched Image Path
if config["specific_patched_image"]:
    patched_image_path = PATCHED_DIR / config["specific_patched_image"]
else:
    patched_files = list(PATCHED_DIR.glob("*.jpg")) + list(PATCHED_DIR.glob("*.png"))
    # Exclude prior visual comparison outputs
    patched_files = [f for f in patched_files if "efficacy_comparison" not in f.name]
    if not patched_files:
        raise FileNotFoundError(f"No patched image found in {PATCHED_DIR}")
    patched_image_path = patched_files[0]

# 3. Resolve Clean Image Path
if config["specific_clean_image"]:
    clean_image_path = CLEAN_DIR / config["specific_clean_image"]
else:
    clean_filename = patched_image_path.name.replace("applied_art_patched_", "").replace("art_patched_", "")
    clean_image_path = CLEAN_DIR / clean_filename
    if not clean_image_path.exists():
        clean_files = list(CLEAN_DIR.glob("*.jpg")) + list(CLEAN_DIR.glob("*.png"))
        clean_image_path = clean_files[0]

print(f"Comparing:")
print(f"  Clean Image:   {clean_image_path}")
print(f"  Patched Image: {patched_image_path}\n")

# 4. Load Model & Execute Inference
model = YOLO(config["model_path"])

clean_results = model(str(clean_image_path), verbose=False)[0]
patched_results = model(str(patched_image_path), verbose=False)[0]

# 5. Display Quantitative Summary
def summarize_detections(results, title):
    boxes = results.boxes
    print(f"=== {title} ===")
    print(f"Total Detections: {len(boxes)}")
    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]
            conf = float(box.conf.item())
            coords = [round(c, 1) for c in box.xyxy[0].tolist()]
            print(f"  [{i+1}] Class: {cls_name:<12} Conf: {conf:.2f}  Box: {coords}")
    else:
        print("  No objects detected.")
    print()

summarize_detections(clean_results, "CLEAN IMAGE DETECTIONS")
summarize_detections(patched_results, "PATCHED IMAGE DETECTIONS")

# 6. Save Visual Side-by-Side Comparison
clean_plot = clean_results.plot()
patched_plot = patched_results.plot()

h, w, _ = clean_plot.shape
patched_plot_resized = cv2.resize(patched_plot, (w, h))

comparison_img = cv2.hconcat([clean_plot, patched_plot_resized])

cv2.putText(comparison_img, "CLEAN IMAGE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
cv2.putText(comparison_img, "PATCHED IMAGE", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

output_visual_path = PATCH_DETECTION_DIR / f"efficacy_comparison_{config["specific_patched_image"]}.jpg"
cv2.imwrite(str(output_visual_path), comparison_img)

print(f"Visual comparison saved to: {output_visual_path}")