import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# 1. Directory & File Setup
CLEAN_DIR = Path(r"D:\GitHub\experiment_data\bdd100k_images_10k\train")
PATCHED_DIR = Path(r"D:\GitHub\yr-4-research\mechanisms-by-claude\gen-adv-patch\bdd100k_patched")

# Locate the generated patched image
patched_files = list(PATCHED_DIR.glob("*.jpg")) + list(PATCHED_DIR.glob("*.png"))
if not patched_files:
    raise FileNotFoundError(f"No patched image found in {PATCHED_DIR}")

patched_image_path = patched_files[0]

# Determine original clean image filename from the patched filename prefix
clean_filename = patched_image_path.name.replace("applied_art_patched_", "").replace("art_patched_", "")
clean_image_path = CLEAN_DIR / clean_filename

if not clean_image_path.exists():
    # Fallback search if filename formatting differs
    clean_files = list(CLEAN_DIR.glob("*.jpg")) + list(CLEAN_DIR.glob("*.png"))
    clean_image_path = clean_files[0]

print(f"Comparing:")
print(f"  Clean Image:   {clean_image_path}")
print(f"  Patched Image: {patched_image_path}\n")

# 2. Load YOLOv8 Model
model = YOLO("yolov8n.pt")

# 3. Run Inference on Both Images
clean_results = model(str(clean_image_path), verbose=False)[0]
patched_results = model(str(patched_image_path), verbose=False)[0]

# 4. Display Quantitative Summary
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

# 5. Save Visual Comparison Side by Side
clean_plot = clean_results.plot()    # Draws bounding boxes on clean image
patched_plot = patched_results.plot()  # Draws bounding boxes on patched image

# Resize both to equal dimensions for side-by-side concatenation
h, w, _ = clean_plot.shape
patched_plot_resized = cv2.resize(patched_plot, (w, h))

comparison_img = cv2.hconcat([clean_plot, patched_plot_resized])

# Add text labels to the top of each half
cv2.putText(comparison_img, "CLEAN IMAGE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
cv2.putText(comparison_img, "PATCHED IMAGE", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

output_visual_path = PATCHED_DIR / "efficacy_comparison.jpg"
cv2.imwrite(str(output_visual_path), comparison_img)

print(f"Visual comparison saved to: {output_visual_path}")