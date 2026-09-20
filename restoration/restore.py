import cv2
import numpy as np
from pathlib import Path

# 1. Load the target image
image_path = "restoration/data/4-applied_art_patched_0d1fc580-44142e94.jpg"
input_stem = Path(image_path).stem
img = cv2.imread(image_path)

# The dimensions used by the detector
INPUT_WIDTH = 640
INPUT_HEIGHT = 640   

if img is None:
    print(f"Error: Could not load {image_path}")
else:
    # Original image dimensions
    orig_height, orig_width = img.shape[:2]
    
    # Calculate scaling factors
    scale_x = orig_width / INPUT_WIDTH
    scale_y = orig_height / INPUT_HEIGHT

    # 2. Define the original bounding box coordinates from the detector
    x1, y1, x2, y2 = 64, 24, 192, 152
    
    # Scale coordinates to match the original image size
    # Using int() to ensure pixel indices are whole numbers
    x1_scaled = int(x1 * scale_x)
    y1_scaled = int(y1 * scale_y)
    x2_scaled = int(x2 * scale_x)
    y2_scaled = int(y2 * scale_y)
    
    # ==========================================
    # OPTION 1: Absolute Minimum Compute (Solid Fill)
    # ==========================================
    img_occluded = img.copy()
    img_occluded[y1_scaled:y2_scaled, x1_scaled:x2_scaled] = (128, 128, 128) 
    cv2.imwrite("restoration/restored_solid_fill_1.jpg", img_occluded)

    # ==========================================
    # OPTION 2: Basic Restoration (Telea Inpainting)
    # ==========================================
    mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
    
    # Apply scaled coordinates to the mask
    mask[y1_scaled:y2_scaled, x1_scaled:x2_scaled] = 255
    
    img_inpainted = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    # img_inpainted = cv2.inpaint(img, mask, inpaintRadius=2, flags=cv2.INPAINT_NS)
    cv2.imwrite(f"restoration/runs/restored_inpainted_{input_stem}.jpg", img_inpainted)