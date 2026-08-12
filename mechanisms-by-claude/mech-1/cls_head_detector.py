import torch
import cv2
import pickle
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from cls_head_adapter import YOLOv8ClsHeadAdapter

class ClsHeadMahalanobisDetector:
    def __init__(self, model_version='yolov8n.pt', threshold=15.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_version).to(self.device)
        self.adapter = YOLOv8ClsHeadAdapter(self.model.model)
        self.threshold = threshold
        self.stride = 8
        
        # Load the offline calibration artifact
        profile_path = Path(r"D:\GitHub\yr-4-research\mechanisms-by-claude\mech-1\profiles\cls_head_calibration_p3.pkl")
        with open(profile_path, 'rb') as f:
            self.stats = pickle.load(f)
            
        # Move stats to GPU for fast runtime matrix multiplication
        for cls_id in self.stats:
            self.stats[cls_id]['mean'] = self.stats[cls_id]['mean'].to(self.device)
            self.stats[cls_id]['inv_cov'] = self.stats[cls_id]['inv_cov'].to(self.device)

    def detect(self, img_path, save_visualization=True):
        results = self.model(img_path, verbose=False)[0]
        
        # Create a predicted class grid matching the P3 spatial resolution
        img_h, img_w = results.orig_shape
        grid_h, grid_w = img_h // self.stride, img_w // self.stride
        predicted_classes = torch.full((grid_h, grid_w), -1, dtype=torch.long, device=self.device)
        
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            x1, y1, x2, y2 = (box / self.stride).int()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(grid_w, x2), min(grid_h, y2)
            predicted_classes[y1:y2, x1:x2] = int(cls.item())

        img_tensor = torch.from_numpy(results.orig_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        activations = self.adapter.get_activations(img_tensor)
        feat = activations["P3"]
        
        B, C, H, W = feat.shape
        feat_flat = feat.permute(0, 2, 3, 1).reshape(H * W, C)

        scores = torch.zeros(H, W, device=self.device)
        
        # Compute Mahalanobis distance strictly on regions with predicted objects
        for idx in range(H * W):
            y, x = divmod(idx, W)
            cls_id = predicted_classes[y, x].item()
            
            if cls_id in self.stats:
                mean = self.stats[cls_id]['mean']
                inv_cov = self.stats[cls_id]['inv_cov']
                diff = feat_flat[idx] - mean
                scores[y, x] = torch.sqrt(torch.matmul(torch.matmul(diff, inv_cov), diff.unsqueeze(-1))).squeeze()

        # Free VRAM
        torch.cuda.empty_cache()

        # Masking and extraction
        mask = (scores > self.threshold)
        is_attack = mask.any().item()
        score_val = scores.max().item()
        bounding_box = (0, 0, 0, 0)

        if is_attack:
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                # Scale up from grid coordinates to pixel coordinates using the stride
                bounding_box = (x * self.stride, y * self.stride, (x + w) * self.stride, (y + h) * self.stride)

        # --- VISUALIZATION OVERLAY GENERATOR ---
        if save_visualization:
            scores_np = scores.cpu().numpy()
            
            # Normalize Mahalanobis distance scores to [0, 255] range for display
            scores_norm = cv2.normalize(scores_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Resize heatmap grid to match full input image resolution
            heatmap_resized = cv2.resize(scores_norm, (img_w, img_h))
            
            # Apply JET color map (Red = High Mahalanobis anomaly score, Blue = Low/Normal)
            colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            # Blend 50% heatmap over original image
            original_bgr = results.orig_img
            overlay = cv2.addWeighted(original_bgr, 0.5, colored_heatmap, 0.5, 0)
            
            # Draw detected bounding box in bright red if an attack was triggered
            if is_attack and bounding_box != (0, 0, 0, 0):
                x1, y1, x2, y2 = bounding_box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(overlay, f"PATCH DETECTED (Score: {score_val:.1f})", 
                            (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            out_visualization_path = Path(img_path).parent / f"detected_cls_head_{Path(img_path).name}"
            cv2.imwrite(str(out_visualization_path), overlay)
            print(f"\n[Visualizer] Heatmap saved to: {out_visualization_path}")

        # The contract output remains strictly identical
        return {
            "is_attack": is_attack,
            "score": float(score_val),
            "bounding_box": bounding_box
        }

if __name__ == "__main__":
    detector = ClsHeadMahalanobisDetector()
    
    # Point to your generated patched image
    PATCHED_DIR = Path(r"D:\GitHub\yr-4-research\mechanisms-by-claude\gen-adv-patch\bdd100k_patched")
    patched_files = list(PATCHED_DIR.glob("*.jpg")) + list(PATCHED_DIR.glob("*.png"))
    
    if patched_files:
        test_image = str(patched_files[0])
        result = detector.detect(test_image, save_visualization=True)
        print("\n--- Decoupled Head Middleware Output ---")
        print(f"Attack Detected: {result['is_attack']}")
        print(f"Anomaly Score:   {result['score']:.2f}")
        print(f"Bounding Box:    {result['bounding_box']}")
    else:
        print("No patched image found in target directory.")