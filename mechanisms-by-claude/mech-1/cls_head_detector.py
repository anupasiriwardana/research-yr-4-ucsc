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
        profile_path = Path(r"D:\GitHub\experiment_data\profiles\cls_head_calibration_p3.pkl")
        with open(profile_path, 'rb') as f:
            self.stats = pickle.load(f)
            
        # Move stats to GPU for fast runtime matrix multiplication
        for cls_id in self.stats:
            self.stats[cls_id]['mean'] = self.stats[cls_id]['mean'].to(self.device)
            self.stats[cls_id]['inv_cov'] = self.stats[cls_id]['inv_cov'].to(self.device)

    def detect(self, img_path):
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
                # (diff @ inv_cov @ diff).sqrt() calculation
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

        return {
            "is_attack": is_attack,
            "score": float(score_val),
            "bounding_box": bounding_box
        }

if __name__ == "__main__":
    detector = ClsHeadMahalanobisDetector()
    test_image = r"D:\GitHub\experiment_data\bdd100k_patched\patched_ac9be3fe-790d1f8e.jpg"
    
    result = detector.detect(test_image)
    print("\n--- Decoupled Head Middleware Output ---")
    print(f"Attack Detected: {result['is_attack']}")
    print(f"Anomaly Score:   {result['score']:.2f}")
    print(f"Bounding Box:    {result['bounding_box']}")