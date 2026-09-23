import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("ApeDetector")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class AdversarialPatchExtractor:
    """Extracts adversarial patch bounding boxes from a 2D Z-score feature activation map."""
    
    DEFAULT_CONFIG = {
        "floor": 2.6,
        "smooth_k": 5,
        "levels": [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40],
        "core_frac": 0.70,
        "close_k": 3,
        "open_k": 5,
        "min_area": 80,
        "max_area_frac": 0.05,
        "min_side": 8,
        "max_aspect": 2.2,
        "min_rect": 0.60,
        "min_density": 0.55,
        "nms_iou": 0.30,
        "refine": True,
        "max_glare_frac": 0.10,
        "glare_v_thresh": 245,
        "glare_pad": 2,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.floor = float(cfg["floor"])
        self.smooth_k = int(cfg["smooth_k"])
        self.levels = tuple(cfg["levels"])
        self.core_frac = float(cfg["core_frac"])
        self.close_k = int(cfg["close_k"])
        self.open_k = int(cfg["open_k"])
        self.min_area = int(cfg["min_area"])
        self.max_area_frac = float(cfg["max_area_frac"])
        self.min_side = int(cfg["min_side"])
        self.max_aspect = float(cfg["max_aspect"])
        self.min_rect = float(cfg["min_rect"])
        self.min_density = float(cfg["min_density"])
        self.nms_iou = float(cfg["nms_iou"])
        self.refine = bool(cfg["refine"])
        self.max_glare_frac = float(cfg["max_glare_frac"])
        self.glare_v_thresh = int(cfg["glare_v_thresh"])
        self.glare_pad = int(cfg["glare_pad"])

        # Pre-compile morphological kernels so they aren't rebuilt in every loop iteration
        self.k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (self.close_k, self.close_k)) if self.close_k > 1 else None
        self.k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (self.open_k, self.open_k)) if self.open_k > 1 else None

    def _blown_highlight_fraction(self, frame: np.ndarray, box_cells: List[int], scale_x: float, scale_y: float) -> float:
        x1, y1, x2, y2 = box_cells
        fx1 = max(0, int(x1 * scale_x) - self.glare_pad)
        fy1 = max(0, int(y1 * scale_y) - self.glare_pad)
        fx2 = min(frame.shape[1], int(x2 * scale_x) + self.glare_pad)
        fy2 = min(frame.shape[0], int(y2 * scale_y) + self.glare_pad)
        if fx2 <= fx1 or fy2 <= fy1:
            return 0.0
        crop = frame[fy1:fy2, fx1:fx2]
        v = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 2]
        return float((v > self.glare_v_thresh).mean())

    def _apply_morphology(self, E: np.ndarray, T: float) -> np.ndarray:
        mask = (E > T).astype(np.uint8)
        if self.k_close is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        if self.k_open is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)
        return mask

    def _is_valid_contour(self, contour: np.ndarray, E: np.ndarray, z: np.ndarray, T: float, H: int, W: int, e_peak: float) -> Tuple[bool, int, int, int, int, int]:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < self.min_area or area > self.max_area_frac * H * W or min(w, h) < self.min_side:
            return False, x, y, w, h, area
        if E[y:y + h, x:x + w].max() < self.core_frac * e_peak:
            return False, x, y, w, h, area
        if max(w / h, h / w) > self.max_aspect:
            return False, x, y, w, h, area
        if cv2.contourArea(contour) / area < self.min_rect:
            return False, x, y, w, h, area
        if float((z[y:y + h, x:x + w] > T).mean()) < self.min_density:
            return False, x, y, w, h, area
        return True, x, y, w, h, area

    def _refine_bbox(self, z: np.ndarray, mask: np.ndarray, x: int, y: int, w: int, h: int, T: float) -> List[int]:
        x1, y1, x2, y2 = x, y, x + w, y + h
        ys, xs = np.nonzero((z[y:y + h, x:x + w] > T) & (mask[y:y + h, x:x + w] > 0))
        if xs.size >= 8:
            x1 = x + int(np.floor(np.percentile(xs, 3)))
            x2 = x + int(np.ceil(np.percentile(xs, 97))) + 1
            y1 = y + int(np.floor(np.percentile(ys, 3)))
            y2 = y + int(np.ceil(np.percentile(ys, 97))) + 1
        return [x1, y1, x2, y2]

    def _nms(self, cands: List[Tuple[int, List[int]]]) -> List[List[int]]:
        cands.sort(key=lambda t: -t[0])
        kept = []
        for _, b in cands:
            dup = False
            for k in kept:
                iw = max(0, min(b[2], k[2]) - max(b[0], k[0]))
                ih = max(0, min(b[3], k[3]) - max(b[1], k[1]))
                inter = iw * ih
                A = (b[2] - b[0]) * (b[3] - b[1])
                B = (k[2] - k[0]) * (k[3] - k[1])
                if inter / (A + B - inter) > self.nms_iou or inter / min(A, B) > 0.6:
                    dup = True
                    break
            if not dup:
                kept.append(b)
        return kept

    def extract(self, z_map: np.ndarray, stride: int = 1, frame: Optional[np.ndarray] = None) -> List[List[int]]:
        z = np.nan_to_num(np.asarray(z_map, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if z.ndim != 2:
            raise ValueError("z_map must be a 2-D array (H x W)")
        z = np.maximum(z, 0.0)
        H, W = z.shape

        E = cv2.boxFilter(z, -1, (self.smooth_k, self.smooth_k), borderType=cv2.BORDER_REPLICATE)
        e_peak = float(E.max())
        if e_peak < self.floor:
            return []

        cands = []
        for a in self.levels:
            T = max(self.floor, a * e_peak)
            mask = self._apply_morphology(E, T)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                valid, x, y, w, h, area = self._is_valid_contour(c, E, z, T, H, W, e_peak)
                if not valid:
                    continue

                bbox = self._refine_bbox(z, mask, x, y, w, h, T) if self.refine else [x, y, x + w, y + h]
                cands.append((area, bbox))

        kept = self._nms(cands)

        if frame is not None:
            scale_x = frame.shape[1] / W
            scale_y = frame.shape[0] / H
            kept = [b for b in kept if self._blown_highlight_fraction(frame, b, scale_x, scale_y) <= self.max_glare_frac]

        return [[int(v * stride) for v in b] for b in kept]
