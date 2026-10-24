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
        "smooth_k": 5,
        "levels": [0.65, 0.55, 0.45],
        "core_frac": 0.70,
        "close_k": 3,
        "open_k": 5,
        "min_area": 100,
        "max_area_frac": 0.05,
        "max_aspect": 2.2,
        "min_rect": 0.60,
        "min_density": 0.55,
        "nms_iou": 0.30,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        self.smooth_k = int(cfg["smooth_k"])
        self.levels = tuple(cfg["levels"])
        self.core_frac = float(cfg["core_frac"])
        self.close_k = int(cfg["close_k"])
        self.open_k = int(cfg["open_k"])
        self.min_area = int(cfg["min_area"])
        self.max_area_frac = float(cfg["max_area_frac"])
        self.max_aspect = float(cfg["max_aspect"])
        self.min_rect = float(cfg["min_rect"])
        self.min_density = float(cfg["min_density"])
        self.nms_iou = float(cfg["nms_iou"])

        # Pre-compile morphological kernels so they aren't rebuilt in every loop iteration
        self.k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (self.close_k, self.close_k)) if self.close_k > 1 else None
        self.k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (self.open_k, self.open_k)) if self.open_k > 1 else None

    def _apply_morphology(self, E: np.ndarray, T: float) -> np.ndarray:
        mask = (E > T).astype(np.uint8)
        if self.k_close is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        if self.k_open is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)
        return mask

    def _is_valid_contour(
        self,
        contour: np.ndarray,
        E: np.ndarray,
        T: float,
        H: int,
        W: int,
        e_peak: float,
        e_bg: float,
        stride: int = 1,
    ) -> Tuple[bool, int, int, int, int, int]:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # 1. Size (Area) - Stride-invariant image pixel area check (Bug 4 fix)
        pixel_area = area * (stride ** 2)
        max_feat_area = self.max_area_frac * H * W
        if pixel_area < self.min_area or area > max_feat_area:
            return False, x, y, w, h, area

        # 2. Core Energy (anchored above background baseline e_bg)
        core_thresh = e_bg + self.core_frac * (e_peak - e_bg)
        if E[y:y + h, x:x + w].max() < core_thresh:
            return False, x, y, w, h, area

        # 3. Aspect Ratio
        if max(w / h, h / w) > self.max_aspect:
            return False, x, y, w, h, area

        # 4. Rectangularity (Fill Factor)
        if cv2.contourArea(contour) / area < self.min_rect:
            return False, x, y, w, h, area

        # 5. Smoothed Energy Density (Bug 2 fix: E > T instead of z > T)
        if float((E[y:y + h, x:x + w] > T).mean()) < self.min_density:
            return False, x, y, w, h, area

        return True, x, y, w, h, area

    def _refine_bbox(self, E: np.ndarray, mask: np.ndarray, x: int, y: int, w: int, h: int, T: float) -> List[int]:
        x1, y1, x2, y2 = x, y, x + w, y + h
        # Bug 2 fix: Use E > T for consistent coordinate refinement
        ys, xs = np.nonzero((E[y:y + h, x:x + w] > T) & (mask[y:y + h, x:x + w] > 0))
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
        e_peak = float(np.percentile(E, 99.9))

        # Upper background baseline derived directly from max_area_frac (e.g., 95th or 90th percentile)
        bg_percentile = 100.0 * (1.0 - min(max(self.max_area_frac, 0.01), 0.25))
        e_bg = float(np.percentile(E, bg_percentile))


        cands = []
        for a in self.levels:
            # Method 1: Interpolate threshold between upper background (e_bg) and peak (e_peak)
            T = a * (e_peak - e_bg)
            mask = self._apply_morphology(E, T)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                valid, x, y, w, h, area = self._is_valid_contour(
                    c, E, T, H, W, e_peak, e_bg, stride=stride
                )
                if not valid:
                    continue

                bbox = self._refine_bbox(E, mask, x, y, w, h, T)
                cands.append((area, bbox))

        kept = self._nms(cands)

        return [[int(v * stride) for v in b] for b in kept]