# Mechanism 1, Option B: Tightening the Detected Patch's Bounding Box

## Status
Documents the oversized-bounding-box problem observed once Option B (`mech-1-optionB.md`) successfully started detecting hidden-object attacks, the approach that was tried and rejected, and the moment-based solution actually implemented in `cls_head_detector_tight.py`. Companion to `mech-1-optionB.md` and `README.md`.

---

## 1. The Problem: The Detected Box Is Noticeably Larger Than the Actual Patch

Once Option B's minimum-Mahalanobis scoring correctly flagged a successfully-hidden object, the red bounding box drawn around the detected anomaly was consistently larger than the patch itself — a visible gap between the box edge and the patch's actual boundary. This matters beyond cosmetics: this box is meant to drive **recovery** (masking/re-inference), and an oversized box risks unnecessarily covering neighboring real objects during that masking step.

### Why it happens

The classification-branch layer being tapped (`cv3[-2]`) sits several convolutional layers deep — through the backbone, into the neck, and partway through the classification branch itself. Each layer expands the **effective receptive field**: a grid cell's activation isn't determined only by its nominal 8×8-pixel stride tile, it's influenced by a much wider region of the input. Cells whose centers sit outside the patch, but whose receptive field partially overlaps it, still pick up anomalous signal — producing a soft halo around the true patch rather than a hard edge. This is the same general property that makes CNN activation-based localization methods (Class Activation Mapping, Grad-CAM) inherently coarser than the true object boundary — it is not specific to this pipeline.

Two smaller, additive contributors:
- **Grid quantization**: each cell covers an 8×8 pixel tile, so any partial overlap rounds up to the full tile.
- **Rotation** (if `apply_settings.rotation` was non-zero when the patch was applied): the axis-aligned bounding box of a rotated shape is always larger than the shape itself.

---

## 2. First Attempt (Rejected): Stricter Threshold + Morphological Erosion

The first fix tried a stricter "core" threshold specifically for box extraction (decoupled from the detection threshold, so tightening the box wouldn't cost detection sensitivity), followed by morphological erosion on the resulting mask:

```python
box_multiplier = self.config["detector_settings"].get("box_core_threshold_multiplier", 1.3)
erosion_cells = self.config["detector_settings"].get("box_erosion_cells", 1)

core_mask = (scores > self.threshold * box_multiplier)
core_mask_np = core_mask.cpu().numpy().astype(np.uint8) * 255

if erosion_cells > 0:
    kernel = np.ones((erosion_cells * 2 + 1, erosion_cells * 2 + 1), np.uint8)
    core_mask_np = cv2.erode(core_mask_np, kernel, iterations=1)
```

### Why it didn't work well

This operates on whatever *shape* the raw thresholded blob happens to have. If that blob is already irregular or off-center — which is common, since the halo isn't a clean circle; it's shaped by the patch's own texture, nearby objects, and edge effects — erosion just uniformly peels pixels off that same lopsided shape. It can shrink asymmetry, but it can't correct it, which is exactly the observed failure mode: the box sometimes shrank only from one side. A single global threshold multiplier also can't adapt to different blob sizes across different patches, producing inconsistent over- and under-shrinking across test images. This approach was tried, found unreliable, and replaced — documented here rather than silently dropped, since the reasoning is relevant to anyone tuning this further.

---

## 3. The Chosen Fix: Moment-Based Box (Weighted Centroid and Spread)

### The idea
Instead of eroding the raw pixel mask, treat the anomaly scores within the detected connected component as a weighted point cloud, and derive the box from its **statistical moments** — weighted centroid (first moment) and weighted spread (second moment) — the same underlying idea as fitting a blob via image moments in classical computer vision (e.g., `cv2.fitEllipse`-style moment fitting), applied here to the score distribution rather than a binary mask.

### Why this fixes the asymmetry problem specifically
The resulting box is `[cx - half_w, cx + half_w] × [cy - half_h, cy + half_h]` — **symmetric around the computed centroid by construction**. There is no code path by which this can shrink from only one side; the box's shape is derived from where the score *mass* actually sits, not from which discrete pixels survive an erosion kernel. The spread (`std_x`, `std_y`) is also computed fresh per detection, so it naturally scales with each blob's own size rather than applying one fixed erosion-cell count to every patch regardless of size.

### Code

```python
if is_attack:
    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Isolate just this connected component, so unrelated anomalous
        # cells elsewhere in the image don't bias the moment calculation.
        component_mask = np.zeros_like(mask_np)
        cv2.drawContours(component_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        component_bool = torch.from_numpy(component_mask > 0).to(scores.device)

        # Weight each cell by how far ABOVE threshold its score is (its
        # "excess" anomaly), then take the weighted centroid and spread.
        ys, xs = torch.where(component_bool)
        weights = (scores[ys, xs] - self.threshold).clamp(min=0) + 1e-6

        cx = (xs.float() * weights).sum() / weights.sum()
        cy = (ys.float() * weights).sum() / weights.sum()
        std_x = ((xs.float() - cx) ** 2 * weights).sum().div(weights.sum()).sqrt()
        std_y = ((ys.float() - cy) ** 2 * weights).sum().div(weights.sum()).sqrt()

        spread_multiplier = self.config["detector_settings"].get("box_spread_multiplier", 1.5)
        half_w = max(std_x.item() * spread_multiplier, 1.0)
        half_h = max(std_y.item() * spread_multiplier, 1.0)

        x1 = max(0, int(round(cx.item() - half_w)))
        x2 = min(W, int(round(cx.item() + half_w)))
        y1 = max(0, int(round(cy.item() - half_h)))
        y2 = min(H, int(round(cy.item() + half_h)))

        bounding_box = (x1 * self.stride, y1 * self.stride, x2 * self.stride, y2 * self.stride)
```

### Tuning `box_spread_multiplier`
Starts at `1.5` (roughly 1.5 standard deviations of score mass on each side). Raise it if the box is consistently too tight (risking under-covering the patch, the more dangerous failure for a recovery-masking use case); lower it toward `1.0` if it's still too loose. Because the spread is now computed from the blob's own statistics, small adjustments to this single multiplier should behave more predictably across differently-sized patches than the previous integer-step erosion count did.

---

## 4. A Path Explicitly Not Taken: GrabCut / Pixel-Level Segmentation Refinement

A further tightening option was considered — using this box as a seed for OpenCV's GrabCut segmentation, letting it snap to the patch's actual color/texture edge in the raw pixel image rather than relying only on the coarse 8px-stride activation grid. This was **deliberately not adopted**, for a reason worth stating explicitly rather than leaving implicit: it would mean the bounding-box step no longer derives entirely from the model's internal activations, breaking the core methodological premise of this whole detection approach (that patch detection and localization happen via behavioral monitoring of internal signals, not classical external image processing on the raw pixel image). The moment-based approach above stays entirely within activation space, consistent with that premise, at the cost of being bounded by the tapped layer's inherent receptive-field resolution rather than achieving pixel-perfect edges.

---

## 5. Relationship to `mech-1-optionB.md`

This tightening is a change to **box extraction only** — it sits entirely downstream of Option B's detection scoring (`_min_mahalanobis_scores`, the `is_attack` decision, and the threshold) and does not alter any of it. `cls_head_detector_tight.py` and `cls_head_detector.py` share identical detection logic; they differ only in how the final `bounding_box` is derived from the same `scores` tensor. See `README.md` for when to use which script.