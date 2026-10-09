# Mechanism 1, Option B: Box-Independent Anomaly Detection

## Current status

Mechanism 1 has been updated from the original box-gated classifier to an
Option B detector that scores the complete P3 classification feature map.
The detector no longer depends on YOLO retaining a detection box for the
object attacked by a hiding patch.

The current implementation is split into two compatible runtime variants:

- [`cls_head_detector.py`](./cls_head_detector.py) uses minimum Mahalanobis
  scoring and reports every sufficiently large connected anomalous region.
- [`cls_head_detector_for_tightPatch.py`](./cls_head_detector_for_tightPatch.py)
  uses the same scoring, but estimates one anomaly box from the weighted
  centroid and spread of the strongest connected component.

The shared calibration and feature-hook code are in
[`calibrate_cls_head.py`](./calibrate_cls_head.py) and
[`cls_head_adapter.py`](./cls_head_adapter.py). The module is configured by
[`config.json`](./config.json).

## Why the original detector failed

The original detector first converted YOLO's own predictions into a
`predicted_classes` grid, then calculated a class-conditional Mahalanobis
distance only inside those predicted boxes. Cells without a predicted box
were left at a score of zero.

That creates a structural blind spot: when a patch successfully hides an
object, YOLO can remove the object's box, and the mechanism consequently
stops scoring the exact region where the patch is located. This is not
corrected by changing the threshold; the detector is using the attacked
output as its scoring gate.

## Current architecture

```text
Input image
    |
    v
Resize to 640 x 640 and run YOLOv8n
    |
    +--> YOLO detections (visual comparison only)
    |
    v
Forward hook on YOLOv8 detect.cv3[P3][-2]
    |
    v
P3 classification activation map, stride 8
    |
    +--> Offline calibration:
    |       per-object-class statistics
    |       + bounded background statistics
    |
    v
Every grid cell compared with every calibrated distribution
    |
    v
Minimum Mahalanobis distance per cell
    |
    v
Threshold mask -> connected-region filtering -> anomaly boxes
```

The hook remains on the classification branch (`cv3`), not the
box-regression branch (`cv2`). This preserves the original decoupled-head
design: the mechanism monitors classification features rather than using
YOLO's regression output as its detector.

## Calibration

[`calibrate_cls_head.py`](./calibrate_cls_head.py) runs YOLO on clean images
and collects the P3 feature vector for every grid cell inside each detected
object box. It also samples up to 50 uncovered cells per image as a
reserved background class (`-1`). Capping the background sample prevents
ordinary background cells from overwhelming the object-class samples and
keeps calibration memory bounded.

For each class with more than `min_calibration_samples` vectors, calibration
saves:

- the feature mean;
- the inverse covariance matrix;
- the class identifier, including the background identifier.

The resulting profile is written to:

```text
profiles/cls_head_calibration_p3_mech1_updated.pkl
```

This profile is only compatible with the current Option B scoring path. It
must be regenerated when the protected model, clean calibration data, or
feature tap changes.

## Runtime scoring

For a feature vector `x`, the current score is:

```text
score(x) = min over calibrated classes c of
           sqrt((x - mean_c)^T inverse_covariance_c (x - mean_c))
```

Every P3 cell is scored against every calibrated object class and the
background class. The minimum is used because the detector does not assume
that YOLO's current class prediction is trustworthy.

The runtime implementation stacks all means and inverse covariance matrices
once during initialization, then evaluates each class over the complete
flattened feature map. `results.boxes` is retained only to draw YOLO's
current detections on the optional visualization; it is not used to decide
which cells are scored.

## Region extraction and outputs

The base detector thresholds the score map and uses
`connectedComponentsWithStats` to produce one result for each connected
region. Regions smaller than `min_region_cells` are discarded. The current
configuration requires at least six feature-map cells.

The returned result has this shape:

```python
{
    "is_attack": bool,
    "score": float,  # highest surviving region score
    "bounding_boxes": [
        {
            "box": (x1, y1, x2, y2),  # input-image pixel coordinates
            "score": float,
        }
    ],
}
```

The visualization contains:

- the score heatmap;
- YOLO's current boxes in green, for comparison only;
- anomaly boxes in red;
- one red annotation per surviving region.

The tight-box variant keeps the same score map, threshold, and
connected-component logic. Its box extraction uses score excess above the
threshold as weights, computes a weighted centroid and weighted standard
deviation, and expands around the centroid by
`box_spread_multiplier`. This reduces the receptive-field halo and produces
a box that is symmetric around the detected score mass. It is the preferred
variant when the box will drive recovery or masking.

## Current configuration

The checked-in [`config.json`](./config.json) currently specifies:

| Setting | Current value |
|---|---|
| Model | `yolov8n.pt` |
| Input size | `640 x 640` |
| Feature scale | `P3` |
| Feature stride | `8` |
| Score threshold | `25.0` |
| Minimum calibration samples | `100` |
| Background class id | `-1` |
| Background samples per image | `50` |
| Tight-box spread multiplier | `1.5` |
| Minimum anomaly-region size | `6` cells |
| Default test image | `20-multi-patched_ccf42930-03bcfe59.jpg` |

The threshold is a configuration value, not a universal accuracy metric.
Because minimum-over-classes scoring differs from the former box-gated
scoring rule, it should be re-derived from a held-out clean-image score
distribution whenever the calibration profile or scoring rule changes.

## Current generated artifacts

The [`detections/`](./detections/) directory currently contains 53 rendered
outputs, including 18 clean-image outputs and 35 patched or multi-patched
outputs. The patched examples include single and multiple anomalies placed on
vehicles in traffic scenes, including night, highway, city, bridge, and
residential-road conditions. The multi-patch outputs demonstrate that the
base detector can return separate boxes instead of silently retaining only
the largest anomalous region.

The visual comparison files are evidence that the anomaly signal can remain
localized after YOLO's own object box is weakened or removed. They should be
read as qualitative validation artifacts: this folder does not currently
contain a clean-image false-positive distribution, precision/recall table,
or a threshold-selection report.

## How to run

From the module directory, after installing PyTorch, Ultralytics, OpenCV,
NumPy, and the configured YOLO environment:

```bash
# Regenerate the Option B calibration profile.
python calibrate_cls_head.py

# Run base Option B scoring.
python cls_head_detector.py

# Run Option B with the tighter moment-based box.
python cls_head_detector_for_tightPatch.py
```

Both runtime scripts use CUDA automatically when available and otherwise
fall back to CPU. Each runtime script raises an explicit
`FileNotFoundError` if its calibration profile is missing or its input
image cannot be read.

## Validation interpretation

For a successful hiding attack, the strongest direct check is an overlay in
which YOLO's green box is absent or weakened over the attacked object while
the Option B heatmap and red anomaly box remain active in that region. This
tests the actual requirement of the redesign: scoring must continue even
when the attacked detector no longer supplies the box.

The next quantitative validation still required before reporting detection
performance is a held-out clean-image run to derive the threshold and
measure false positives, followed by a patched-image run reporting
per-image detection and localization results.

## Reference

The minimum Mahalanobis-distance formulation follows the class-conditional
anomaly/OOD scoring approach described in:

> Lee, Lee, Lee, and Shin, “A Simple Unified Framework for Detecting
> Out-of-Distribution Samples and Adversarial Attacks,” NeurIPS 2018.
