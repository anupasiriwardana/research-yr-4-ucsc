# Multi-Image Patch Detection Metrics

This directory contains the workflow for:

1. Applying a saved adversarial patch to multiple clean images.
2. Saving the ground-truth coordinates of every rendered patch.
3. Running the patch detector on the generated images.
4. Saving the detector's predicted patch coordinates and latency.
5. Comparing ground-truth and predicted coordinates to generate object-level and image-level metrics.

The scripts use a shared **640 x 640 pixel coordinate system**. The paths used by
the scripts are controlled by:

- [`gen-patched-img/config.json`](./gen-patched-img/config.json)
- [`det-patch-mech1-updated/detection_config.json`](./det-patch-mech1-updated/detection_config.json)
- [`calc-metrics/metrics_config.json`](./calc-metrics/metrics_config.json)

## Directory layout

```text
test-patch-det-multi-img/
├── gen-patched-img/
│   ├── apply_saved_multi_patch_batch.py
│   ├── config.json
│   ├── patches/
│   ├── yolov8-patched-imgs/              # generated patched images
│   └── yolov8-patch_coordinates/         # ground-truth patch boxes
├── det-patch-mech1-updated/
│   ├── detect_cls_head_multi_img.py
│   ├── detection_config.json
│   └── yolov8-detected-patch-coordinates/ # detector output boxes
├── calc-metrics/
│   ├── compare_patch_detection_metrics.py
│   ├── metrics_config.json
│   └── patch-detection-metrics/          # JSON and CSV metric reports
├── yolov8n.pt
└── readme.md
```

The exact directories can be changed in the JSON configuration files. If a
configuration file is changed, update all downstream configuration values that
refer to the changed output directory.

## Requirements

- Python 3.10 or a compatible Python version supported by the installed
  PyTorch, OpenCV, NumPy, and Ultralytics packages.
- PyTorch.
- OpenCV (`cv2`).
- NumPy.
- Ultralytics.
- A YOLOv8 model file, configured as `model_path`.
- The detector calibration profile configured by
  `calibration_profile_filename`.

Use the Python environment containing the project's dependencies. On Windows,
activate that environment before running the commands below, or replace
`python` with the full path to its Python executable.

## Complete workflow

Run the commands from this directory:

```powershell
cd D:\GitHub\yr-4-research\test-patch-det-multi-img
```

### 1. Generate patched images and ground-truth coordinates

```powershell
python .\gen-patched-img\apply_saved_multi_patch_batch.py
```

This script:

- Reads clean images from `clean_dir`.
- Loads `specific_patch_name` from `patch_dir`.
- Finds up to `multi_max_patches` target objects per image.
- Applies the saved patch to each selected target.
- Writes patched images to `batch_settings.output_dir`.
- Writes one coordinate file per source image to
  `batch_settings.coordinates_dir`.

Each coordinate file contains one line per rendered patch, for example:

```text
box=(269, 409, 365, 505)
box=(104, 272, 192, 352)
```

If no target is found in an image, an empty coordinate file is still created.
This is intentional: it allows the metrics stage to distinguish clean images
with no detections from missing files.

To process a different batch range without editing the configuration:

```powershell
python .\gen-patched-img\apply_saved_multi_patch_batch.py --num-images 100 --start-index 0
```

### 2. Run the detector

```powershell
python .\det-patch-mech1-updated\detect_cls_head_multi_img.py
```

This script reads the generated patched images from
`detection_config.json`, runs the detector, and writes one text file per image
to `batch_settings.coordinates_dir`.

Each detection file can contain metadata and zero or more predicted boxes:

```text
latency_ms=63.066
is_attack=true
score=37.197308
box=(104, 272, 192, 352) score=37.197308
```

The `latency_ms` value measures the feature-map detection stage: classification
feature-map extraction, anomaly scoring, connected-component detection, and
bounding-box extraction. It starts immediately before the feature-map
extraction call and excludes image loading, preprocessing, the separate YOLO
prediction, and visualization. A file may contain latency and no `box=` lines
when the detector does not produce a patch detection.

The detector also writes visualization images when
`save_visualizations` is enabled.

To override the batch range:

```powershell
python .\det-patch-mech1-updated\detect_cls_head_multi_img.py --num-images 100 --start-index 0
```

### 3. Generate the metrics reports

```powershell
python .\calc-metrics\compare_patch_detection_metrics.py
```

To use another metrics configuration file:

```powershell
python .\calc-metrics\compare_patch_detection_metrics.py `
  --config .\calc-metrics\metrics_config.json
```

The script writes:

- `calc-metrics/patch-detection-metrics/patch_detection_metrics.json`
  - Dataset summary, matching rules, aggregate metrics, latency statistics,
    and per-image results.
- `calc-metrics/patch-detection-metrics/patch_detection_per_image.csv`
  - One row per image for spreadsheet analysis.

The metrics script requires matching ground-truth and detection files by image
key. With `strict_file_matching: true`, it stops with an error if a counterpart
file is missing rather than silently treating the missing file as empty.

## Metric definitions

The workflow reports two different kinds of metrics: **object-level** metrics
for individual patch boxes and **image-level** metrics for whether an image
contains at least one patch detection.

### Object-level terms

For each image:

- **Ground-truth box**: a box generated by the patching script from the visible
  rendered patch mask. It represents one actual patch instance.
- **Detection box**: a box produced by the detector for one predicted patch
  region.
- **Object true positive (TP)**: one detected box successfully matched to one
  ground-truth patch box under the overlap rule described below.
- **Object false positive (FP)**: a detected box that cannot be matched to any
  still-unmatched ground-truth box.
- **Object false negative (FN)**: a ground-truth patch box that cannot be
  matched to any still-unmatched detection box.
- **True negative (TN)**: not reported for object-level boxes. There is no
  well-defined finite set of negative patch boxes in an image, so object-level
  accuracy and object-level TN counts are intentionally omitted.

### Object-level formulas

Let `TP`, `FP`, and `FN` be the totals over all compared images:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

If a denominator is zero, the implementation reports `0.0` for that ratio
instead of raising a division-by-zero error.

- **Precision** measures how many predicted patch boxes were valid:
  `TP / (TP + FP)`.
- **Recall** measures how many actual patch boxes were detected:
  `TP / (TP + FN)`.
- **F1 score** is the harmonic mean of precision and recall. It is high only
  when both precision and recall are high.

### Image-level terms

At image level, an image is considered positive if it contains at least one
ground-truth box or at least one detected box:

- **Image true positive (TP)**: the image contains at least one ground-truth
  patch and at least one detection box.
- **Image false negative (FN)**: the image contains at least one ground-truth
  patch but contains no detection boxes.
- **Image false positive (FP)**: the image contains no ground-truth patches but
  contains one or more detection boxes.
- **Image true negative (TN)**: the image contains no ground-truth patches and
  contains no detection boxes.

Important: image-level TP only means that the detector produced at least one
box in a patched image. It does **not** guarantee that every patch in that
image was detected, and it does not remove object-level false positives. Those
cases are represented by object-level FP and FN counts.

The image-level formulas use the same precision, recall, and F1 definitions:

```text
image_precision = image_TP / (image_TP + image_FP)
image_recall    = image_TP / (image_TP + image_FN)
image_F1        = 2 * image_precision * image_recall /
                  (image_precision + image_recall)
```

The report additionally calculates the clean-image false-positive rate:

```text
false_positive_rate = image_FP / (image_FP + image_TN)
```

This measures how often clean images incorrectly receive one or more patch
detections.

### Latency

- **Images with latency**: number of detection files containing a valid
  non-negative `latency_ms` value.
- **Average latency**: arithmetic mean of the valid per-image latency values.
- **Minimum latency**: smallest valid latency.
- **Maximum latency**: largest valid latency.

Missing latency values are not converted to zero. They are excluded from the
latency statistics and reflected by `images_with_latency`.

## Matching rules and assumptions

The metric results depend on the following rules:

1. **Coordinate space**: ground-truth and detected boxes are compared in the
   same 640 x 640 pixel coordinate space.
2. **Positive-area overlap**: a detection is a true positive when its bounding
   box and an unmatched ground-truth patch box have an intersection area
   greater than zero.
3. **Partial coverage counts as detection**: a detected box covering only a
   portion of the actual patch box is still a true positive if the boxes
   overlap with positive area. No minimum IoU threshold is currently required.
4. **Boxes inside the actual patch are accepted**: a smaller detection entirely
   inside a ground-truth patch can be a true positive.
5. **Boxes larger than the actual patch are accepted**: a detection surrounding
   the ground-truth patch can be a true positive.
6. **Shifted boxes are accepted**: a detection does not need identical
   coordinates; any positive-area overlap is sufficient.
7. **Edge and corner touching do not count**: boxes that only share an edge or
   corner have zero intersection area and are not matched.
8. **One-to-one matching**: each ground-truth box and each detection box can be
   used in at most one match.
9. **Ambiguous overlaps use highest IoU first**: all overlapping candidate pairs
   are sorted by descending Intersection over Union (IoU), and the best
   available pair is matched first. IoU is used to prioritize matches only; it
   is not an acceptance threshold.
10. **Extra detections are false positives**: after one-to-one matching, every
    unmatched detection box increments object-level FP.
11. **Missed patches are false negatives**: after one-to-one matching, every
    unmatched ground-truth box increments object-level FN.
12. **Invalid boxes are rejected**: a box with zero or negative width or height
    is invalid and causes the metrics script to report an error.
13. **Coordinate ordering is normalized**: reversed corner coordinates are
    reordered before area and overlap calculations.
14. **Filename matching is explicit**: detection files beginning with
    `multi-patched_` are matched to ground-truth files after that prefix is
    removed.
15. **Strict file matching is enabled**: every ground-truth coordinate file is
    expected to have a corresponding detection file and vice versa.

### Answer to the partial-overlap question

**Yes. Under the current implementation, a detected patch box that covers only
part of the actual patch box is counted as an object-level true positive, as
long as the shared area is greater than zero.**

For example:

```text
Ground truth: (100, 100, 200, 200)
Detection:    (150, 150, 175, 175)
```

The detection is inside the actual patch and has positive-area overlap, so it
is matched as a TP. This does not mean the localization is geometrically
accurate; it means the current evaluation is measuring whether the detector
identified a region that intersects the patch. The per-image report still
records the matched IoU for diagnostic purposes.

By contrast:

```text
Ground truth: (100, 100, 200, 200)
Detection:    (200, 100, 250, 200)
```

These boxes only touch at the left/right edge. Their intersection area is zero,
so they are not matched; the detection is an FP and the ground-truth box is an
FN unless another detection can match it.

## Interpreting the reports

The aggregate JSON contains:

- `coordinate_matching`: the matching criterion and coordinate-space
  assumptions used for the run.
- `dataset`: compared image count and missing-file diagnostics.
- `object_metrics`: box-level TP, FP, FN, precision, recall, and F1.
- `image_metrics`: image-level TP, FP, FN, TN, precision, recall, F1, and clean
  image false-positive rate.
- `latency`: valid latency count and summary statistics.
- `per_image`: box counts, per-image TP/FP/FN, image status, latency, and matched
  IoUs.

For patch-detection evaluation, object-level recall is the main measure of how
many inserted patches were found, while object-level precision indicates how
many reported boxes corresponded to inserted patches. Image-level metrics
answer the separate question of whether an image was flagged at all.

## Re-running only the comparison

If the ground-truth and detection coordinate files already exist, there is no
need to rerun patch application or detection. Run only:

```powershell
python .\calc-metrics\compare_patch_detection_metrics.py
```

This regenerates both reports from the current coordinate and detection files.
