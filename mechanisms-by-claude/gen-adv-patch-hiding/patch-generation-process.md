# Patch-generation process

## Status

This document describes the current patch-hiding pipeline in
`gen-adv-patch-hiding`. The active configuration uses `yolo11n.pt`, the V2
generator, and multi-object application. `generate_hiding_patch.py` and
`apply_saved_patch.py` remain in the directory as earlier single-patch
variants; they should not be treated as interchangeable with the V2/multi
workflow.

The pipeline is a white-box attack against the configured Ultralytics YOLO
model. It is intended to suppress detections of an existing target object,
not merely to create arbitrary detections elsewhere in an image.

## 1. Problems that motivated the redesign

### 1.1 The original loss was not detector-aware

The first implementation used a whole-image aggregate similar to:

```python
class_probs = preds[:, :, 4:]
max_class_probs, _ = torch.max(class_probs, dim=-1)
loss = torch.mean(max_class_probs)
```

This objective did not identify a real object or its location. The optimizer
could therefore satisfy the loss by producing unrelated detections
elsewhere, while leaving the intended vehicle or other target detectable.

### 1.2 Placement was not tied to the target

Earlier versions sampled a position independently of the detected object, and
the demo used a fixed or canvas-centre position. A patch trained mostly away
from its target does not receive a reliable gradient for hiding that target.
The current generator derives both the training placement and the demo
placement from a clean detection.

### 1.3 Operational problems

The earlier pipeline also:

- loaded too much image data into memory at once;
- allowed `specific_clean_image` to accidentally become the training set;
- saved a raw array without a convenient standalone patch image;
- applied a saved patch through a compositor whose geometry differed from the
  training compositor; and
- had no distinct-object placement mode for applying several patches to one
  image.

## 2. Current design

### 2.1 Spatially targeted pseudo-label loss

`generate_hiding_patch-V2.py` performs one clean forward pass per sampled
image. `select_target_object`:

1. computes the highest class confidence for each decoded prediction;
2. keeps predictions above `conf_thresh`, optionally restricted to
   `target_class`;
3. selects the highest-confidence candidate;
4. uses that candidate's box to select all confident prediction centres inside
   the box; and
5. returns the selected anchors and the object's normalized centre.

The patched image is then evaluated and the loss is the mean maximum class
confidence **only at those selected anchors**:

```python
patched_preds = get_raw_preds(yolo_model.model, patched_image.unsqueeze(0))
max_probs, _ = patched_preds[0, :, 4:].max(dim=-1)
loss = max_probs[target_mask].mean()
```

This is pseudo-labeling from the model's clean prediction; the dataset does
not provide ground-truth boxes. YOLOv8/YOLO11's decoded output has box values
followed by per-class confidence values and no separate objectness channel,
so the implementation suppresses class confidence at the target anchors
rather than a classic YOLO objectness score.

### 2.2 Object-centred EoT placement

For every valid training image, the patch is:

- randomly rotated in `[-rotation_max, rotation_max]`;
- randomly scaled in `[scale_min, scale_max]`; and
- placed near the detected target centre with `placement_jitter`.

The V2 compositor uses an affine grid with translation terms corrected for
both scale and rotation. This matters because the earlier translation formula
caused placements to collapse toward the canvas centre when small scales were
used. Patches trained with the earlier compositor should be considered
incompatible with the corrected geometry and retrained.

At demo/application time, the same clean-detection convention is used:

- `auto` places one patch on the highest-confidence target;
- `manual` converts `manual_x`/`manual_y` from the 640×640 working space; and
- `multi` finds several candidates in one clean forward pass, orders them by
  confidence, and rejects centres closer than `multi_min_separation`.

The `multi` mode is implemented in `apply_saved_multi_patch.py`. It is not
equivalent to repeatedly calling `auto`, which would select the same highest
confidence object and stack patches on it.

### 2.3 Disk-streamed training

The generator indexes filenames only:

```python
train_files = discover_image_files(DATA_DIR, max_images)
```

Images are read and resized to 640×640 only when sampled. At most
`gpu_batch_size` images are held for a training step, and CUDA's cache is
released after the step. `max_images` therefore limits the candidate filename
pool rather than allocating a tensor for the entire dataset.

### 2.4 `specific_clean_image` semantics

`specific_clean_image` selects the image used for the generator's final demo
composite and by the application scripts. It does **not** restrict the
training pool. When it is empty, the scripts fall back to the first available
image.

### 2.5 Saved artifacts

The V2 generator saves:

- `art_patch_v11.npy` in `patch_dir`, for reuse by the application scripts;
- `art_patch_standalone_v8.png`, the raw optimized patch as an image; and
- `art_patched_<image-name>.jpg` in `output_dir`, a single-patch demo
  composite.

The current checked-in configuration points `specific_patch_name` at
`art_patch_v11(0.10-0.20).npy`; that name must match an actual file before
running an application script. The repository currently also contains older
V6/V7/V8 patch arrays and standalone images; their filenames identify their
generation, but do not by themselves guarantee compatibility with the V2
compositor.

## 3. Current configuration

The checked-in `config.json` currently specifies:

| Setting | Current value | Meaning |
|---|---:|---|
| `model_path` | `yolo11n.pt` | Model used for training and inference |
| `patch_scale` | `0.15` | Demo/application scale fallback |
| `patch_size` | `150` | Raw square patch dimensions |
| `num_steps` | `1000` | Optimization iterations |
| `learning_rate` | `0.02` | Adam learning rate |
| `conf_thresh` | `0.25` | Clean-prediction confidence threshold |
| `rotation_max` | `22.5` | Maximum training rotation in degrees |
| `scale_min`, `scale_max` | `0.10`, `0.20` | Training scale range |
| `placement_jitter` | `0.1` | Normalized centre jitter |
| `target_class` | `2` | COCO class id for cars |
| `gpu_batch_size` | `20` | Images loaded per training step |
| `max_images` | `2000` | Candidate training-image count |
| `placement_mode` | `multi` | Apply separate patches to several targets |
| `multi_max_patches` | `3` | Maximum number of applied patches |
| `multi_min_separation` | `0.3` | Minimum normalized centre separation |

The application scripts read `patch_dir` for the saved array and
`output_dir` for composites. The efficacy verifier reads
`specific_patched_image`, pairs it with `specific_clean_image`, and writes a
side-by-side detector visualization to `patch_detection_dir`. In the current
configuration, `target_class` is nested under `patch_settings`, while the
application scripts use that value as their fallback target class.

## 4. Script responsibilities

### `generate_hiding_patch-V2.py`

This is the current generator. It freezes the configured model, optimizes
only the patch, streams images from disk, applies object-centred EoT, and
saves the V11 array, V8 standalone image, and one demo composite.

### `generate_hiding_patch.py`

This is an older single-patch generator. It has the same broad
object-aware/pseudo-label approach, but uses different output names and
should not be assumed to produce the patch named by the current
`config.json`.

### `apply_saved_multi_patch.py`

This is the current application path for the checked-in `placement_mode:
"multi"` configuration. It performs one clean detection pass, selects up to
`multi_max_patches` distinct targets, applies a separate corrected overlay to
each, and writes a `multi-patched` output.

### `apply_saved_patch.py`

This is the earlier one-patch application path. It supports `auto` and
`manual`, but its current overlay implementation is not identical to the
corrected V2/multi compositor. Use it only for legacy comparisons, or update
it before relying on it for V2 quantitative results.

### `verify_patch_efficacy.py`

This runs the configured YOLO model on the selected clean/patched pair,
prints all detections and confidences, and saves a side-by-side annotated
comparison. It is the required check that the intended target was suppressed,
rather than simply replaced by spurious detections.

## 5. Execution workflow

1. Confirm that `model_path`, `clean_dir`, `patch_dir`, and the patch filename
   in `specific_patch_name` are valid.
2. Generate the current patch:

   ```bash
   python generate_hiding_patch-V2.py
   ```

3. Set `specific_patch_name` to `art_patch_v11.npy` (the V2 generator's
   output), or rename/copy that file to the configured name
   `art_patch_v11(0.10-0.20).npy`. Apply it to the selected image:

   ```bash
   python apply_saved_multi_patch.py
   ```

4. Set `specific_patched_image` to the generated `21-multi-patched_...`
   output (or another selected patched image), then compare clean and patched
   detections:

   ```bash
   python verify_patch_efficacy.py
   ```

5. For a single target, use the legacy-compatible application script only
   after checking its compositor convention:

   ```bash
   python apply_saved_patch.py
   ```

6. Feed the resulting image into the downstream detection-mechanism
   experiments described by `mech-1-updated`.

The confidence trend printed by the generator is an optimization diagnostic,
not a substitute for the clean/patched comparison. A successful run should
show lower confidence for the intended target while preserving the rest of
the scene as much as possible.

## 6. Supporting literature

| Claim | Source |
|---|---|
| Objectness is a class-agnostic measure of whether an image window contains an object | Alexe, Deselaers & Ferrari, “Measuring the Objectness of Image Windows,” IEEE TPAMI, 2012 |
| Classic YOLO combines objectness and class probability | Redmon, Divvala, Girshick & Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” CVPR, 2016 |
| Classifier-style adversarial patches are spatially unaware | Brown, Mané, Roy, Abadi & Gilmer, “Adversarial Patch,” NeurIPS Workshop, 2017 |
| Detector-specific patch attacks motivate spatial targeting | Liu et al., “DPatch: An Adversarial Patch Attack on Object Detectors,” arXiv:1806.02299, 2018 |
| Detection confidence can be suppressed at a real target location | Thys, Van Ranst & Goedemé, “Fooling Automated Surveillance Cameras,” CVPR Workshops, 2019 |
| Expectation over Transformation improves robustness to placement variation | Athalye, Engstrom, Ilyas & Kwok, “Synthesizing Robust Adversarial Examples,” ICML, 2018 |

The pseudo-label selection, corrected compositor, disk streaming, and
multi-object separation are engineering decisions specific to this project.
