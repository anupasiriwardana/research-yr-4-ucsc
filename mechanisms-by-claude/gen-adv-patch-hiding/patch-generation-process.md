# Redesigning the Patch Generator: From a Classifier-Style Attack to a Practical, Object-Aware Pipeline

## Status
Documents the three problems that drove the patch generator away from its original ART-based design, and the current solution in `generate_hiding_patch.py` / `apply_saved_patch.py`, with supporting literature. Companion to the updated `README.md` for this module (setup and execution).

---

## 1. The Problems That Led to This Redesign

These were discovered in sequence, each fix exposing the next issue once the previous one was resolved.

### 1.1 Problem 1 — The attack loss was classifier-style, not detector-aware

The original script used ART's `AdversarialPatchPyTorch` with a custom loss:

```python
class_probs = preds[:, :, 4:]
max_class_probs, _ = torch.max(class_probs, dim=-1)
loss_total = torch.mean(max_class_probs)     # averaged across EVERY anchor in the image
```

This is structurally the same kind of objective as the original, classifier-only adversarial patch concept — "raise confidence of *something*, somewhere in the image" — with no notion of *where* the real target object actually is. In practice, this caused the patch to hallucinate spurious new detections elsewhere in the frame (persons, umbrellas) while barely affecting the real target's own confidence, since nothing in the loss singled out that location. Our supervisor's feedback specifically pointed at this: classic object detectors compute an **objectness score** — a class-agnostic "is there an object here" signal, separate from "what class is it" — and a proper detector-specific patch attack needs to suppress that score *at the target's location specifically*, not chase a whole-image aggregate. Full background on objectness and why classifier-style patches don't transfer to detectors is in Section 3.

### 1.2 Problem 2 — Patch placement was decoupled from the target object's location

Once the loss was rewritten to target specific anchors (Section 2.1), the patch still appeared at effectively random positions during both training and the demo application step, and in one case was itself recognized as an unrelated object ("vase"). The cause: the *loss* knew which anchors to suppress, but the *placement* logic sampled position uniformly across the whole canvas during training, and used a hardcoded canvas-center position for the demo — neither ever looked at where the actual target object was. A patch trained at random, mostly-irrelevant positions has little consistent gradient signal tying its pixels to "suppress the object when placed here," because it usually isn't placed anywhere near the object it's supposed to affect.

### 1.3 Problem 3 — Practical/operational issues

Three separate issues surfaced once the core attack was working:
- **GPU memory**: the original data-loading step pushed the *entire* configured image batch (up to thousands of images) to GPU memory upfront, which is roughly 9+ GB for 2,000 640×640 images — far beyond what a memory-constrained GPU (e.g. an RTX 2050) can hold, well before the model itself even runs. This caused `CUDA error: out of memory` crashes.
- **Config semantics**: `specific_clean_image`, when set, was silently shrinking the *entire training set* down to one image, rather than only selecting which image the finished patch gets demonstrated on.
- **Reusability**: the optimized patch was only ever saved as a raw `.npy` array and as one pre-composited demo image — there was no plain image file of the patch itself to paste manually elsewhere, and no reliable way to apply a saved patch to a *different* image later, since `apply_saved_patch.py` still used ART's own compositing function (`attack.apply_patch`), which does not use the same geometric convention as the training script's own placement logic. Applying a patch through a different function than the one it was optimized under silently changes its effective size/position, even when the code "looks" correct.

---

## 2. What Changed

### 2.1 Objectness-aware, spatially-targeted loss

Instead of a whole-image aggregate, the loss now targets only the anchors that correspond to a real, confidently-detected object in the *clean* (unpatched) image — used as a pseudo-label, since no ground-truth annotations are available for this dataset:

```python
def select_target_object(preds, conf_thresh=CONF_THRESH, target_class=None):
    """From a CLEAN forward pass, pick ONE object to attack (highest
    confidence, optionally restricted to target_class). Returns the
    anchors covering that object, its center (for placement), and its
    box. This is the pseudo-labeling step -- no annotations needed."""
    boxes_cxcywh = preds[0, :, 0:4]
    class_probs = preds[0, :, 4:]
    max_probs, max_classes = class_probs.max(dim=-1)

    candidate_mask = max_probs > conf_thresh
    if target_class is not None:
        candidate_mask = candidate_mask & (max_classes == target_class)
    if candidate_mask.sum() == 0:
        return None

    candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
    best_idx = candidate_indices[max_probs[candidate_indices].argmax()]
    bx, by, bw, bh = boxes_cxcywh[best_idx].tolist()
    x1, y1, x2, y2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2

    ax, ay = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1]
    anchor_mask = (ax >= x1) & (ax <= x2) & (ay >= y1) & (ay <= y2) & (max_probs > conf_thresh)
    return anchor_mask, ((bx / INPUT_SIZE) * 2 - 1, (by / INPUT_SIZE) * 2 - 1), (x1, y1, x2, y2)
```

The training step then minimizes confidence **only** at those anchors:

```python
patched_preds = get_raw_preds(yolo_model.model, patched_image.unsqueeze(0))
max_probs, _ = patched_preds[0, :, 4:].max(dim=-1)
targeted_conf = max_probs[target_mask]
loss = targeted_conf.mean()   # directly minimize confidence at the real object's anchors
```

Note that YOLOv8's anchor-free, decoupled head has **no separate objectness branch** (confirmed in multiple technical sources — see Section 3) — so "suppress the objectness score" from the classic literature is adapted here to "suppress the class confidence specifically at the target's anchors," which is the closest equivalent quantity YOLOv8 actually exposes.

### 2.2 Object-centered placement, both during training and at demo time

Placement is now derived from the same `select_target_object` call used for the loss, both during training (with a small random jitter, for physical realism and EoT-style robustness) and in the script's own demo step:

```python
# Training -- jittered AROUND the object's own location, not the whole canvas
jitter = PS.get("placement_jitter", 0.1)
cx = float(np.clip(obj_cx + np.random.uniform(-jitter, jitter), -0.9, 0.9))
cy = float(np.clip(obj_cy + np.random.uniform(-jitter, jitter), -0.9, 0.9))
```

```python
# Demo step -- re-detects the target in THIS specific demo image, rather
# than assuming a fixed canvas position
demo_result = select_target_object(demo_preds, target_class=PS.get("target_class"))
if demo_result is not None:
    _, (demo_cx, demo_cy), _ = demo_result
```

### 2.3 Disk-streamed training data, with an explicit `gpu_batch_size`

The dataset is now only ever listed as filenames (`discover_image_files`) — no pixel data is loaded until a specific image is sampled for a training step, at which point exactly `gpu_batch_size` images are read from disk straight to GPU, used, and released:

```python
def discover_image_files(folder_path, max_images):
    """Lists filenames only -- no pixel data loaded. Keeps memory flat
    regardless of dataset size."""
    return [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))][:max_images]

def load_single_image(path, target_size=INPUT_SIZE):
    """Loads ONE image from disk straight to GPU, on demand -- called
    fresh each time it's sampled, so the dataset never sits in GPU (or
    CPU) memory all at once."""
    ...
```

`max_images` can now safely be set to the full dataset size (e.g. `2000`) without any memory cost until training actually samples that many distinct files across all steps.

### 2.4 `specific_clean_image` now only controls the demo image

Training always pulls its batch from `clean_dir` directly, regardless of `specific_clean_image` — that setting now exclusively determines which image the finished patch is demonstrated on, decoupling "what the patch learns from" (diverse, for generalization) from "what you look at afterward" (one chosen image).

### 2.5 A standalone patch image, for manual use outside the pipeline

Alongside the `.npy` array, the raw trained patch is now also saved as a plain image file:

```python
patch_img_np = (patch.detach().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
patch_img_bgr = cv2.cvtColor(patch_img_np, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(OUTPUT_DIR / "art_patch_standalone.png"), patch_img_bgr)
```

### 2.6 `apply_saved_patch.py` rewritten to match training's own compositing convention

The script no longer imports ART at all. It uses the **exact same** `build_patch_overlay` affine-warp function that `generate_art_patch.py` trains with, so a given `scale`/`cx`/`cy` means the same thing at application time as it did during optimization — this was the direct fix for weak results observed when applying the patch through ART's `attack.apply_patch` (a different function, different geometric convention) or by hand in an image editor (which also risks lossy recompression and off-envelope scale/rotation, further degrading the patch's fine-grained pattern).

It also adds two placement modes, since the script otherwise has no way to know where an object is in a *new* image:

```python
if PLACEMENT_MODE == "manual":
    cx = (MANUAL_X / INPUT_SIZE) * 2 - 1
    cy = (MANUAL_Y / INPUT_SIZE) * 2 - 1
else:  # "auto" -- re-run the same detection-based targeting used in training
    preds = get_raw_preds(yolo_model.model, image.unsqueeze(0))
    result = select_target_object(preds, target_class=TARGET_CLASS)
    (cx, cy), _ = result
```

`"auto"` (the default) requires no manual coordinate-finding at all; `"manual"` exists for testing a specific spot, or when the target isn't confidently detected on its own in a particular test image.

---

## 3. Supporting Research Papers

| Claim | Source |
|---|---|
| "Objectness" is a distinct, class-agnostic concept ("is there an object here") separate from classification | Alexe, Deselaers & Ferrari, *"Measuring the Objectness of Image Windows,"* IEEE TPAMI 34(11), 2012 |
| Object detectors formalize objectness as part of their confidence score (`objectness × class_probability`) | Redmon, Divvala, Girshick & Farhadi, *"You Only Look Once: Unified, Real-Time Object Detection,"* CVPR 2016 |
| The original classifier-style adversarial patch concept (whole-image target class, no spatial awareness) | Brown, Mané, Roy, Abadi & Gilmer, *"Adversarial Patch,"* NeurIPS Workshop, 2017 |
| Classifier-style patches **fail** against object detectors, motivating detector-specific attack design | Liu, Yang, Liu, Song, Li & Chen, *"DPatch: An Adversarial Patch Attack on Object Detectors,"* arXiv:1806.02299, 2018 |
| A patch can be trained to suppress detection confidence specifically at a real target's location (the "hiding" objective this project adapts) | Thys, Van Ranst & Goedemé, *"Fooling Automated Surveillance Cameras: Adversarial Patches to Attack Person Detection,"* CVPR Workshops, 2019 |
| EoT — randomizing transformations during optimization produces patches robust to real-world viewing/placement variation | Athalye, Engstrom, Ilyas & Kwok, *"Synthesizing Robust Adversarial Examples,"* ICML 2018 |
| YOLOv8 removed the separate objectness branch (anchor-free, decoupled head) | MMYOLO project documentation (OpenMMLab); Ju & Cai, *"Fracture Detection in Pediatric Wrist Trauma X-ray Images Using YOLOv8 Algorithm,"* Scientific Reports, 2023; *"Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8 and YOLOv5 Object Detectors,"* 2025 |

**Not independently cited, and deliberately so:** the pseudo-labeling approach (Section 2.1), the object-centered placement fix (Section 2.2), the disk-streaming/memory design (Section 2.3), and the `apply_saved_patch.py` rewrite (Section 2.6) are this project's own engineering solutions to problems specific to this pipeline and dataset (no ground-truth annotations, limited GPU memory, a training/application convention mismatch) — not techniques drawn from a specific paper. Frame them in your writeup as your own contributions, distinct from the literature-grounded design choices above (objectness-targeting, EoT).