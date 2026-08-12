# Generating a Test Adversarial Patch for YOLOv8

## Status
Practical guide for producing an adversarial patch to test Mechanism 1 (decoupled-head classification-branch tap) against. This document is self-contained and can be used to start a fresh chat.

---

## 1. The Problem

To validate the detection mechanism, you need at least one adversarial patch that actually works against your specific YOLOv8 model — i.e., one that measurably degrades detection (hides an object, or triggers a false detection) when placed in a test image. Two broad ways to get one:

1. **Find/reuse an existing patch.** Fast, but adversarial patches are frequently model-specific — a patch optimized against one detector (or an older YOLO version, or a different architecture like Faster R-CNN/SSD) often transfers poorly to your exact YOLOv8 checkpoint. You may get a patch instantly but find it barely affects your model.
2. **Generate a patch optimized directly against your YOLOv8 model.** Slower to set up, but the resulting patch is guaranteed to be effective against your exact model (a true white-box attack), which is what you actually need to properly exercise the detection mechanism.

Given your goal — testing whether your detector can find a patch that *actually blinds YOLOv8* — option 2 is the reliable path, and it's not much extra work since the tooling for it (ART, EoT) is already in your project's planned toolkit. Option 1 is worth trying first as a quick sanity check, but treat it as a bonus, not your primary test case.

---

## 2. Option 1: Fast Path — Existing Patches / Datasets

### 2.1 APRICOT dataset
[apricot.mitre.org](https://apricot.mitre.org) — a public dataset of **1,011 real photographs of 60 unique printed adversarial patches**, placed in real-world scenes, crafted against three COCO-trained object detectors. It's designed exactly for benchmarking detection defenses against physical patches.

**Caveats specific to your use case:**
- The patches were optimized against the paper's three target models (not YOLOv8 specifically), and the paper's own results show black-box transfer to *other* models is inconsistent ("sometimes" effective in black-box settings, per the authors). Don't be surprised if an APRICOT patch barely registers on your YOLOv8 checkpoint.
- The attack goal in APRICOT is **triggering false-positive detections of a target class**, not hiding real objects — useful, but a different attack goal than "blind the detector to a real object," if that's specifically what you want to test.
- Good use: a fast sanity check of your pipeline end-to-end (load image → run patch through your adapter → confirm your detector's hooks and heatmap code actually run correctly) before you invest time in generating your own patch.

### 2.2 Public adversarial-patch repos (person-hiding style)
Repos implementing the original Thys, Van Ranst & Goedemé "Fooling automated surveillance cameras" approach (the canonical person-hiding patch) exist publicly (search GitHub for `adversarial-yolo`), but they were built against **YOLOv2**. Porting the training loop to YOLOv8 is realistically as much work as building the ART pipeline in Option 2 directly, so this is not a shortcut — mentioned here mainly so you know it exists and can skip it.

**Recommendation:** use APRICOT for a 15-minute pipeline sanity check, then move to Option 2 for your actual test cases.

---

## 3. Option 2: Generate a Patch Directly Against Your YOLOv8 Model (Recommended)

### 3.1 Why ART specifically

The Adversarial Robustness Toolbox (ART) — already in your blueprint's planned tooling alongside EoT — added native support for **YOLO v8 and later** via its `PyTorchYolo` object-detection estimator starting with ART 1.20.0. Its `AdversarialPatchPyTorch` attack:
- Works directly against object detectors (not just classifiers).
- Has **Expectation-over-Transformation built in** — during optimization it applies random scale, rotation, and location jitter to the patch on each training image, so the resulting patch is robust to the kind of viewpoint/distance variation your threat model already assumes for physical patches, rather than only working at one exact position/size.
- Produces a patch you can then digitally composite onto any test image via `apply_patch`, which is all you need for validating a detection mechanism (you don't need to physically print anything for this stage).

### 3.2 Install dependencies

```bash
pip install adversarial-robustness-toolbox[pytorch]
pip install ultralytics
pip install torch torchvision
```

Verify your ART version supports YOLOv8:
```bash
python -c "import art; print(art.__version__)"
# Must be >= 1.20.0
```

### 3.3 Wrap your YOLOv8 model in ART's estimator

```python
from ultralytics import YOLO
from art.estimators.object_detection import PyTorchYolo

# Load your actual checkpoint (the same one used in your middleware)
yolo_model = YOLO("yolov8n.pt")   # or your trained/fine-tuned checkpoint
yolo_model.model.eval()

detector = PyTorchYolo(
    model=yolo_model.model,
    input_shape=(3, 640, 640),
    clip_values=(0, 1),
    channels_first=True,
    attack_losses=("loss_total",),  # verify exact loss key name against your ART version's PyTorchYolo docstring
)
```

> The exact `attack_losses` key name has varied slightly across ART versions as YOLO support matured — check `help(PyTorchYolo)` or the source for your installed version before running this, since a wrong key will raise immediately rather than silently misbehave.

### 3.4 Prepare training images

The patch is optimized over a **batch of images containing the object class you want to attack** (e.g., "stop sign" if you're testing the AV/traffic-sign threat scenario from your blueprint, or "person," or "car"). Pull a subset directly from your existing KITTI/BDD100K pipeline:

```python
import numpy as np

# images: np.ndarray of shape [N, 3, 640, 640], normalized to [0,1]
# You can reuse your existing dataset-loading code from the detection
# module's calibration pipeline — same preprocessing, different purpose.
images = load_and_preprocess_images(your_image_subset, size=640)
```

Using multiple images (not just the one you'll ultimately test on) during patch optimization is what makes the resulting patch generalize, rather than overfitting to a single photo's exact lighting/background.

### 3.5 Configure and run the patch attack

```python
from art.attacks.evasion import AdversarialPatchPyTorch

attack = AdversarialPatchPyTorch(
    estimator=detector,
    rotation_max=22.5,        # EoT rotation range in degrees
    scale_min=0.2,             # EoT scale jitter — smallest patch size tried
    scale_max=0.4,             # EoT scale jitter — largest patch size tried
    learning_rate=0.01,
    max_iter=500,              # start here, increase if the patch under-performs
    batch_size=8,
    patch_shape=(3, 150, 150), # patch size in pixels — tune relative to your 640px images
    targeted=False,            # False = "hide/disrupt detection"; True = force a specific wrong class
)

patch, patch_mask = attack.generate(x=images)
```

- `targeted=False` optimizes for general disruption (suppressing correct detections) — the more relevant goal for testing a *detection* mechanism, since you want to know if your monitor notices something is wrong, not specifically whether it can be tricked into a particular wrong label.
- If you specifically want to reproduce the "stop sign ignored entirely" scenario from your blueprint's threat model, set `targeted=False` and select training images where the stop-sign class dominates the loss target — check ART's object-detection patch examples/notebooks for the exact way to bias the attack toward disrupting one specific class's detections, since the parameter surface for class-targeting varies by ART version.

### 3.6 Apply the patch to your test image

```python
test_image = load_and_preprocess_image("your_test_image.jpg", size=640)

patched_image = attack.apply_patch(
    x=test_image[np.newaxis, ...],
    patch_external=patch,
    # scale/location can be fixed here for a specific test case, or
    # left to the attack's default placement logic
)

# Save for inspection / feed into your middleware
save_image(patched_image[0], "test_image_patched.png")
```

### 3.7 Sanity-check the patch actually works before testing your detector

Before running your anomaly detector on the patched image, confirm the attack succeeded on its own terms — run the *unmodified* YOLOv8 model on both the clean and patched image and compare:

```python
clean_results = yolo_model.predict(test_image)
patched_results = yolo_model.predict(patched_image[0])

print("Clean detections:", clean_results[0].boxes)
print("Patched detections:", patched_results[0].boxes)
```

If detections on the patched image are not meaningfully different from the clean image (missing detections, dropped confidence, or a spurious wrong-class detection), the patch isn't effective yet — see the tuning notes below before moving on to testing your detection mechanism against it.

---

## 4. Step-by-Step Guide

1. **Confirm ART ≥ 1.20.0 is installed** (required for native YOLOv8 support via `PyTorchYolo`).
2. **Load your exact YOLOv8 checkpoint** — use the same model/weights your middleware will run against, not a different pretrained variant, so the patch is a genuine white-box attack on your actual target.
3. **Wrap it in `PyTorchYolo`**, checking the correct `attack_losses` argument for your installed ART version.
4. **Assemble a training image subset** (10–50 images is a reasonable starting range) containing the object class you want to attack, reusing your existing KITTI/BDD100K preprocessing pipeline.
5. **Run `AdversarialPatchPyTorch`** with `targeted=False` for a general-disruption patch; start with `max_iter=500` and increase if step 7 below shows weak results.
6. **Apply the resulting patch to your actual test image** via `apply_patch`.
7. **Verify effectiveness against the unmodified model first** — compare clean vs. patched detections before testing your defense, so you know any failure to detect later is about your defense, not a weak patch.
8. **If the patch is weak:** increase `max_iter`, widen the `scale_min`/`scale_max` EoT range to match how the patch will actually appear in your test image, or reduce the diversity of training images (a patch trained on too broad a set can under-optimize for any one condition — trade generalization for potency if your test only needs to cover one scenario).
9. **Once the patch reliably degrades detection**, use `test_image_patched.png` as the input to your Mechanism 1 detection pipeline (the `YOLOv8ClsHeadAdapter` → `ClsHeadMahalanobisDetector` flow) and evaluate the resulting `AnomalyResult` — this is your actual test case.
10. **Repeat with 2–3 different patch configurations** (different sizes, different target classes) rather than relying on a single patch — a defense that only handles one specific patch instance is a much weaker result than one validated against several variations, and this is cheap to do now that steps 1–8 are already set up.
