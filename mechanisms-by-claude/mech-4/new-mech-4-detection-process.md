# Mechanism 4: Self-Referential Feature-Energy Detection

## Status
Explains why this mechanism is worth building alongside Mechanism 1, the APE-style approach it's adapted from, and the actual implementation in `stem_activation_adapter.py` / `feature_energy_detector.py`. Companion to `README.md` for this module (setup and execution).

---

## 1. The Problem This Mechanism Answers

Two separate, independently-motivated concerns from earlier work on Mechanism 1 both point at the same gap:

**Deployment friction.** Mechanism 1 requires an offline calibration step — collect a clean dataset, run the model over it, fit per-class (and background) Mahalanobis statistics, persist a profile. In a real deployment, asking every client to run this themselves isn't realistic (see `automating-anomaly-threshold.md`, Option 8 in that discussion). Any detector that needs zero calibration data is a direct, structural answer to that constraint, not just an optimization of it.

**Structural independence from the model's own output.** Separately, Mechanism 1's original design depended on the detector's own predicted boxes to decide what to score — a dependency that turned out to have a serious blind spot once patches got good enough to actually hide objects (see `box-dependency-problem-and-solutions.md`): a successfully hidden object produces no box, and no box meant no score. Mechanism 1 was fixed for this (Option B: score every cell, minimum distance across all classes), but that fix was necessary specifically *because* the original design leaned on the detector's own output. A mechanism that never depended on box predictions, or even the concept of "predicted class," in the first place would never have had this failure mode to begin with.

Mechanism 4 answers both at once: it needs no calibration dataset, and it scores every spatial location by comparing it only to *that same image's own* statistics — no box, no predicted class, no external reference distribution of any kind.

---

## 2. The Approach: Feature Energy, Compared Against the Image's Own Statistics

### 2.1 Core idea

Instead of asking "does this feature vector look unusual compared to a reference built from thousands of other images" (Mechanism 1's approach), this asks a self-referential question: **"does this location's feature energy look unusual compared to the rest of THIS SAME image's feature map?"**

**Feature energy** at a spatial location is a scalar summary of how strongly that location is activated, aggregated across the channel dimension:

```
energy(y, x) = Σ_c  feature_map[c, y, x]²
```

The threshold is then derived **per image, per inference**, from that same energy map's own mean and standard deviation:

```
threshold = mean(energy_map) + k · std(energy_map)
```

A location is flagged if its energy exceeds this adaptive cutoff. Because the comparison is entirely internal to the current image, a foggy scene, an unusual lighting condition, or a camera the calibration set never saw doesn't require any prior exposure — the threshold simply recomputes itself fresh, every time.

### 2.2 Where this comes from, and an honest caveat

This is adapted from Kim, Yu & Ro's **"Defending Physical Adversarial Attack on Object Detection via Adversarial Patch-Feature Energy"** (Proceedings of the 30th ACM International Conference on Multimedia, pp. 1905–1913, 2022; DOI: 10.1145/3503161.3548362) — the one paper in this project's reference set built specifically for object detectors, using an early-layer feature-energy signal to localize adversarial patches. The exact energy formula and threshold constant above are a best-effort reconstruction from the paper's title, abstract, and how it's characterized in citing surveys — not a line-by-line reproduction of its method section, which sits behind an ACM paywall. Treat the formula and `k ≈ 3.5` as a well-motivated **starting implementation** to validate empirically against your own data, not a verbatim match to the original paper. If this mechanism ends up centrally reported in your thesis, it's worth tracking down full access to the paper (via your institution's library or the authors) to confirm the exact formulation before citing it as "reproducing APE."

### 2.3 Why the tap point is the backbone stem specifically

The original concept is described as operating on **early, first-layer features**. In this project's terms, that points to the **backbone stem** — the very first convolutional block(s), before any CSP/bottleneck stage builds up semantic abstraction. This tap point has a useful property specific to this project: it sits upstream of *both* problems that motivated Mechanisms 1 and 2 — PANet's multi-scale fusion (which muddies spatial correspondence) and the classification/regression head split (which doesn't exist yet this early in the network). Tapping here avoids both by construction, at the cost of being a low-level, texture/edge representation rather than a semantic one — meaning this mechanism is likely more sensitive to natural high-texture false positives (foliage, gravel, complex urban clutter) than Mechanism 1's more semantic classification-branch tap. This trade-off should be validated empirically, not assumed.

---

## 3. Implementation

### 3.1 Adapter (`stem_activation_adapter.py`) — the only model-specific file

```python
STEM_LAYER_INDEX = {
    "yolov8": 0,   # VERIFY against your checkpoint -- see README.md
}

class YOLOv8StemAdapter:
    def __init__(self, model, version_key="yolov8"):
        self.model = model
        self.layer_index = STEM_LAYER_INDEX[version_key]
        ...
        target_layer = self.model.model[self.layer_index]
        self._hook = target_layer.register_forward_hook(self._on_forward)

    def get_activations(self, image):
        self._activation = None
        with torch.no_grad():
            self.model(image)
        return {"stem": self._activation}
```

Everything model-specific is contained here. `FeatureEnergyDetector` never imports `ultralytics` and never sees a YOLO-specific concept — the entire model-agnosticism guarantee rests on every adapter returning the same `{"stem": Tensor}` contract, regardless of what's actually inside `get_activations`.

### 3.2 Detector (`feature_energy_detector.py`) — model-agnostic

```python
def _compute_energy_map(self, feat):
    return (feat[0] ** 2).sum(dim=0)  # [H, W]

def detect(self, img_path, save_visualization=True):
    ...
    activations = self.adapter.get_activations(img_tensor)
    feat = activations["stem"]
    energy_map = self._compute_energy_map(feat)

    mean, std = energy_map.mean(), energy_map.std()
    threshold = mean + self.k * std
    mask = (energy_map > threshold)
    ...
```

No calibration profile is loaded, no `.pkl` file exists for this mechanism at all — every quantity needed for the decision is computed inline, from the current image alone.

### 3.3 Noise filtering

A single spuriously energetic pixel isn't a patch. Connected-component filtering (`min_region_cells`) requires a contiguous block of flagged cells before treating anything as a genuine detection — the same principle used in Mechanism 1's box-tightening work, applied here as a basic sanity filter rather than a precision-tuning step.

---

## 4. Relationship to Mechanism 1

This is not positioned as a replacement for Mechanism 1 — it's a structurally independent second signal, worth running alongside it (or ensembling: flag an attack if *either* mechanism fires, or require agreement between both to raise confidence). Mechanism 1, once fixed (Option B), is more precise where it has good calibration data; Mechanism 4 costs nothing to deploy and is immune to the box-dependency failure mode by construction, at the cost of being more exposed to natural-texture false positives. Reporting both side by side, including where they agree and where they diverge, is a stronger empirical result than either alone.

---

## 5. Supporting Reference

| Claim | Source |
|---|---|
| Per-location feature-energy thresholding, derived from the image's own statistics, for localizing adversarial patches in object detectors | Kim, Yu & Ro, *"Defending Physical Adversarial Attack on Object Detection via Adversarial Patch-Feature Energy,"* Proceedings of the 30th ACM International Conference on Multimedia (MM '22), pp. 1905–1913, 2022. DOI: 10.1145/3503161.3548362 |