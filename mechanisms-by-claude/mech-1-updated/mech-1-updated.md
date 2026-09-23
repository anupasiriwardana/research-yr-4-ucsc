# Mechanism 1, Option B: Box-Independent Detection via Minimum Mahalanobis Distance

## Status
Explains the problem that led to Option B, briefly surveys the alternatives considered, then documents the chosen solution in full, with the actual code from `calibrate_cls_head.py` and `cls_head_detector.py`. Companion to `mech-1-optionB-tightPatchBox.md` (the bounding-box refinement built on top of this) and `README.md` (setup and execution for both scripts).

---

## 1. The Problem: A Successful Hiding Patch Becomes Invisible to the Detector

Mechanism 1's original design scored a grid cell's Mahalanobis distance only if YOLO's own prediction placed a box over that cell:

```python
predicted_classes = torch.full((H, W), -1, dtype=torch.long, device=self.device)
if len(results.boxes) > 0:
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        ...
        predicted_classes[y1:y2, x1:x2] = int(cls.item())
```

Scoring was then gated on that assignment:

```python
if cls_id in self.stats:
    ... # compute Mahalanobis distance
# else: scores[y, x] silently stays 0
```

`-1` (the default, "no box here" value) is never a key in `self.stats`, so any cell without a box scores exactly `0` — not "low," literally untouched. Once the patch generator (see `objectness-aware-patch-redesign-explained.md`) succeeded at actually hiding its target object, YOLO stopped emitting a box for that object — which removed the *only* signal this mechanism used to decide a region was worth scoring at all. **The detector's blind spot is created by the very success of the attack it exists to catch.** This is a structural property of box-gated, class-conditioned scoring, not a tunable parameter.

Full root-cause analysis: `box-dependency-problem-and-solutions.md`.

---

## 2. Options Considered (Brief)

| Option | Idea | Why not chosen (or chosen) |
|---|---|---|
| **A** | One global, class-agnostic Mahalanobis distribution; score every cell against it | Simple, but pooling every class (car, person, sky, road...) into one Gaussian produces very high variance, forcing a loose threshold and losing sensitivity. Documented as a considered-and-rejected baseline. |
| **B** | Keep per-class distributions (plus a new background class); score every cell against **all** of them and take the minimum distance | **Chosen.** Keeps class-conditioning's precision, removes the box dependency, has a direct literature basis (Section 4). |
| **C** | Reconstruction-based (autoencoder/PCA) scoring instead of explicit Gaussians | Alternative implementation family for the same goal; kept in reserve in case Option B's single background Gaussian proves too coarse for how visually diverse background actually is. |
| **D** | Mechanism 4 (APE-style feature energy) — self-referential, never box-gated in the first place | Not a replacement for Mechanism 1, but a structurally independent detection layer worth ensembling alongside it, since this whole failure mode never applied to it. |

---

## 3. Option B in Detail

### 3.1 Core idea

Stop asking "which class does the detector currently claim is here" before scoring a cell. Instead, ask: **"how well does this cell's feature vector fit the *best* available explanation, whatever that turns out to be?"** — computed by comparing every cell against every calibrated class distribution and taking the minimum Mahalanobis distance:

```
score(x) = min_c  sqrt( (x - μ_c)ᵀ Σ_c⁻¹ (x - μ_c) )    for c ∈ classes ∪ {background}
```

This is not a novel technique invented for this project — it's the core idea of Lee, Lee, Lee & Shin's **"A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks"** (NeurIPS 2018), which uses exactly this minimum-Mahalanobis-distance quantity as a confidence/anomaly score, specifically so detection doesn't require knowing the correct class in advance.

### 3.2 Why a background class is required

Most grid cells in any image are not inside any object's box. Without a background reference, every ordinary background cell (road, sky, foliage) would fail to match any object class well and score high by default — a flood of false positives on completely normal images. A background distribution is fitted the same way as any object class, from clean-image cells that fall outside every detected box.

### 3.3 Calibration changes (`calibrate_cls_head.py`)

Object-class collection is unchanged from the original Mechanism 1 calibration. The addition is a bounded, random background sample per image:

```python
BACKGROUND_ID = config["detector_settings"].get("background_class_id", -1)
BG_SAMPLES_PER_IMAGE = config["detector_settings"].get("background_samples_per_image", 50)

...

# Tracks which grid cells are covered by a real object box, so the
# remaining cells can be sampled as background candidates below.
covered = torch.zeros((H, W), dtype=torch.bool)

if len(results.boxes) > 0:
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        cls_id = int(cls.item())
        x1, y1, x2, y2 = (box / stride).int().tolist()
        x1 = min(max(x1, 0), W - 1); x2 = min(max(x2, 0), W)
        y1 = min(max(y1, 0), H - 1); y2 = min(max(y2, 0), H)
        if x2 <= x1 or y2 <= y1:
            continue
        covered[y1:y2, x1:x2] = True
        for gy in range(y1, y2):
            for gx in range(x1, x2):
                per_class_features[cls_id].append(feat[0, :, gy, gx].cpu())

# --- Background sampling ---
# A random, CAPPED subsample of non-object cells per image. Capping
# matters: an 80x80 P3 grid has 6,400 cells, and most of any image is
# background, so collecting every uncovered cell across a
# multi-thousand-image dataset would dwarf the object-class data and
# risk a memory blowup across the whole calibration run.
bg_coords = (~covered).nonzero(as_tuple=False).tolist()
random.shuffle(bg_coords)
for gy, gx in bg_coords[:BG_SAMPLES_PER_IMAGE]:
    per_class_features[BACKGROUND_ID].append(feat[0, :, gy, gx].cpu())
```

Fitting proceeds identically for every class, background included:

```python
stats = {}
for cls_id, feats in per_class_features.items():
    if len(feats) > min_samples:
        feats_tensor = torch.stack(feats)
        mean = feats_tensor.mean(dim=0)
        cov = torch.cov(feats_tensor.T) + 1e-6 * torch.eye(feats_tensor.shape[1])
        inv_cov = torch.linalg.inv(cov)
        stats[cls_id] = {'mean': mean, 'inv_cov': inv_cov}
```

The resulting profile (`cls_head_calibration_p3_optionB.pkl`) is **not** interchangeable with a profile produced by the original box-gated calibration script — it has an extra class (background) and is meant to be consumed by min-distance scoring, not class-lookup scoring.

### 3.4 Detection changes (`cls_head_detector.py`)

`results.boxes` is still computed, but purely for optional visual comparison in the heatmap overlay — it plays no role in deciding what gets scored.

**Setup — stack every class's stats once, for vectorized scoring:**

```python
self.class_ids = list(self.stats.keys())
self.means = torch.stack([self.stats[c]['mean'] for c in self.class_ids]).to(self.device)        # [K, C]
self.inv_covs = torch.stack([self.stats[c]['inv_cov'] for c in self.class_ids]).to(self.device)   # [K, C, C]
```

**Scoring — every cell against every class, minimum wins:**

```python
def _min_mahalanobis_scores(self, feat_flat):
    """feat_flat: [N, C] -- one row per grid cell, N = H*W.
    Returns scores: [N], the MINIMUM Mahalanobis distance across ALL
    calibrated classes for each cell. The loop is over K classes
    (a handful to a few dozen), never over N cells individually --
    each iteration scores every cell in the image against one class
    in a single batched operation."""
    N = feat_flat.shape[0]
    K = self.means.shape[0]
    best = torch.full((N,), float('inf'), device=feat_flat.device)

    for k in range(K):
        diff = feat_flat - self.means[k]                                    # [N, C]
        maha_sq = torch.einsum('nc,cd,nd->n', diff, self.inv_covs[k], diff)  # [N]
        dist = torch.sqrt(torch.clamp(maha_sq, min=0))
        best = torch.minimum(best, dist)

    return best
```

**In `detect()`, this replaces the old box-gated lookup entirely:**

```python
feat_flat = feat.permute(0, 2, 3, 1).reshape(H * W, C)
scores_flat = self._min_mahalanobis_scores(feat_flat)
scores = scores_flat.reshape(H, W)
```

No `predicted_classes` grid, no `if cls_id in self.stats` gate — every cell in the image gets a real score, always.

### 3.5 What this preserves

The classification-branch (`cv3`) tap point is unchanged — this fix operates entirely on *how scores are computed from* the activations, not *which* activations are tapped. The original reason for tapping `cv3` (isolating classification signal from box-regression signal, since YOLO's neck fuses both into one tensor) is untouched by this change.

### 5. Validation Signature

The concrete test for this fix: on an image where the patch has successfully hidden its target object, open the saved heatmap overlay. You should see YOLO's own green detection box **missing** over the object (drawn from `results.boxes`, purely informational), while the heatmap underneath still shows elevated scores in that same region. Box gone, heatmap still lit — that combination is the direct evidence that Option B closed the blind spot the original design had.

---

## 6. Supporting Reference

| Claim | Source |
|---|---|
| Minimum Mahalanobis distance across class-conditional Gaussians, without needing the class known in advance, as an anomaly/OOD score | Lee, Lee, Lee & Shin, *"A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks,"* NeurIPS 2018 |