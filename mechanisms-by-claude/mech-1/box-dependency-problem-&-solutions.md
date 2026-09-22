# The Box-Dependency Problem: Why Mechanism 1 Can't See a Successfully Hidden Object

## Status
Documents a structural failure mode discovered once the patch generator actually succeeded at hiding objects (see `objectness-aware-patch-redesign-explained.md`), and lays out the options to fix it. This document is self-contained and can be used to start a fresh chat.

---

## 1. The New Problem

Once the redesigned patch generator (spatially-targeted, objectness-aware loss) started successfully hiding target objects from YOLOv8, the Mahalanobis-distance detection mechanism (Mechanism 1, `cls_head_detector.py`) stopped flagging those images as attacks — not because the patch became harder to detect statistically, but because the detector **never looks at the region at all** once the object is hidden.

This is a different, more fundamental issue than the coordinate-system and thresholding bugs fixed earlier in this project. Those were implementation bugs; this is a **design assumption that breaks under exactly the attack scenario the whole system exists to catch**.

---

## 2. Why the Current Mechanism Doesn't Work

### 2.1 How scoring currently decides where to look

`cls_head_detector.py` builds a class-assignment grid entirely from the *current* run's own detection output:

```python
predicted_classes = torch.full((H, W), -1, dtype=torch.long, device=self.device)
if len(results.boxes) > 0:
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        ...
        predicted_classes[y1:y2, x1:x2] = int(cls.item())
```

Every grid cell starts at `-1`. A cell only receives a real class ID if `results.boxes` — YOLO's own, possibly-attacked prediction — contains a box covering it.

### 2.2 Why that's fatal for a hiding attack specifically

The Mahalanobis distance is only computed where a class was assigned:

```python
if cls_id in self.stats:
    ... # compute Mahalanobis distance against that class's calibrated distribution
# else: scores[y, x] silently stays 0 (its initialized value)
```

`-1` is never a key in `self.stats`. So for any cell where no box was predicted, the score is **exactly zero** — not "low," not "uncertain," literally the same untouched zero the tensor was initialized with. It's indistinguishable from "nothing anomalous happening here," when the true situation is "the mechanism never looked."

A successful hiding attack, by definition, suppresses YOLO's own confidence for the target object below the detection threshold — which means it removes the box. Removing the box removes the *only* signal this mechanism uses to decide a region deserves scoring. **The detector's blind spot is created by the very success of the attack it's meant to catch.** This is a structural property of the design, not a parameter to retune.

### 2.3 Why this matters for the thesis, not just the codebase

This is worth stating plainly and not glossing over: a detection mechanism that derives its search region from the output of the model it's protecting has a built-in circularity. If the attack's goal is to suppress that exact output, the defense loses its own targeting information at precisely the moment it's needed most. This is a genuine, reportable finding about *why* class-conditioned, box-gated approaches (a fairly natural first design) have a specific weakness against hiding-style attacks, as opposed to misclassification-style attacks (where a box still exists, just with the wrong label — Mechanism 1 as originally built would actually still catch those fine).

---

## 3. What Any Fix Must Preserve

Before getting to solutions: whatever replaces the current scoring logic must **still tap the classification branch (`cv3`) only**, not the neck or the box-regression branch (`cv2`). That separation was the entire point of Mechanism 1 from the start of this project — YOLO's neck fuses multi-scale information and its feature maps otherwise carry both box-regression and classification signal in the same tensor, which is what made naive patch localization on raw YOLO features unreliable in the first place. Every option below keeps that property; none of them require touching a different tap point.

---

## 4. Option A — One Global, Class-Agnostic Distribution (the supervisor's suggestion)

### The idea
Instead of fitting one Mahalanobis distribution *per class*, fit a single pooled distribution over **every** classification-branch feature vector collected from clean images — car cells, person cells, background cells, all together as one reference. Score **every** grid cell against this one distribution, with no box-gating at all: `predicted_classes` and the `if cls_id in self.stats` gate disappear entirely.

### Why it fixes the bug
Scoring no longer depends on `results.boxes` in any way, so a hidden object's region gets scored exactly like any other region — the box vanishing no longer means "skip this cell."

### The cost
A single Gaussian representing "car," "person," "traffic sign," "sky," "road," and "grass" all at once will have very high variance, because those are legitimately very different feature distributions being forced into one reference. The practical effect: the threshold has to be set loosely enough to avoid flagging ordinary class-to-class variation as anomalous, which reduces sensitivity to the actual patch signature — the same precision-vs-coverage tradeoff that motivated per-class conditioning in the first place, just resolved by giving up the precision side entirely.

**Verdict:** the simplest fix, and a reasonable first experiment to run since it requires almost no new code — but likely not the final answer given the sensitivity cost.

---

## 5. Option B — Per-Class Distributions with Minimum-Distance Scoring (Recommended)

### The idea
Keep per-class conditioning (Mechanism 1's original strength), but stop requiring the detector to *tell you* which class applies to a cell before you can score it. Instead, at every cell, compare against **all** known class distributions (plus one new "background" distribution — see below) and take the **minimum** distance as the cell's score: *"how well does this cell's feature vector fit its best available explanation, whatever that turns out to be?"*

### The math
For a feature vector `x` at a given grid cell, and a set of class-conditional Gaussians `{(μ_c, Σ_c) : c ∈ classes ∪ {background}}`:

```
score(x) = min_c  sqrt( (x - μ_c)ᵀ Σ_c⁻¹ (x - μ_c) )
```

This is not a novel technique to invent from scratch — it's the core idea of Lee, Lee, Lee & Shin's **"A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks"** (NeurIPS 2018), which fits class-conditional Gaussians on network features and uses exactly this minimum-Mahalanobis-distance quantity as a confidence/anomaly score — designed specifically so detection doesn't require already knowing the correct class ahead of time. That's a solid, citable basis for this design.

### Why it needs a background class
Most of any given image is not inside any object's bounding box. If you only fit distributions for object classes (car, person, etc.) and score every cell against them, ordinary background cells (road, sky, foliage) will fit *none* of the object classes well and score high by default — a flood of false positives on completely normal images. Fit a background distribution the same way as the others, using clean-image cells that fall **outside** every detected box during calibration:

```python
# During calibration: alongside per-class feature collection,
# also collect feature vectors from cells NOT covered by any box.
background_features = []
for gy in range(H):
    for gx in range(W):
        if not covered_by_any_box[gy, gx]:
            background_features.append(feat[0, :, gy, gx])
# Fit mean/inv_cov for this exactly like any other class, stored as e.g. stats[-1]
```

### Code sketch for the detection side

```python
def score_cell(feat_vec, stats):
    """stats includes per-object-class entries AND a background entry."""
    best = float("inf")
    for cls_id, params in stats.items():
        diff = feat_vec - params['mean']
        dist = torch.sqrt(diff @ params['inv_cov'] @ diff)
        best = min(best, dist.item())
    return best

# No predicted_classes grid, no box-gating -- every cell scored:
scores = torch.zeros(H, W)
for y in range(H):
    for x in range(W):
        scores[y, x] = score_cell(feat_flat[y * W + x], self.stats)
```

(For performance, this per-cell Python loop over all classes should eventually be vectorized — compute Mahalanobis distance to every class distribution for the *entire* feature grid at once via batched matrix operations, then take the per-cell minimum across the class dimension. Get the logic correct first, then optimize.)

### What this preserves and what it changes
Almost the entire existing calibration pipeline survives unchanged (still per-class mean/covariance from `cv3`, same stride mapping). The changes are: (1) also fit a background distribution, (2) remove the `predicted_classes`/box-gating step entirely, (3) change the scoring rule from "look up the box's claimed class" to "minimum distance across all classes." This is a moderate, well-scoped change, not a rewrite.

---

## 6. Option C — Reconstruction-Based Scoring (Alternative Implementation Family)

### The idea
Instead of explicit Gaussian statistics, train a small autoencoder (or fit a PCA subspace) on clean `cv3` feature vectors, and score each cell by **reconstruction error**: how well the learned "normal feature manifold" can reproduce this vector. Score every cell, no box dependency — same goal as Options A and B, different underlying model.

### Why it might help specifically with the background problem
Background is visually far more diverse than any single object class (sky, road, foliage, buildings, weather variation) — a single Gaussian (as in Option B's background entry) might still be a coarse fit for something this heterogeneous. A reconstruction model with enough capacity can potentially represent that diversity more gracefully than one Gaussian can, at the cost of needing to train and validate a small model rather than just computing closed-form statistics.

**Verdict:** worth keeping in your back pocket if Option B's single background Gaussian proves too coarse in practice (e.g., high false-positive rate specifically on background cells during validation) — not necessary to build unless that specific problem shows up.

---

## 7. Option D — Mechanism 4 (APE-Style Feature Energy) Was Never Vulnerable to This

### Why it matters here specifically
Mechanism 4 (self-referential feature-energy thresholding, from earlier in this project) computes a per-cell anomaly score by comparing each cell against **that same image's own internal statistics** — it was never gated by, or dependent on, YOLO's own box predictions in the first place. It scores every cell unconditionally by construction. This box-dependency failure mode, which just took down Mechanism 1's class-conditioned scoring, was never something Mechanism 4 was exposed to.

### Why this is now a stronger argument than when Mechanism 4 was first proposed
Mechanism 4 was originally motivated by a deployment concern (no manual calibration needed). This finding adds a second, independent motivation: **structural robustness to hiding-style attacks specifically**, demonstrated empirically rather than argued theoretically. Ensembling Mechanism 4 alongside a fixed Mechanism 1 (Option B) gives you two detection signals with different failure modes — worth building out and evaluating as a pair rather than treating this purely as a Mechanism 1 patch job.

---

## 8. Recommended Path

1. **Implement Option B first.** Smallest change to the existing, already-working calibration pipeline; keeps the precision benefit of class-conditioning; has a direct literature citation (Lee et al. 2018) backing the core technique.
2. **Validate specifically against the working hiding patch** — this is now the critical test case, not just a generic clean/patched comparison. Confirm the previously-invisible hidden-object region now produces a meaningfully non-zero, ideally elevated, score.
3. **Watch background false-positive rate closely** during validation — if it's high, that's the signal to invest in Option C (reconstruction-based background modeling) rather than tuning Option B's background Gaussian further.
4. **Build out Mechanism 4 (Option D) as a second, independent layer**, and report both mechanisms' behavior against the hiding patch side by side — this box-dependency finding is a strong argument for why a single detection strategy isn't sufficient, and directly motivates an ensemble design for your evaluation chapter.
5. **Keep Option A documented as a considered-and-rejected baseline** in your writeup — it directly answers "did you try the simplest thing," and having concrete reasoning for why it's insufficient (the variance/precision cost) strengthens the case for Option B rather than leaving it unaddressed.

---

## 9. Supporting Reference

| Claim | Source |
|---|---|
| Minimum Mahalanobis distance across class-conditional Gaussians, without needing the class known in advance, as an anomaly/OOD score | Lee, Lee, Lee & Shin, *"A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks,"* NeurIPS 2018 |