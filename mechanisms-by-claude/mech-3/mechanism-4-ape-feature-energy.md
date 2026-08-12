# Mechanism 4: APE-Style Feature-Energy Detection

## Status
Candidate detection mechanism for the Anomaly Detection module (Module 1) of the Runtime Security Middleware, adapted from Kim, Yu & Ro's "Adversarial Patch-Feature Energy" (APE), ACM Multimedia 2022 — the one paper in your reference list built specifically for object detectors rather than classifiers. This document is self-contained — it can be used to start a fresh chat focused only on implementing this mechanism.

**Important caveat before you start:** the description below is reconstructed from the paper's abstract, its self-reported concept name ("Adversarial Patch-Feature Energy"), and how it's characterized in survey/citation contexts (feature energy computed at early/first-layer features, an outlier threshold in the 3–4 sigma range, masking followed by a refinement step). I have **not** read the full method section of the original paper line-by-line to confirm the exact energy formula or threshold value. Treat the formula and hyperparameters below as a well-motivated **starting implementation**, not a verbatim reproduction of the paper — pull the actual paper (ACM MM 2022, DOI 10.1145/3503161.3548362) before finalizing anything you plan to cite as "APE's method" in your thesis.

---

## 1. The Problem

Your existing detectors (Mahalanobis-distance, PCA-based) require building an offline reference distribution of "clean" activations, then flagging new inputs whose activations deviate from that distribution. This has two costs specific to your setting:

- **It requires a large, representative clean calibration set** to generalize across KITTI's urban driving conditions and BDD100K's diverse weather/lighting conditions. If the calibration set doesn't cover a condition well (e.g., heavy rain, low light), the reference distribution will be a poor fit and the detector will misfire — either flagging legitimate rare conditions as attacks, or failing to flag real attacks that happen to fall inside the (badly calibrated) "normal" range for that condition.
- **It's a global comparison, not a per-image one.** A feature vector is judged against a distribution built from thousands of *other* images, not against the rest of *this* image. This is exactly the kind of external-reference dependency that makes cross-domain generalization (KITTI → BDD100K, or clean weather → adverse weather) fragile.

Additionally, both Mechanism 1 and Mechanism 2 (the other two you're pursuing) still inherit YOLO's box/class entanglement and multi-scale fusion issues to varying degrees, depending on exactly which layer is tapped. It's worth having a genuinely different *kind* of signal — one that doesn't depend on offline calibration at all — as a third, independent detection strategy.

---

## 2. The Suggested Mechanism

### 2.1 Core idea

Instead of asking "does this feature vector look unusual compared to a reference distribution built from other clean images," APE asks a **self-referential** question: "does this spatial location's feature energy look unusual compared to the rest of *this same image's* feature map?"

**Feature energy** at a spatial location is a scalar summary of how "activated" that location is across the channel dimension — typically something like the sum (or L2 norm) of squared activation values across channels at that grid cell:

```
energy(y, x) = Σ_c  feature_map[c, y, x]²
```

The intuition (consistent with several other papers in your reference list — feature-norm-clipping approaches in particular) is that adversarial patches are optimized to produce a strong, artificial gradient signal in the network, which tends to manifest as **abnormally high activation energy** concentrated in the patch region, compared to the energy distribution across the rest of the same feature map.

Because the comparison is **within the same image**, not against an external calibration set, this sidesteps the "does my clean dataset cover this condition" generalization problem — a foggy BDD100K frame is compared against its own energy distribution, not against a KITTI-trained reference that never saw fog.

### 2.2 The thresholding rule

For a given feature map (shape `[C, H, W]`), compute the energy map (shape `[H, W]`), then compute that energy map's own mean and standard deviation across all `H×W` spatial locations. Flag any location whose energy exceeds:

```
threshold = mean(energy_map) + k * std(energy_map)
```

with `k` around 3.5 as a starting point (adjust empirically — see Step-by-Step guide). This is a per-image, per-inference adaptive threshold: it recalculates automatically for every frame, rather than using one fixed global cutoff.

### 2.3 Where to tap

The original concept is described as operating on **early / first-layer features**. In YOLO terms, this points to the **backbone stem** — the first one or two convolutional blocks, before any of the CSP/bottleneck stages that build up semantic abstraction. This has a useful side effect for your project specifically: the stem is upstream of *everything* — the PANet fusion problem and the box/class entanglement problem both happen much later in the network, so tapping this early automatically avoids both.

The trade-off: stem features are low-level (edges, color gradients, simple textures), not semantic. This is the same trade-off the P3-backbone-stage option in Mechanism 2 faces, but more pronounced — expect this mechanism to be the most sensitive to natural high-texture false positives (foliage, gravel, complex urban clutter), and plan validation accordingly.

### 2.4 Relationship to the paper's "APE-refinement" step

The original paper pairs APE-masking (the detection step described above) with **APE-refinement**, which reforms/clips the flagged feature-energy values before continuing inference — that is a *recovery* mechanism, not detection. For your architecture, this maps to your Tier 2 recovery path (masking & re-inference), not to Module 1. Keep the two separate: implement and evaluate the masking/detection half here; if you later want to explore an APE-style recovery path as an alternative to your current masking-and-re-inference approach, that's a separate design decision for the Recovery Engine, not this module.

---

## 3. Code Structure

This mechanism is the simplest of the three to deploy operationally, because it **requires no offline calibration pipeline** — there's no reference distribution to fit and persist. It still fits your Adapter/Strategy architecture: a new `IActivationProvider` implementation for the stem tap, and a new `IAnomalyDetector` implementation with self-contained per-image statistics.

### 3.1 Adapter: backbone stem tap

```python
class IActivationProvider(ABC):
    @abstractmethod
    def get_activations(self, image: Tensor) -> Dict[str, Tensor]:
        """Returns a dict with a single key ('stem') mapping to the
        backbone's early-layer feature map."""
        ...


# Per-version index for the stem block(s) — the first 1-2 conv layers
# of the backbone, before the first CSP/bottleneck stage begins.
STEM_LAYER_INDEX = {
    "yolov8": 1,    # verify against your checkpoint
    "yolov5": 1,    # verify against your checkpoint
    "yolov10": 1,   # verify against your checkpoint
}


class BackboneStemAdapter(IActivationProvider):
    def __init__(self, model: nn.Module, version_key: str):
        self.model = model
        self.layer_index = STEM_LAYER_INDEX[version_key]
        self._activation: Optional[Tensor] = None
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        target_layer = self.model.model[self.layer_index]
        self._hook = target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, input, output):
        self._activation = output.detach()

    def get_activations(self, image: Tensor) -> Dict[str, Tensor]:
        self._activation = None
        with torch.no_grad():
            self.model(image)
        return {"stem": self._activation}

    def teardown(self):
        if self._hook:
            self._hook.remove()
```

### 3.2 Strategy: feature-energy detector

```python
class FeatureEnergyDetector(IAnomalyDetector):
    """Self-referential, per-image anomaly detection — no offline
    calibration dependency. Implements the same IAnomalyDetector
    interface as your existing MahalanobisDetector/PCADetector,
    so it can be swapped in as a third Strategy without touching
    the Recovery Engine."""

    def __init__(self, k: float = 3.5, stride: int = 2,
                 min_region_size: int = 4):
        self.k = k
        self.stride = stride
        self.min_region_size = min_region_size

    def detect(self, activations: Dict[str, Tensor]) -> AnomalyResult:
        feat = activations["stem"]                  # [1, C, H, W]
        energy_map = self._compute_energy(feat)      # [H, W]

        mean = energy_map.mean()
        std = energy_map.std()
        threshold = mean + self.k * std

        mask = energy_map > threshold
        mask = self._remove_small_regions(mask, self.min_region_size)

        if not mask.any():
            return AnomalyResult(is_attack=False, score=0.0, bounding_box=None)

        bbox = self._extract_tight_bbox(mask, self.stride)
        score = (energy_map[mask].max() - mean) / std   # normalized deviation
        return AnomalyResult(is_attack=True, score=score.item(), bounding_box=bbox)

    def _compute_energy(self, feat: Tensor) -> Tensor:
        # Sum of squared activations across the channel dimension.
        return (feat[0] ** 2).sum(dim=0)   # [H, W]

    def _remove_small_regions(self, mask: Tensor, min_size: int) -> Tensor:
        # Connected-component filtering — drop components smaller than
        # min_size to suppress single-pixel noise before bbox extraction.
        ...

    def _extract_tight_bbox(self, mask: Tensor, stride: int) -> Tuple[int, int, int, int]:
        ys, xs = torch.where(mask)
        y0, y1 = ys.min().item(), ys.max().item()
        x0, x1 = xs.min().item(), xs.max().item()
        return (x0 * stride, y0 * stride, (x1 + 1) * stride, (y1 + 1) * stride)
```

### 3.3 No calibration script needed

Unlike Mechanisms 1 and 2, there is **no offline fitting step** — remove that stage from your MLOps pipeline for this Strategy specifically. This is worth calling out explicitly in your architecture writeup as a genuine operational trade-off between the three mechanisms: this one has zero calibration-drift risk (nothing to go stale as conditions shift) at the cost of being more exposed to per-image noise (a naturally very high-contrast clean image could produce a locally "energetic" region that isn't a patch at all).

---

## 4. Step-by-Step Implementation Guide

1. **Locate and read the original paper's method section** (ACM MM 2022, DOI 10.1145/3503161.3548362) before finalizing your implementation — confirm the exact energy formula (sum of squares vs. L2 norm vs. something else), the exact layer(s) used, and the exact threshold value/rule, since this document reconstructs those from secondary characterizations, not the primary source.
2. **Identify the stem layer index** for your target YOLO version(s) by printing the model's module list and cross-referencing the architecture config, same as the other two mechanisms.
3. **Implement the adapter.** Write `BackboneStemAdapter` implementing `IActivationProvider`, hooking the first conv block(s) of the backbone.
4. **Implement the detector.** Write `FeatureEnergyDetector` implementing `IAnomalyDetector`, computing the per-image energy map, adaptive threshold, and connected-component-filtered mask.
5. **Sanity-check on clean images first.** Since there's no calibration step to catch obvious misconfiguration early, run this on a batch of clean images before touching adversarial data, and manually inspect the resulting heatmaps — you're checking that "nothing is flagged" (or only small, filtered-out noise) on ordinary frames.
6. **Sweep the threshold constant `k`.** Try a range (e.g. 2.5 to 4.5) on a validation split and plot the resulting TPR/FPR trade-off — don't assume the paper's reported value transfers directly to YOLO's stem features, since the original was validated on a different architecture and layer.
7. **Validate on adversarial patches.** Run against ART/EoT-generated physical patches; measure TPR/FPR and bounding-box IoU against ground truth, same protocol as the other two mechanisms for a fair comparison.
8. **Explicitly test the natural-high-texture false-positive case.** This mechanism, tapping the earliest layer of the three, is the most likely to be sensitive to foliage/gravel/cluttered-background false positives (this is a documented failure mode for early-layer, entropy/energy-style detectors in general, per Bunzel et al. in your reference list). Treat this as a required experiment, not optional.
9. **Compare against Mechanisms 1 and 2 on the identical test set.** Since this mechanism requires no calibration data while the other two do, also compare **operational cost** (calibration pipeline maintenance, sensitivity to calibration-set drift) alongside raw detection accuracy — this three-way comparison, spanning both accuracy and SE/operational trade-offs, is a strong candidate for a dedicated section in your evaluation chapter.
10. **If accuracy is promising but false-positive rate on natural textures is too high, consider a two-stage combination**: use this mechanism as a fast, cheap first-pass filter, and only run the more expensive per-class Mahalanobis detector (Mechanism 1) on regions this mechanism flags — reducing average-case latency while keeping precision from the class-conditioned check.
