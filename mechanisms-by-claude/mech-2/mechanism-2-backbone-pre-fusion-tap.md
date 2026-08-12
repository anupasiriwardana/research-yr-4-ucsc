# Mechanism 2: Backbone (Pre-PANet) Feature Tap

## Status
Candidate detection mechanism for the Anomaly Detection module (Module 1) of the Runtime Security Middleware. This document is self-contained — it can be used to start a fresh chat focused only on implementing this mechanism.

---

## 1. The Problem

The Runtime Security Middleware needs to detect adversarial patches by monitoring internal behavioral signals of a YOLO object detector, without being tightly coupled to the detector's specific architecture. The obvious tap point — the PANet neck output — has proven unreliable for accurate, model-agnostic patch localization.

The specific failure mode:

- **PANet performs multi-scale fusion.** The neck merges information top-down and bottom-up across P3/P4/P5 feature maps at different resolutions. A single spatial location in the *fused* output is not the result of one clean receptive field — it's a blend of signal from multiple depths and scales in the network. This breaks the core assumption almost all published latent-space adversarial-patch detectors rely on: that a spatial location in a feature map corresponds to a well-defined region of the input image.
- **This directly hurts localization accuracy.** Even when the anomaly detector correctly flags "something is wrong here," the mapping back to precise input-pixel coordinates is muddied by the fusion, producing masks that are too diffuse or misaligned — which conflicts with your requirement that the patch localization be accurate without covering other areas.

Almost all the reference literature you've reviewed (Mahalanobis-distance segment methods, PCA-based methods, feature-norm-clipping methods, DPG's feature-subspace approach) was built and validated on **plain CNN classifiers** (ResNet family), which have exactly one feature-extraction pathway per depth, with no cross-scale fusion. To use these methods' underlying statistical machinery successfully on YOLO, you need a tap point that behaves the same way.

---

## 2. The Suggested Mechanism

### 2.1 Core idea

Skip the neck entirely. Tap the **backbone** (CSPDarknet in YOLOv5/v8, or the equivalent CSPNet-derived backbone in later versions) at the points where it hands off its C3/C4/C5-equivalent feature maps **to** the neck — i.e., *before* any top-down/bottom-up fusion happens.

```
Input image
   │
   ▼
Backbone stage 1 (stem)
   │
   ▼
Backbone stage 2 ──────► feature map @ stride 8   (P3 input to neck)
   │
   ▼
Backbone stage 3 ──────► feature map @ stride 16  (P4 input to neck)
   │
   ▼
Backbone stage 4 ──────► feature map @ stride 32  (P5 input to neck)
   │
   ▼
   [ PANet neck: fusion happens here — we tap BEFORE this point ]
```

Each of these backbone-stage outputs is produced by a **single, unbranched convolutional pathway** — no fusion with other scales has occurred yet. A feature vector at spatial location `(y, x)` in the stride-8 backbone output corresponds to one well-defined receptive field in the input image, shaped only by the stacked convolutions of the backbone up to that stage. This is architecturally the closest match to the ResNet-classifier setting your reference papers assume, while still being genuinely "internal model behavior" (satisfying the runtime-monitoring framing of your research, unlike an external, model-independent approach).

### 2.2 Choosing which stage to tap

There's a resolution/semantics trade-off across the three candidate stages:

| Stage | Stride | Spatial resolution (640px input) | Characteristics |
|---|---|---|---|
| Backbone stage feeding P3 | 8 | 80×80 | High resolution → fine-grained localization of small patches. Lower-level features, closer to texture/edge information — more prone to false positives on naturally high-texture backgrounds (this is the exact failure mode Bunzel et al.'s entropy-based method reports on grass/bushes). |
| Backbone stage feeding P4 | 16 | 40×40 | Balanced resolution and semantic abstraction. |
| Backbone stage feeding P5 | 32 | 20×20 | More semantic/abstract features, but coarse grid — a small patch may not even occupy one full cell, hurting localization precision. |

**Recommendation:** start with the **P3-stride backbone stage** since your patch-localization requirement favors resolution, but explicitly plan to test the natural-texture false-positive risk against KITTI/BDD100K backgrounds (foliage, road texture, cluttered urban scenes) as part of validation — this is a known, well-documented failure mode in the literature, not a hypothetical.

### 2.3 Reducing false positives via multi-stage ensembling

Rather than committing to a single stage, you can require **agreement across stages** before flagging a location as anomalous — conceptually similar to Saliuitl's ensemble-across-thresholds idea, but applied across backbone depths instead of saliency thresholds:

- Compute a per-cell anomaly score at the P3-stride stage (fine resolution).
- Compute a per-cell anomaly score at the P4-stride stage (coarser resolution, but each cell there overlaps 4 P3 cells).
- Flag a region only if **both** stages register an anomaly in spatially corresponding areas.

This directly targets the texture false-positive problem: natural high-texture regions (grass, foliage) tend to produce noisy, spatially inconsistent anomalies across resolutions, while an actual adversarial patch — a coherent, deliberately placed object — tends to register consistently at multiple scales.

Implement the single-stage version first and validate it; add the ensemble only if false-positive rate on clean natural-texture images is unacceptably high.

---

## 3. Code Structure

Like Mechanism 1, this fits into your existing Adapter/Strategy architecture — same interfaces, different tap point, and the anomaly-detection Strategy classes (Mahalanobis, PCA) can largely be reused unchanged, since the input shape contract (`[B, C, H, W]` per stage) is the same.

### 3.1 New Adapter: backbone stage extraction

```python
class IActivationProvider(ABC):
    @abstractmethod
    def get_activations(self, image: Tensor) -> Dict[str, Tensor]:
        """Returns a dict keyed by stage name (e.g. 'P3','P4','P5')
        mapping to the backbone's pre-fusion feature map for that stage."""
        ...


# Per-version index maps: which backbone layer index corresponds to
# each P3/P4/P5 hand-off point, before it enters the neck's concat/
# upsample operations. These indices come from inspecting each
# version's model definition (yaml config or printed module list).
BACKBONE_STAGE_INDICES = {
    "yolov8": {"P3": 4, "P4": 6, "P5": 9},
    "yolov5": {"P3": 4, "P4": 6, "P5": 9},   # verify against your checkpoint
    "yolov10": {"P3": 4, "P4": 6, "P5": 9},  # verify against your checkpoint
}


class BackboneStageAdapter(IActivationProvider):
    """Model-agnostic access to pre-fusion backbone feature maps,
    driven by a per-version index config rather than hardcoded layer
    references — this is what keeps the Adapter swappable across
    YOLO versions without touching the anomaly-detection Strategy."""

    def __init__(self, model: nn.Module, version_key: str,
                 stages: List[str] = ("P3", "P4", "P5")):
        self.model = model
        self.stages = stages
        self.stage_indices = BACKBONE_STAGE_INDICES[version_key]
        self._activations: Dict[str, Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for stage in self.stages:
            layer_idx = self.stage_indices[stage]
            target_layer = self.model.model[layer_idx]
            handle = target_layer.register_forward_hook(self._make_hook(stage))
            self._hooks.append(handle)

    def _make_hook(self, stage: str):
        def hook(module, input, output):
            self._activations[stage] = output.detach()
        return hook

    def get_activations(self, image: Tensor) -> Dict[str, Tensor]:
        self._activations.clear()
        with torch.no_grad():
            self.model(image)
        return dict(self._activations)

    def teardown(self):
        for h in self._hooks:
            h.remove()
```

> Verifying the exact layer indices per version is a required, one-time manual step (Step 1 in the guide below) — `BACKBONE_STAGE_INDICES` above is illustrative, not guaranteed correct for your checkpoints. Confirm with `print(model.model)` and cross-reference against the architecture's yaml definition (Ultralytics ships these under `ultralytics/cfg/models/`).

### 3.2 Reusing your existing anomaly-detection Strategy

```python
class BackboneMahalanobisDetector(IAnomalyDetector):
    """Same statistical machinery as your existing MahalanobisDetector —
    only the input source (backbone stage vs. neck output) differs."""

    def __init__(self, calibration_stats: GlobalStats, threshold: float,
                 stage: str = "P3", stride: int = 8):
        self.stats = calibration_stats
        self.threshold = threshold
        self.stage = stage
        self.stride = stride

    def detect(self, activations: Dict[str, Tensor]) -> AnomalyResult:
        feat = activations[self.stage]          # [1, C, H, W]
        scores = self._mahalanobis_map(feat, self.stats)   # [H, W]
        mask = scores > self.threshold
        bbox = self._extract_tight_bbox(mask, self.stride)
        return AnomalyResult(
            is_attack=mask.any().item(),
            score=scores.max().item(),
            bounding_box=bbox,
        )
```

### 3.3 Multi-stage ensemble aggregator (optional, add after single-stage validation)

```python
class MultiStageEnsembleDetector(IAnomalyDetector):
    """Wraps two single-stage detectors and requires spatial agreement
    before flagging a region — reduces false positives from
    natural high-texture backgrounds."""

    def __init__(self, fine_detector: IAnomalyDetector,
                 coarse_detector: IAnomalyDetector,
                 overlap_threshold: float = 0.5):
        self.fine = fine_detector
        self.coarse = coarse_detector
        self.overlap_threshold = overlap_threshold

    def detect(self, activations: Dict[str, Tensor]) -> AnomalyResult:
        fine_result = self.fine.detect(activations)
        coarse_result = self.coarse.detect(activations)

        if not (fine_result.is_attack and coarse_result.is_attack):
            return AnomalyResult(is_attack=False, score=0.0, bounding_box=None)

        iou = self._bbox_iou(fine_result.bounding_box, coarse_result.bounding_box)
        if iou < self.overlap_threshold:
            return AnomalyResult(is_attack=False, score=0.0, bounding_box=None)

        # Agreement confirmed — return the finer-resolution bbox for precision
        return fine_result
```

### 3.4 Offline calibration pipeline

Identical structure to Mechanism 1's calibration step, but simpler — since backbone features aren't tied to a specific predicted class the way the classification branch is, you likely want a **global** (or coarse category-conditioned, e.g. "vehicle-like" vs "person-like") reference distribution rather than a full per-class one:

```python
def build_backbone_calibration_stats(clean_dataset, adapter: IActivationProvider,
                                      stage: str = "P3") -> GlobalStats:
    all_features = []
    for image in clean_dataset:
        activations = adapter.get_activations(image)
        feat = activations[stage]                    # [1, C, H, W]
        flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        all_features.append(flat)

    all_features = torch.cat(all_features, dim=0)
    mean = all_features.mean(dim=0)
    cov = torch.cov(all_features.T) + 1e-6 * torch.eye(all_features.shape[1])
    return GlobalStats(mean=mean, inv_cov=torch.linalg.inv(cov))
```

---

## 4. Step-by-Step Implementation Guide

1. **Identify exact backbone hand-off points.** For each YOLO version you need to support, print the model's module list and cross-reference against the architecture's config (yaml) to find the exact layer index where each P3/P4/P5-equivalent backbone output is produced, *before* it's consumed by the neck's concat/upsample blocks. Record this in a version-keyed config (as in `BACKBONE_STAGE_INDICES`).
2. **Implement the adapter.** Write `BackboneStageAdapter` implementing `IActivationProvider`, driven by the per-version index config so a new YOLO version only requires a new config entry, not new code.
3. **Pick your starting stage.** Begin with the P3-stride stage (finest resolution) for localization precision, accepting the known trade-off of higher false-positive risk on natural high-texture backgrounds.
4. **Build the calibration pipeline.** Run the adapter over a clean image set (KITTI + BDD100K, diverse conditions) and fit a global (or coarse-category-conditioned) reference distribution for the chosen stage. Persist the resulting statistics.
5. **Implement the detector.** Reuse or lightly adapt your existing `MahalanobisDetector`/`PCADetector` Strategy implementation to consume the backbone-stage activations and calibration stats.
6. **Implement mask → bounding box extraction.** Threshold the per-cell anomaly score map, extract the largest connected component, compute a tight bounding box, and scale to pixel coordinates using the stage's stride.
7. **Validate on clean natural-texture images first.** Before touching adversarial data, explicitly test against clean KITTI/BDD100K images with foliage, cluttered backgrounds, and complex urban textures to measure the false-positive rate this mechanism is most exposed to.
8. **Validate on adversarial patches.** Run against ART/EoT-generated physical patches; measure TPR/FPR and bounding-box IoU against ground-truth patch regions.
9. **If false-positive rate on natural textures is too high, implement the multi-stage ensemble.** Add a second detector at the P4 stage and require spatial agreement (as in `MultiStageEnsembleDetector`) before finalizing a flag.
10. **Compare against Mechanism 1 (decoupled-head tap) on the same test set.** Since both mechanisms address the same root problem from different angles, running them side-by-side on identical data gives you a concrete, citable comparison for the thesis — worth keeping this as a planned experiment from the start rather than an afterthought.
