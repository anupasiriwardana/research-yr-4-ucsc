# Mechanism 1: Decoupled-Head Classification-Branch Tap

## Status
Candidate detection mechanism for the Anomaly Detection module (Module 1) of the Runtime Security Middleware. This document is self-contained — it can be used to start a fresh chat focused only on implementing this mechanism.

---

## 1. The Problem

The Runtime Security Middleware needs to detect adversarial patches by monitoring internal behavioral signals of a YOLO object detector, without being tightly coupled to the detector's specific architecture (the core SE gap the research addresses). The natural place to tap signals is the PANet neck, since that is the richest fused representation available before the detection heads.

This has proven to hit a wall in practice, for a structural reason:

- **PANet performs multi-scale fusion.** The neck combines top-down and bottom-up pathways across P3/P4/P5, so a single spatial location in the fused feature map is a blend of receptive fields from different depths and resolutions. An anomaly detector operating on this fused tensor cannot cleanly attribute an outlier score to one region of the input image the way it could on a plain CNN.
- **The fused feature map feeds both classification and box regression.** In many existing academic defenses (built on ResNet-style image classifiers), a feature vector at a spatial location encodes only "what object is here." In YOLO's neck output, the same tensor is the input to both the class-prediction branch and the box-regression branch. A large-magnitude activation might mean "this looks like an adversarial patch," or it might just mean "this is a very confidently and precisely localized real bounding-box edge." The anomaly detector cannot currently distinguish these two cases, producing both localization noise and false positives on legitimate high-confidence detections.

The result: patch localization from neck-level activations is imprecise (too diffuse, sometimes covering real objects) and unreliable (flags legitimate strong detections).

---

## 2. The Suggested Mechanism

### 2.1 Core idea

Modern YOLO architectures (YOLOv8, YOLOv9, YOLOv10, YOLOv11 — essentially everything since Ultralytics introduced the anchor-free decoupled head) do **not** predict class and box from a single shared tensor at the final stage. Instead, the head splits into two **parallel, independent convolutional branches** immediately after receiving the neck's fused feature map:

```
Neck output (fused P3/P4/P5 feature map, per scale)
        │
        ├──► Classification branch (conv → BN → SiLU → conv → BN → SiLU → conv[nc channels])
        │
        └──► Regression branch (conv → BN → SiLU → conv → BN → SiLU → conv[4*reg_max channels])
```

These branches share the same *input* tensor (the neck output) but apply **separate learned weights** and produce separate intermediate feature maps before the two final 1×1 prediction convolutions. The classification branch's intermediate activations (the output of its second conv block, *before* the final 1×1 class-logit conv) are shaped purely by the objective of "what class is this," with no gradient signal from box regression shaping them.

This is architecturally the closest thing YOLO has to the plain-CNN classifier setting that most of the published latent-space adversarial-patch literature assumes (Mahalanobis-distance approaches, PCA-based approaches, feature-norm approaches). Tapping here should:

1. Remove the box-regression contamination that causes false positives on real, well-localized objects.
2. Preserve a well-defined, single-scale spatial grid (no cross-scale fusion within this branch itself — fusion already happened upstream in the neck, but the classification branch does not re-mix across P3/P4/P5).
3. Stay model-agnostic across the whole decoupled-head YOLO family, since the branch structure is stable in Ultralytics' `Detect` module implementation across v8–v11 (only exact channel counts and internal layer counts vary by model scale, e.g. YOLOv8n vs YOLOv8x).

### 2.2 Where exactly to hook

In the Ultralytics implementation, the detection head module (commonly `Detect`, or `v10Detect` / `Detect` variants depending on version) holds, per detection scale `i` (typically 3 scales: stride 8, 16, 32):

- `self.cv2[i]` — the regression branch (box/DFL), producing `4 * reg_max` channels.
- `self.cv3[i]` — the classification branch, producing `nc` channels (number of classes).

Each of `cv2[i]` / `cv3[i]` is itself a small `nn.Sequential` of 2–3 conv blocks. You want to hook the **output of the last conv block before the final 1×1 projection to `nc` channels** — i.e., the penultimate layer of `cv3[i]` — not the raw class logits themselves. The intermediate hidden representation carries much richer information for anomaly detection than the final low-dimensional logit vector.

### 2.3 Handling multiple scales

Because P3/P4/P5 each have their own classification branch and their own spatial grid resolution (e.g. 80×80, 40×40, 20×20 for a 640×640 input), a patch may show up more clearly at one scale than another depending on its physical size relative to the camera. Two viable strategies:

- **Run the anomaly detector independently per scale** and produce one `AnomalyResult` candidate per scale, then merge (e.g., take the highest-confidence non-overlapping detections, or require agreement across at least 2 of 3 scales to reduce false positives).
- **Pick the scale whose stride best matches expected patch size** (small patches → P3/stride-8; large patches on nearby objects → P4/P5) as a first, simpler implementation, and generalize to multi-scale fusion later once the single-scale pipeline is validated.

Start with the second option for a working baseline, then evaluate the first.

### 2.4 Handling the "legitimate strong detection" false-positive risk

Even after removing box-regression contamination, a very confidently detected real object will still produce strong classification-branch activations. To avoid flagging these as attacks:

- Build your reference statistics **conditioned on predicted class**, not as one global distribution. A "car" activation profile and a "pedestrian" activation profile are legitimately different; comparing both against one pooled Mahalanobis distribution will inflate false positives on rarer-but-legitimate classes.
- Use a sufficiently large and diverse **clean calibration set** (draw from KITTI and BDD100K clean subsets, across weather/lighting conditions) so that "confident but legitimate" activation patterns are well represented in the reference distribution.

---

## 3. Code Structure

This mechanism plugs directly into your existing Adapter (model-agnostic activation extraction) and Strategy (interchangeable anomaly detectors) patterns — it changes *what* is tapped, not the interfaces around it.

### 3.1 New Adapter: `IActivationProvider` implementation

```python
class IActivationProvider(ABC):
    @abstractmethod
    def get_activations(self, image: Tensor) -> Dict[str, Tensor]:
        """Returns a dict keyed by scale name (e.g. 'P3','P4','P5')
        mapping to the classification-branch intermediate activation
        tensor of shape [B, C_hidden, H, W]."""
        ...


class YOLOv8ClsHeadAdapter(IActivationProvider):
    """Model-agnostic access to the classification-branch activations
    of an Ultralytics decoupled detection head."""

    def __init__(self, model: nn.Module, scales: List[str] = ("P3", "P4", "P5")):
        self.model = model
        self.scales = scales
        self._activations: Dict[str, Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        detect_module = self._find_detect_module(self.model)
        for i, scale in enumerate(self.scales):
            # cv3[i] is the classification branch for scale i.
            # Hook the penultimate block (index -2), not the final
            # 1x1 projection to nc channels.
            branch = detect_module.cv3[i]
            target_layer = branch[-2]  # second-to-last conv block
            handle = target_layer.register_forward_hook(
                self._make_hook(scale)
            )
            self._hooks.append(handle)

    def _make_hook(self, scale: str):
        def hook(module, input, output):
            self._activations[scale] = output.detach()
        return hook

    def _find_detect_module(self, model: nn.Module) -> nn.Module:
        # Ultralytics stores the head as the last module of model.model
        return model.model[-1]

    def get_activations(self, image: Tensor) -> Dict[str, Tensor]:
        self._activations.clear()
        with torch.no_grad():
            self.model(image)
        return dict(self._activations)

    def teardown(self):
        for h in self._hooks:
            h.remove()
```

> Note: exact attribute names (`cv2`, `cv3`, `model.model[-1]`) are correct for Ultralytics YOLOv8; verify against `print(model.model[-1])` for the specific version/checkpoint you're using, since internal naming has shifted slightly across v8 → v10 → v11 releases (e.g. `v10Detect` in YOLOv10 removes NMS-related branches but keeps the same cv2/cv3 split). Build one adapter subclass per major version, all implementing the same `IActivationProvider` interface — this is exactly the Adapter pattern you already have for `YOLOv5Adapter` / `YOLOv8Adapter`.

### 3.2 New Strategy: `ClsHeadAnomalyDetector`

```python
class ClsHeadMahalanobisDetector(IAnomalyDetector):
    """Per-class-conditioned Mahalanobis distance over classification-
    branch activations, per grid cell."""

    def __init__(self, calibration_stats: Dict[str, ClassConditionedStats],
                 threshold: float, scale: str = "P3", stride: int = 8):
        self.stats = calibration_stats   # loaded from calibration step
        self.threshold = threshold
        self.scale = scale
        self.stride = stride

    def detect(self, activations: Dict[str, Tensor],
               predicted_classes: Tensor) -> AnomalyResult:
        feat = activations[self.scale]           # [1, C, H, W]
        B, C, H, W = feat.shape
        feat_flat = feat.permute(0, 2, 3, 1).reshape(H * W, C)  # [HW, C]

        scores = torch.zeros(H, W)
        for idx in range(H * W):
            y, x = divmod(idx, W)
            cls_id = predicted_classes[y, x].item()
            mean, inv_cov = self.stats[cls_id].mean, self.stats[cls_id].inv_cov
            diff = feat_flat[idx] - mean
            scores[y, x] = (diff @ inv_cov @ diff).sqrt()

        mask = (scores > self.threshold)
        bbox = self._extract_tight_bbox(mask, self.stride)
        return AnomalyResult(
            is_attack=mask.any().item(),
            score=scores.max().item(),
            bounding_box=bbox,
        )

    def _extract_tight_bbox(self, mask: Tensor, stride: int) -> Tuple[int, int, int, int]:
        # Largest connected component -> tight bbox in grid coords -> scale by stride
        ...
```

### 3.3 Offline calibration pipeline (separate script, not part of the runtime middleware)

```python
def build_calibration_stats(clean_dataset, adapter: IActivationProvider,
                             scale: str = "P3") -> Dict[int, ClassConditionedStats]:
    """Run once, offline, over a clean image set. Persist the resulting
    stats (e.g. as .npz or pickle) and load them at runtime."""
    per_class_features: Dict[int, List[Tensor]] = defaultdict(list)

    for image, predictions in clean_dataset:
        activations = adapter.get_activations(image)
        feat = activations[scale]
        for (cls_id, y, x) in predictions:  # from clean-run detector output
            per_class_features[cls_id].append(feat[0, :, y, x])

    stats = {}
    for cls_id, feats in per_class_features.items():
        feats_tensor = torch.stack(feats)
        mean = feats_tensor.mean(dim=0)
        cov = torch.cov(feats_tensor.T) + 1e-6 * torch.eye(feats_tensor.shape[1])
        stats[cls_id] = ClassConditionedStats(mean=mean, inv_cov=torch.linalg.inv(cov))
    return stats
```

### 3.4 Grid-to-pixel coordinate mapping

For scale P3 with stride 8: a grid cell `(y, x)` corresponds to an image region roughly centered at `(x * 8 + 4, y * 8 + 4)`, but the *effective* receptive field is larger than the stride due to accumulated backbone+neck depth. For a tight bounding box, use the stride-based grid mapping for the coarse region, then optionally refine with a lightweight boundary-tightening pass (connected-component bounding box on the thresholded mask is usually sufficient for a first version; see the shared refinement note below).

---

## 4. Step-by-Step Implementation Guide

1. **Inspect the target model.** Load your YOLO checkpoint and print `model.model[-1]` to confirm the exact attribute names (`cv2`/`cv3`) and the number of conv blocks inside each branch for your specific version and model scale (n/s/m/l/x).
2. **Implement the adapter.** Write `YOLOv8ClsHeadAdapter` (and a corresponding one for any other version you need, e.g. `YOLOv10ClsHeadAdapter`) implementing `IActivationProvider`, hooking the penultimate conv block of `cv3[i]` for your chosen scale(s).
3. **Pick your starting scale.** Begin with P3 (stride 8, finest grid) since it gives the best localization resolution for small patches; expand to multi-scale fusion later.
4. **Build the calibration pipeline.** Run the adapter over a clean image subset (KITTI + BDD100K, diverse conditions), collect per-class classification-branch feature vectors at the predicted-object grid cells, and fit per-class mean/covariance. Persist this artifact (e.g. `cls_head_calibration_p3.npz`).
5. **Implement the detector.** Write `ClsHeadMahalanobisDetector` implementing `IAnomalyDetector`, loading the calibration artifact and computing per-cell Mahalanobis distance against the predicted class's distribution.
6. **Implement the mask → bounding box pipeline.** Threshold the per-cell score map, take the largest connected component, compute its tight bounding box in grid coordinates, and scale by stride to get pixel coordinates.
7. **Wrap the output in `AnomalyResult`.** Ensure the interface contract (`is_attack`, `score`, `bounding_box`) is satisfied so the Recovery Engine can consume it without any knowledge of this detection mechanism.
8. **Validate on an adversarial dataset.** Run against ART/EoT-generated physical patches on KITTI/BDD100K test images. Measure TPR/FPR and, separately, the **IoU between the predicted bounding box and the ground-truth patch region** — this is your accuracy metric for the localization requirement.
9. **Diagnose false positives on clean images specifically.** Since this mechanism's main promise is fixing "flagged real detections," explicitly test against clean, high-confidence detections (large, close, unoccluded objects) and confirm the per-class conditioning actually suppresses false positives there.
10. **Tune and iterate.** Adjust the threshold per class if needed (some classes may have tighter/looser natural variance), and consider extending to multi-scale agreement (step 2.3, option 1) if single-scale P3 proves too noisy or misses larger patches.
