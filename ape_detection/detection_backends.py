"""
Backend adapters so ApeDetector can drive different detector families behind
one common interface: load the model, preprocess a frame, run inference,
and parse detections into a plain (box, label, score) list.

Why this exists: YOLO (via ultralytics.YOLO) and torchvision's R-CNN family
(GeneralizedRCNN) differ in every one of these respects --
  - loading:        YOLO(path)                 vs  torchvision.models.detection.*(weights=...)
  - module tree:     model.model[idx]           vs  model.backbone.body.layer1 / .rpn / .roi_heads
  - input format:    one batched [1,C,H,W]      vs  a LIST of unbatched [C,H,W] tensors
  - output format:   Results.boxes.xyxy/.cls    vs  List[{"boxes","labels","scores"}]
  - class indices:   COCO-80 (no background)    vs  COCO-91 (index 0 = background)
None of this is specific to adversarial-patch detection -- it's just what's
needed to run either family's ordinary forward pass and read its ordinary
output. AdversarialPatchExtractor never sees any of it: it only ever
receives a plain 2-D Z-score array, regardless of which backend produced it.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class Detection:
    box: List[int]   # [x1, y1, x2, y2] in resized-frame pixel space
    label: str
    score: float


# ---------------------------------------------------------------------------
# YOLO (Ultralytics) backend -- the pipeline's original behaviour, unchanged
# ---------------------------------------------------------------------------
class YoloBackend:
    family = "yolo"
    default_layer = "model.2"   # first backbone block after the 2 stride-2 stem
                                  # convs -> stride 4, for a 1280x736 input that's 320x184

    def load(self, model_path: str, device, input_size: Tuple[int, int], **kwargs):
        from ultralytics import YOLO
        if not model_path:
            raise ValueError("model_family='yolo' requires a model_path (e.g. a .pt checkpoint).")
        return YOLO(model_path).to(device)

    def preprocess(self, resized_img_rgb: np.ndarray, device) -> torch.Tensor:
        return torch.from_numpy(resized_img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    def infer(self, model, model_input, target_classes: Optional[List[int]] = None):
        return model.predict(source=model_input, classes=target_classes, verbose=False)[0]

    def class_names(self, model) -> Dict[int, str]:
        return dict(model.names)

    def parse_detections(self, raw_output, class_names: Dict[int, str]) -> List[Detection]:
        dets = []
        boxes = getattr(raw_output, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return dets
        for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = box.int().tolist()
            dets.append(Detection([x1, y1, x2, y2], class_names.get(int(cls), str(int(cls))), float(conf)))
        return dets

    def compute_stride(self, input_width: int, input_height: int, feat_shape) -> int:
        # This pipeline's YOLO path does its own resize and feeds the model
        # that exact tensor with no further internal resizing, so the
        # configured input size and the captured feature map are directly
        # comparable.
        return max(1, input_width // feat_shape[-1])


# ---------------------------------------------------------------------------
# torchvision R-CNN-family backend (Faster R-CNN, Mask R-CNN, RetinaNet, ...)
# ---------------------------------------------------------------------------
class TorchvisionRCNNBackend:
    family = "torchvision_rcnn"
    # backbone.body.layer1 is a ResNet-FPN backbone's first stage *after* the
    # stem (conv1 stride-2 + maxpool stride-2): stride 4 overall, with no
    # further downsampling inside layer1 itself -- the direct structural
    # equivalent of YOLO's model.2. Confirmed against torchvision's own
    # backbone_utils.py (IntermediateLayerGetter wraps layer1..layer4 under
    # backbone.body; backbone.fpn is the separate FPN head).
    default_layer = "backbone.body.layer1"

    # architecture name -> (constructor function name, matching Weights enum name)
    _CONSTRUCTORS = {
        "fasterrcnn_resnet50_fpn": ("fasterrcnn_resnet50_fpn", "FasterRCNN_ResNet50_FPN_Weights"),
        "fasterrcnn_resnet50_fpn_v2": ("fasterrcnn_resnet50_fpn_v2", "FasterRCNN_ResNet50_FPN_V2_Weights"),
        "fasterrcnn_mobilenet_v3_large_fpn": ("fasterrcnn_mobilenet_v3_large_fpn", "FasterRCNN_MobileNet_V3_Large_FPN_Weights"),
        "retinanet_resnet50_fpn": ("retinanet_resnet50_fpn", "RetinaNet_ResNet50_FPN_Weights"),
        "maskrcnn_resnet50_fpn": ("maskrcnn_resnet50_fpn", "MaskRCNN_ResNet50_FPN_Weights"),
    }

    def __init__(self, score_thresh: float = 0.5):
        self._last_input_hw = None            # (H, W) actually fed to the backbone, read via hook
        self._weights_meta_categories = None  # populated only when using pretrained COCO weights
        # torchvision's own default (box_score_thresh=0.05, up to box_detections_per_img=100)
        # is tuned for mAP sweeps across all 80 classes, not for a clean visualization of a
        # few classes you actually care about. YOLO's .predict() defaults to conf=0.25 for
        # the same reason this needs its own threshold here.
        self.score_thresh = float(score_thresh)

    def load(self, model_path: Optional[str], device, input_size: Tuple[int, int], *,
             architecture: str = "fasterrcnn_resnet50_fpn", num_classes: Optional[int] = None, **kwargs):
        from torchvision.models import detection as tvd

        if architecture not in self._CONSTRUCTORS:
            raise ValueError(
                f"Unknown torchvision architecture '{architecture}'. Known: {list(self._CONSTRUCTORS)}"
            )
        ctor_name, weights_enum_name = self._CONSTRUCTORS[architecture]
        ctor = getattr(tvd, ctor_name)
        weights_enum = getattr(tvd, weights_enum_name)

        width, height = input_size
        # IMPORTANT: GeneralizedRCNNTransform rescales its input to fit a
        # min_size/max_size window AND pads it up to a multiple of 32, even
        # for a single image. Setting min_size/max_size to exactly the frame
        # size we already resized to makes that rescale a no-op (it only
        # pads), keeping the feature map's spatial scale predictable.
        common_kwargs = dict(min_size=min(height, width), max_size=max(height, width))

        if model_path:
            # Custom-trained checkpoint: build the architecture with random
            # (or ImageNet-backbone) init at the right head size, then load
            # the user's weights over it.
            if num_classes is None:
                raise ValueError(
                    "model_family='torchvision_rcnn' with a custom model_path requires "
                    "'num_classes' in config (including the background class)."
                )
            model = ctor(weights=None, num_classes=num_classes, **common_kwargs)
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
        else:
            # No checkpoint given -> use torchvision's own pretrained COCO weights.
            model = ctor(weights=weights_enum.DEFAULT, **common_kwargs)
            self._weights_meta_categories = weights_enum.DEFAULT.meta["categories"]

        model.to(device).eval()

        # Auxiliary hook purely to read the *actual* post-transform,
        # post-padding tensor size the backbone receives (see note above) --
        # this is what makes compute_stride() correct instead of guessed.
        def _capture_input_hw(module, inp, out):
            image_list = out[0]
            self._last_input_hw = tuple(image_list.tensors.shape[-2:])  # (H, W)
        model.transform.register_forward_hook(_capture_input_hw)

        return model

    def preprocess(self, resized_img_rgb: np.ndarray, device) -> List[torch.Tensor]:
        # torchvision detection models take a LIST of unbatched [C,H,W] float
        # tensors in [0,1] -- normalization (ImageNet mean/std) happens
        # inside the model's own GeneralizedRCNNTransform, so we must NOT
        # normalize here ourselves.
        t = torch.from_numpy(resized_img_rgb).permute(2, 0, 1).float().to(device) / 255.0
        return [t]

    def infer(self, model, model_input: List[torch.Tensor], target_classes: Optional[List[int]] = None):
        with torch.no_grad():
            out = model(model_input)[0]   # one image in -> take its (only) result dict

        boxes, labels, scores = out["boxes"], out["labels"], out["scores"]
        keep = scores >= self.score_thresh
        if target_classes is not None:
            target_tensor = torch.tensor(list(target_classes), device=labels.device, dtype=labels.dtype)
            keep = keep & torch.isin(labels, target_tensor)

        return {"boxes": boxes[keep], "labels": labels[keep], "scores": scores[keep]}

    def class_names(self, model) -> Dict[int, str]:
        if self._weights_meta_categories:
            return {i: name for i, name in enumerate(self._weights_meta_categories)}
        return {}   # custom checkpoint with no bundled category list

    def parse_detections(self, raw_output: Dict[str, torch.Tensor], class_names: Dict[int, str]) -> List[Detection]:
        dets = []
        boxes = raw_output["boxes"].detach().cpu().tolist()
        labels = raw_output["labels"].detach().cpu().tolist()
        scores = raw_output["scores"].detach().cpu().tolist()
        for box, lbl, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = (int(v) for v in box)
            dets.append(Detection([x1, y1, x2, y2], class_names.get(lbl, str(lbl)), float(score)))
        return dets

    def compute_stride(self, input_width: int, input_height: int, feat_shape) -> int:
        if self._last_input_hw is not None:
            _actual_h, actual_w = self._last_input_hw
            return max(1, actual_w // feat_shape[-1])
        return max(1, input_width // feat_shape[-1])  # fallback; shouldn't normally trigger


def get_backend(model_family: str, **kwargs):
    if model_family == "yolo":
        return YoloBackend()   # thresholding is ultralytics' own .predict() default (conf=0.25)
    if model_family == "torchvision_rcnn":
        return TorchvisionRCNNBackend(**kwargs)
    raise ValueError(f"Unknown model_family '{model_family}'. Expected 'yolo' or 'torchvision_rcnn'.")