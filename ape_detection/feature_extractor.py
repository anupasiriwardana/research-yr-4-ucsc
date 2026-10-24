import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, Optional
from collections import OrderedDict


class FeatureExtractor:
    """
    Universal feature extraction adapter compatible with:
      - Ultralytics YOLO models (YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLO11, etc.)
      - torchvision detection models (Faster R-CNN, Mask R-CNN, RetinaNet, ...)
      - Standard PyTorch nn.Module architectures (YOLOv7, custom detectors)

    Renamed from YOLOFeatureExtractor -- nothing about this class was ever
    YOLO-specific; the old name is kept as an alias at the bottom of this
    file so existing imports keep working.
    """

    def __init__(
        self,
        model: Union[nn.Module, Any],
        layers: Union[str, int, List[Union[str, int]]],
    ):
        # 1. Automatically unwrap Ultralytics YOLO wrapper if passed directly
        if hasattr(model, "model") and isinstance(model.model, nn.Module):
            self.raw_model = model.model
        elif isinstance(model, nn.Module):
            self.raw_model = model
        else:
            raise TypeError("Expected a PyTorch nn.Module or an Ultralytics YOLO instance.")

        if isinstance(layers, (str, int)):
            layers = [layers]
        self.requested_layers = layers

        self.activations: Dict[str, Any] = OrderedDict()
        self._hooks = []
        self._register_hooks()

    def _resolve_layer(self, layer_id: Union[str, int]) -> tuple[str, nn.Module]:
        """Maps an integer index or dot-separated string name to an nn.Module.

        The string path (named_modules / get_submodule) is architecture-agnostic:
        it works identically whether the model is an Ultralytics YOLO's
        `model.2` or a torchvision Faster R-CNN's `backbone.body.layer1` --
        both are just dotted attribute paths on the underlying nn.Module tree.
        """
        # Case A: Integer index (e.g., layer 15 or layer -1)
        if isinstance(layer_id, int):
            # Ultralytics models store sequential blocks inside self.raw_model.model
            if hasattr(self.raw_model, "model") and isinstance(self.raw_model.model, (nn.Sequential, nn.ModuleList)):
                resolved_name = f"model.{layer_id if layer_id >= 0 else len(self.raw_model.model) + layer_id}"
                return resolved_name, self.raw_model.model[layer_id]
            elif isinstance(self.raw_model, (nn.Sequential, nn.ModuleList)):
                resolved_name = str(layer_id if layer_id >= 0 else len(self.raw_model) + layer_id)
                return resolved_name, self.raw_model[layer_id]
            else:
                raise ValueError(
                    f"Cannot resolve integer index '{layer_id}' on a non-sequential model "
                    f"(e.g. a torchvision R-CNN). Pass a dotted string path instead, "
                    f"such as 'backbone.body.layer1'."
                )

        # Case B: String path (e.g., "model.22.cv3.0" or "backbone.body.layer1")
        clean_name = layer_id.strip(".")
        named_modules = dict(self.raw_model.named_modules())

        if clean_name in named_modules:
            return clean_name, named_modules[clean_name]

        # Try resolving via PyTorch's native get_submodule
        try:
            return clean_name, self.raw_model.get_submodule(clean_name)
        except (AttributeError, KeyError):
            pass

        raise ValueError(
            f"Layer '{layer_id}' not found. Call .list_layers() to inspect available names."
        )

    def _register_hooks(self):
        for target in self.requested_layers:
            resolved_name, module = self._resolve_layer(target)
            handle = module.register_forward_hook(self._make_hook(resolved_name))
            self._hooks.append(handle)

    def _make_hook(self, key: str):
        def hook(module, input, output):
            # Handle standard Tensors (the common case: YOLO blocks, a single
            # ResNet stage such as backbone.body.layer1, etc.)
            if isinstance(output, torch.Tensor):
                self.activations[key] = output.detach().clone()
            # Handle FPN-style backbones, whose .backbone or .fpn submodules
            # return an OrderedDict[str, Tensor] -- one tensor per pyramid level.
            # Each tensor is individually detached/cloned, same as the Tensor
            # case above, so no computation graph is accidentally retained.
            elif isinstance(output, dict):
                self.activations[key] = {
                    k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                    for k, v in output.items()
                }
            # Handle layers returning multi-scale features or (preds, features) tuples
            elif isinstance(output, (tuple, list)):
                self.activations[key] = [
                    x.detach().clone() if isinstance(x, torch.Tensor) else x
                    for x in output
                ]
            else:
                self.activations[key] = output
        return hook

    def list_layers(self, max_depth: Optional[int] = None) -> List[str]:
        """Lists all queryable module names to find exactly what to target."""
        valid_names = []
        for name, _ in self.raw_model.named_modules():
            if not name:
                continue
            if max_depth is not None and name.count(".") >= max_depth:
                continue
            valid_names.append(name)
        return valid_names

    def extract(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Executes a forward pass and returns intermediate activations."""
        self.activations.clear()
        with torch.no_grad():
            self.raw_model(tensor)
        return dict(self.activations)

    def teardown(self):
        """Removes all PyTorch forward hooks to restore normal execution."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()

    def __call__(self, tensor: torch.Tensor) -> Dict[str, Any]:
        return self.extract(tensor)


# Backward-compatible alias -- this class was never actually YOLO-specific,
# but code written against the old name keeps working unchanged.
YOLOFeatureExtractor = FeatureExtractor
