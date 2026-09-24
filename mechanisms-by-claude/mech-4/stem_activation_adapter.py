"""
Model-agnostic activation provider for Mechanism 4 -- now supporting
MULTIPLE simultaneous tap layers, for the paper's actual multi-layer
aggregation design (see mech-4-feature-energy.md for why this changed
from the original single-stem-layer version).

Still the only file that needs to know anything about YOLOv8's module
structure. Swapping to a different model/version: write a new adapter
class with the same get_activations() -> {layer_idx: Tensor} contract.
"""

import torch
import torch.nn as nn
from typing import Dict, List


class YOLOv8StemAdapter:
    def __init__(self, model: nn.Module, layer_indices: List[int]):
        """layer_indices: which backbone layer indices to hook,
        e.g. [3] for a single tap, or [0, 1, 2, 3] for multi-layer
        aggregation. Use find_stem_layer_index.py-style inspection to
        pick these for your specific checkpoint -- see README.md."""
        self.model = model
        self.layer_indices = layer_indices
        self._activations: Dict[int, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for idx in self.layer_indices:
            target_layer = self.model.model[idx]
            handle = target_layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(handle)

    def _make_hook(self, idx: int):
        def hook(module, input, output):
            self._activations[idx] = output.detach()
        return hook

    def get_activations(self, image: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Returns {layer_idx: Tensor[1, C, H, W]} for every hooked
        layer. Keys are the layer indices themselves now (not a fixed
        'stem' string), since there can be more than one."""
        self._activations = {}
        with torch.no_grad():
            self.model(image)
        return dict(self._activations)

    def teardown(self):
        for h in self._hooks:
            h.remove()