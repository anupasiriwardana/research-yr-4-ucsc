import torch
import torch.nn as nn
from typing import Dict, List
from ultralytics import YOLO

class YOLOv8ClsHeadAdapter:
    def __init__(self, model: nn.Module, scales: List[str] = ["P3"]):
        self.model = model
        self.scales = scales
        self._activations: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # Ultralytics stores the detection head as the last module
        detect_module = self.model.model[-1]
        
        for i, scale in enumerate(self.scales):
            # cv3 is the classification branch for scale i
            branch = detect_module.cv3[i]
            
            # Hook the penultimate block (index -2) to capture rich hidden representations
            target_layer = branch[-2] 
            
            handle = target_layer.register_forward_hook(self._make_hook(scale))
            self._hooks.append(handle)

    def _make_hook(self, scale: str):
        def hook(module, input, output):
            # Detach to prevent VRAM accumulation on the RTX 2050
            self._activations[scale] = output.detach().clone()
        return hook

    def get_activations(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._activations.clear()
        with torch.no_grad():
            self.model(image)
        return dict(self._activations)

    def teardown(self):
        for h in self._hooks:
            h.remove()