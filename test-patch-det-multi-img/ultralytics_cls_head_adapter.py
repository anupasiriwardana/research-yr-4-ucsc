"""
Classification-branch adapter for Ultralytics decoupled-head detectors.

One contract for every layout below, which is what lets
ClsHeadMahalanobisDetector stay untouched across models:

    get_activations(image) -> {"P3": Tensor[1, C, H, W]}

Head layouts this adapter handles (read from ultralytics 8.4.164,
nn/modules/head.py and nn/tasks.py -- re-check with describe() on YOUR
installed version):

  * YOLOv8 / YOLOv9  ("legacy" head)
        cv3[i] = Sequential(Conv, Conv, Conv2d)
  * YOLO11 / YOLO12  (depthwise head; set whenever the model contains
                      C3k2 / A2C2f blocks)
        cv3[i] = Sequential(Sequential(DWConv, Conv),
                            Sequential(DWConv, Conv),
                            Conv2d)
  * YOLO26           (end-to-end head, reg_max=1)
        same depthwise blocks, BUT the branch that produces the final
        detections is `one2one_cv3`, and Detect.fuse() sets `cv3` to
        None for inference. Hooking `cv3` alone would tap a branch that
        is either deleted or not the one feeding the output.

In every layout the tapped block is the PENULTIMATE child of the branch:
the last feature-producing block before the 1x1 projection to class
logits. So the tensor means the same thing across models (post-
activation hidden classification features), even though the block's
internals differ (plain 3x3 Conv vs. depthwise + pointwise).

Design choices worth knowing about:
  * Hooks are registered on EVERY classification branch that exists at
    construction (cv3 and/or one2one_cv3). The branch actually used is
    chosen at get_activations() time by mirroring the head's own rule
    (`head.end2end` -> one-to-one branch, else one-to-many). This makes
    the adapter immune to WHEN Ultralytics fuses the model (the first
    .predict() call fuses in place, after this adapter is built).
  * Failure is loud. If the expected hook did not fire, or a branch has
    an unexpected structure, a RuntimeError says what was found --
    instead of an empty dict that later surfaces as KeyError: 'P3'.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

_BRANCH_ATTRS = ("cv3", "one2one_cv3")  # one-to-many / one-to-one classification branches


class UltralyticsClsHeadAdapter:
    def __init__(self, model: nn.Module, scales: List[str] = ("P3",)):
        """model: the underlying nn.Module, i.e. `YOLO(...).model`.
        scales: names for detection levels, in order (P3, P4, P5);
                index i selects branch[i]. Mechanism 1 uses only "P3"."""
        self.model = model
        self.scales = list(scales)
        self._head = model.model[-1]
        self._activations: Dict[Tuple[str, str], torch.Tensor] = {}
        self._tapped: Dict[Tuple[str, str], nn.Module] = {}
        self._hooks = []
        self._register_hooks()

    # ------------------------------------------------------------------ setup
    def _register_hooks(self):
        registered = False
        for attr in _BRANCH_ATTRS:
            branch_list = getattr(self._head, attr, None)
            if branch_list is None:  # e.g. cv3 after end-to-end fusion
                continue
            for i, scale in enumerate(self.scales):
                branch = branch_list[i]
                if not isinstance(branch, nn.Sequential) or len(branch) < 3:
                    raise RuntimeError(
                        f"head.{attr}[{i}] is {branch.__class__.__name__} with "
                        f"{len(branch) if hasattr(branch, '__len__') else '?'} children; expected an "
                        f"nn.Sequential of >= 3 blocks (feature blocks..., final 1x1 class projection). "
                        f"This adapter does not know this head layout -- write a new adapter for it."
                    )
                target = branch[-2]  # penultimate block: last features before class logits
                key = (scale, attr)
                self._hooks.append(target.register_forward_hook(self._make_hook(key)))
                self._tapped[key] = target
                registered = True

        if not registered:
            raise RuntimeError(
                f"Detect head {self._head.__class__.__name__} has neither `cv3` nor `one2one_cv3` "
                f"(found attributes: {sorted(k for k in vars(self._head) if not k.startswith('_'))[:12]}...). "
                f"This is not a decoupled-head layout this adapter supports."
            )

    def _make_hook(self, key: Tuple[str, str]):
        def hook(module, input, output):
            self._activations[key] = output.detach().clone()
        return hook

    # ---------------------------------------------------------------- runtime
    def _active_branch(self) -> str:
        """Mirror Detect.forward(): end-to-end heads feed the output from
        the one-to-one branch, all others from the one-to-many branch.
        Evaluated at call time, after any fusion has happened."""
        return "one2one_cv3" if getattr(self._head, "end2end", False) else "cv3"

    def get_activations(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._activations = {}
        with torch.no_grad():
            self.model(image)

        active = self._active_branch()
        out = {}
        for scale in self.scales:
            act = self._activations.get((scale, active))
            if act is None:
                fired = sorted(f"{s}/{a}" for (s, a) in self._activations)
                raise RuntimeError(
                    f"Expected activation from head.{active} at {scale} but that hook did not fire "
                    f"(hooks that fired: {fired or 'none'}). Likely causes: the model was fused or "
                    f"re-loaded after this adapter was built, or head.end2end changed between "
                    f"construction and this call."
                )
            out[scale] = act
        return out

    # ------------------------------------------------------------- inspection
    def describe(self) -> str:
        """Human-readable summary of what is tapped -- print this once per
        new model and confirm it matches your expectation."""
        lines = [f"Detect head: {self._head.__class__.__name__}, end2end={getattr(self._head, 'end2end', False)}"]
        for (scale, attr), module in self._tapped.items():
            lines.append(f"  {scale}: head.{attr}[..][-2] -> {module.__class__.__name__}")
        lines.append(f"  branch used for output right now: {self._active_branch()}")
        return "\n".join(lines)

    def teardown(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []