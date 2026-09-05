"""Optimizer and learning-rate schedule (paper Sec. 4, "Training details").

AdamW with ``lr=5e-4``, ``betas=(0.9, 0.98)`` and two parameter groups: weight
decay ``0.2`` on the weight matrices, ``0.0`` on everything else.  A parameter
lands in the no-decay group when *any* of these holds:

* it has fewer than two dimensions (biases, LayerNorm gains, every scalar);
* its leaf name is listed in ``optimizer.no_decay_params``;
* its leaf name is ``bias``;
* ``"norm"`` occurs anywhere in its full dotted name (case-insensitively).

Two consequences are worth stating because they are easy to get wrong: the ViT
``cls_token`` is 3-D and has no ``norm`` in its name, so it *does* get weight
decay; and the frozen sin-cos ``pos_embed`` is in neither group, since
non-trainable parameters are skipped entirely.

The schedule is linear warmup for ``warmup_steps`` then cosine decay to zero
over ``total_steps - warmup_steps``.  It is driven by an explicit step index
rather than by ``optimizer.step()``, and :meth:`CosineWithWarmup.step` is
called once per accumulation cycle *before* the forward, so optimizer step
``k`` (0-based) uses the LR computed from index ``k``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from torch import nn
from torch.optim import AdamW, Optimizer

__all__ = ["CosineWithWarmup", "build_optimizer", "parameter_counts", "split_parameter_groups"]


def split_parameter_groups(
    model: nn.Module, no_decay_params: Iterable[str] = ()
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return ``(decay, no_decay)`` trainable parameter lists."""
    no_decay_names = set(no_decay_params)
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        leaf = name.split(".")[-1]
        if param.ndim < 2 or leaf in no_decay_names or leaf == "bias" or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def build_optimizer(
    model: nn.Module,
    *,
    lr: float = 5e-4,
    weight_decay: float = 0.2,
    betas: Sequence[float] = (0.9, 0.98),
    eps: float = 1e-8,
    no_decay_params: Iterable[str] = (),
) -> AdamW:
    """Build the two-group AdamW described in the module docstring."""
    decay, no_decay = split_parameter_groups(model, no_decay_params)
    return AdamW(
        [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(lr),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(eps),
    )


class CosineWithWarmup:
    """Linear warmup then cosine decay to zero, indexed by optimizer step.

    ``total_steps`` is the *cosine horizon*: keeping it at the original value
    while lowering the run's own step budget stops a run early without
    reshaping the curve, which is what the 80k -> 100k ablation protocol relies
    on.
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, base_lr: float) -> None:
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.base_lr = float(base_lr)

    def lr_at(self, step_index: int) -> float:
        """Return the learning rate the schedule assigns to ``step_index``."""
        if step_index < self.warmup_steps:
            return self.base_lr * float(step_index + 1) / float(max(1, self.warmup_steps))
        progress = float(step_index - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
        return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self, step_index: int) -> float:
        """Set every parameter group's LR for ``step_index`` and return it."""
        lr = self.lr_at(step_index)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict[str, float | int]:
        """Return the three numbers that define the curve."""
        return {"warmup_steps": self.warmup_steps, "total_steps": self.total_steps, "base_lr": self.base_lr}

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        """Restore a curve saved by :meth:`state_dict`."""
        self.warmup_steps = int(state["warmup_steps"])
        self.total_steps = int(state["total_steps"])
        self.base_lr = float(state["base_lr"])


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return ``{"total", "trainable", "frozen"}`` parameter counts."""
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
