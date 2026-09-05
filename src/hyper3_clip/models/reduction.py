"""DDP mean rescaling for packed (variable-count) rows.

:func:`global_mean_scale` rescales a rank-local mean so that DDP's gradient
average over ranks equals the true global mean when the per-rank row count
varies.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor

__all__ = ["global_mean_scale"]


def global_mean_scale(local_denominator: Tensor) -> Tensor:
    """Scale a local mean so DDP's rank average is the true global mean.

    DDP averages gradients over ``W`` ranks, so a rank-local mean over
    ``n_rank`` rows must be multiplied by ``W * n_rank / sum_rank(n_rank)`` for
    the averaged result to equal the mean over all rows of all ranks.  Returns
    ``1`` in a single-process run, which is what every test here exercises.

    When *no* rank contributed a row the scale is ``0``, not ``1``: a rank that
    owns no rows must not have its (already meaningless) local mean amplified,
    and the collective must still run on every rank so the process group stays
    in lock-step.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return local_denominator.new_ones(())
    world_size = dist.get_world_size()
    if world_size == 1:
        return local_denominator.new_ones(())
    local = local_denominator.detach().to(dtype=torch.float32)
    total = local.clone()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    safe_total = total.clamp_min(torch.finfo(total.dtype).eps)
    scale = local * world_size / safe_total
    return torch.where(total > 0.0, scale, torch.zeros_like(scale))
