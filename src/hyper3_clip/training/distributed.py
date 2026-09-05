"""torchrun / DDP setup for the training loop.

The gather primitives the model forward needs live in
:mod:`hyper3_clip.distributed` and are re-exported here for convenience; this
module adds the launcher-facing pieces: process-group init and teardown, device
selection, the DDP wrapper and rank-0-only logging.

Launch with::

    torchrun --nproc_per_node=8 scripts/train.py --config configs/paper/hyper3_clip_vitb_500k.yaml

``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` come from torchrun; without them
everything degrades to a single process.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from hyper3_clip.distributed import (
    gather_variable_many_with_grad,
    gather_variable_with_grad,
    gather_with_grad,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    local_target_indices,
)

__all__ = [
    "barrier",
    "destroy_distributed",
    "gather_variable_many_with_grad",
    "gather_variable_with_grad",
    "gather_with_grad",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "init_distributed",
    "is_distributed",
    "is_main_process",
    "local_target_indices",
    "log_main",
    "resolve_device",
    "wrap_ddp",
]


def init_distributed() -> None:
    """Initialise the process group when launched under torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not (dist.is_available() and dist.is_initialized()):
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(get_local_rank())
        dist.init_process_group(backend=backend)


def barrier() -> None:
    """Synchronise every rank (no-op on a single process)."""
    if is_distributed():
        dist.barrier()


def destroy_distributed() -> None:
    """Tear the process group down (no-op on a single process)."""
    if is_distributed():
        dist.destroy_process_group()


def resolve_device() -> torch.device:
    """Return this process's device: its own CUDA device, else CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if "LOCAL_RANK" in os.environ:
        device = torch.device(f"cuda:{get_local_rank()}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cuda")


def wrap_ddp(model: nn.Module, device: torch.device, *, find_unused_parameters: bool = True) -> nn.Module:
    """Wrap ``model`` in ``DistributedDataParallel`` when the world size is > 1.

    ``find_unused_parameters`` defaults to ``True`` because several sub-losses
    can be skipped for a given batch -- an image group with no parts, or a
    control that zeroes the query weights -- which would otherwise trip DDP's
    reduction invariant.  It costs one extra autograd-graph traversal per step.
    """
    if get_world_size() <= 1:
        return model
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(
        model,
        device_ids=[get_local_rank()] if device.type == "cuda" else None,
        broadcast_buffers=False,
        find_unused_parameters=bool(find_unused_parameters),
    )


def log_main(*args: Any, **kwargs: Any) -> None:
    """``print`` on rank 0 only, flushed."""
    if is_main_process():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)
