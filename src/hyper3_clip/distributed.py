"""Process-group helpers shared by the model forward and the training loop.

Training Hyper3-CLIP with DDP needs three things from ``torch.distributed``:

* the rank/world-size accessors every packing rule depends on;
* :func:`gather_with_grad`, the differentiable all-gather that turns a
  rank-local batch of Lorentz points into the global negative pool used by the
  contrastive term (paper Sec. 3.3: the contrastive loss is taken over the full
  768-sample global batch, not the 96-sample rank slice);
* :func:`gather_variable_many_with_grad`, the same thing for the *packed* part
  and query rows, whose count differs per rank and therefore has to be padded
  to the per-rank maximum, gathered and then unpadded.

Everything here degrades to an identity on a single process, which is the case
every test in this repository exercises.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import Tensor

__all__ = [
    "empty_variable_batch_like",
    "gather_variable_many_with_grad",
    "gather_variable_with_grad",
    "gather_with_grad",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_main_process",
    "local_target_indices",
    "rank_offset",
]


def is_distributed() -> bool:
    """Return ``True`` when a process group is initialised."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return this process's rank (``0`` outside a process group)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Return the process-group size (``1`` outside a process group)."""
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    """Return ``LOCAL_RANK`` from the environment (``0`` when unset)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    """Return ``True`` on rank 0."""
    return get_rank() == 0


def empty_variable_batch_like(tensor: Tensor) -> Tensor:
    """Return a graph-connected empty batch with ``tensor``'s trailing shape."""
    if tensor.dim() == 0:
        raise ValueError("empty_variable_batch_like requires a batched tensor")
    return tensor.narrow(0, 0, 0)


def gather_with_grad(tensor: Tensor) -> Tensor:
    """All-gather ``tensor`` along dim 0, keeping gradients flowing to every rank.

    Rows are laid out rank-major, so rank ``r``'s local row ``i`` lands at
    global index ``r * B + i``; :func:`local_target_indices` produces exactly
    those indices as the contrastive targets.
    """
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    from torch.distributed.nn import all_gather as differentiable_all_gather

    return torch.cat(list(differentiable_all_gather(tensor.contiguous())), dim=0)


def _variable_gather_metadata(tensor: Tensor) -> tuple[Tensor, int, Tensor]:
    world_size = get_world_size()
    local_count = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    if world_size == 1:
        keep = torch.ones(tensor.shape[0], device=tensor.device, dtype=torch.bool)
        return local_count, tensor.shape[0], keep

    counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(counts, local_count)
    count_tensor = torch.cat(counts)
    max_count = int(count_tensor.max().item())
    keep = torch.zeros(world_size * max_count, device=tensor.device, dtype=torch.bool)
    for rank, count in enumerate(count_tensor.tolist()):
        start = rank * max_count
        keep[start : start + count] = True
    return count_tensor, max_count, keep


def _gather_variable_from_metadata(tensor: Tensor, max_count: int, keep: Tensor) -> Tensor:
    from torch.distributed.nn import all_gather as differentiable_all_gather

    padded = tensor.new_zeros((max_count, *tensor.shape[1:]))
    padded[: tensor.shape[0]] = tensor
    gathered = torch.cat(list(differentiable_all_gather(padded.contiguous())), dim=0)
    return gathered[keep]


def gather_variable_with_grad(tensor: Tensor) -> tuple[Tensor, Tensor]:
    """All-gather a tensor whose first dimension differs per rank.

    Returns ``(gathered, counts)`` where ``counts[r]`` is rank ``r``'s row
    count; the gathered rows are rank-major and the padding is removed.
    """
    count_tensor, max_count, keep = _variable_gather_metadata(tensor)
    if get_world_size() == 1:
        return tensor, count_tensor
    return _gather_variable_from_metadata(tensor, max_count, keep), count_tensor


def gather_variable_many_with_grad(tensors: Sequence[Tensor]) -> tuple[list[Tensor], Tensor]:
    """Gather several tensors that share one variable first dimension.

    Tensors with the same dtype and trailing shape are concatenated along the
    last dimension so a single collective serves all of them; the result is
    split back apart.  ``counts`` is shared metadata, gathered once.
    """
    if not tensors:
        raise ValueError("gather_variable_many_with_grad requires at least one tensor")
    first = tensors[0]
    for tensor in tensors:
        if tensor.device != first.device:
            raise ValueError("all tensors must be on the same device")
        if tensor.shape[0] != first.shape[0]:
            raise ValueError("all tensors must have the same first dimension")
    count_tensor, max_count, keep = _variable_gather_metadata(first)
    if get_world_size() == 1:
        return list(tensors), count_tensor

    gathered: list[Tensor | None] = [None] * len(tensors)
    groups: dict[tuple[torch.dtype, torch.Size, int], list[int]] = {}
    for index, tensor in enumerate(tensors):
        if tensor.dim() == 0:
            raise ValueError("variable gather tensors must have at least one dimension")
        key = (tensor.dtype, tensor.shape[1:-1], tensor.dim()) if tensor.dim() > 1 else (tensor.dtype, torch.Size(), 1)
        groups.setdefault(key, []).append(index)

    for indices in groups.values():
        group_tensors = [tensors[index] for index in indices]
        if len(group_tensors) == 1 or group_tensors[0].dim() == 1:
            for index, tensor in zip(indices, group_tensors, strict=True):
                gathered[index] = _gather_variable_from_metadata(tensor, max_count, keep)
            continue
        widths = [tensor.shape[-1] for tensor in group_tensors]
        packed = torch.cat(group_tensors, dim=-1)
        gathered_packed = _gather_variable_from_metadata(packed, max_count, keep)
        for index, chunk in zip(indices, gathered_packed.split(widths, dim=-1), strict=True):
            gathered[index] = chunk

    if any(tensor is None for tensor in gathered):
        raise RuntimeError("internal error while gathering variable tensors")
    return [tensor for tensor in gathered if tensor is not None], count_tensor


def local_target_indices(batch_size: int, device: torch.device) -> Tensor:
    """Return this rank's rows' indices inside the gathered global batch."""
    return torch.arange(batch_size, device=device) + batch_size * get_rank()


def rank_offset(counts: Tensor) -> Tensor:
    """Return the number of rows owned by strictly lower ranks.

    ``counts`` is the per-rank row count returned by the variable gathers; the
    offset turns a rank-local packed index into an index into the gathered
    pool.
    """
    if counts.numel() <= 1:
        return counts.new_zeros(())
    return counts[: get_rank()].sum()
