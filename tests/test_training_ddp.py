"""Two-process DDP regression test for the training step.

Runs the tiny model under ``DistributedDataParallel`` (gloo, CPU, two ranks)
with ``find_unused_parameters=True``, which is how the paper recipe trains.
The objective consumes the temperature parameters after the encoder forward,
so evaluating it outside the wrapped forward makes DDP mark those parameters
ready twice ("Expected to mark a variable ready only once").  The single
process tests never wrap in DDP and cannot catch that.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import replace
from pathlib import Path

import torch
import torch.multiprocessing as mp

from conftest import requires_tokenizer
from hyper3_clip.models import Hyper3CLIPConfig

WORLD_SIZE = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rank_main(rank: int, port: int, model_config: Hyper3CLIPConfig, shard: str, output_dir: str) -> None:
    from torch.nn.parallel import DistributedDataParallel

    from hyper3_clip.training.distributed import destroy_distributed, init_distributed
    from hyper3_clip.training.trainer import Trainer, build_dataloader
    from test_training import _tiny_run_config

    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(WORLD_SIZE),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    torch.set_num_threads(1)
    init_distributed()
    try:
        config = _tiny_run_config(model_config, Path(shard), Path(output_dir), total_steps=2)
        payload = config.to_dict()
        payload["training"]["find_unused_parameters"] = True
        payload["data"]["tarfiles"] = [shard] * WORLD_SIZE  # one shard per rank
        config = type(config).from_dict(payload)
        trainer = Trainer(config, device=torch.device("cpu"))
        assert isinstance(trainer.model, DistributedDataParallel)
        trainer.fit(build_dataloader(config, trainer.raw_model.tokenizer))
        assert trainer.step == 2
    finally:
        destroy_distributed()


@requires_tokenizer
def test_train_step_under_ddp_with_unused_parameter_detection(
    tmp_path: Path, tiny_model_config: Hyper3CLIPConfig, grit_shard: Path
) -> None:
    output_dir = tmp_path / "ddp_run"
    mp.start_processes(
        _rank_main,
        args=(_free_port(), replace(tiny_model_config), str(grit_shard), str(output_dir)),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="spawn",
    )
    rows = [json.loads(line) for line in (output_dir / "train_log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert all(row["world_size"] == WORLD_SIZE for row in rows)
    assert all(torch.isfinite(torch.tensor(row["loss"])) for row in rows)


@requires_tokenizer
def test_loss_matches_between_forward_paths(tiny_model_config: Hyper3CLIPConfig, grit_shard: Path) -> None:
    """``return_loss=True`` is exactly ``loss_from_nodes(forward(...))``."""
    from hyper3_clip.models import Hyper3CLIP
    from hyper3_clip.training.trainer import build_dataloader
    from test_training import _tiny_run_config

    config = _tiny_run_config(replace(tiny_model_config), grit_shard, Path("unused"), total_steps=1)
    model = Hyper3CLIP(config.model_config()).eval()
    batch = next(iter(build_dataloader(config, model.tokenizer)))
    with torch.no_grad():
        direct = model(batch, step=0, return_loss=True)
        via_nodes = model.loss_from_nodes(model(batch, step=0))
    for key in ("loss", "contrastive", "entailment"):
        assert torch.allclose(direct[key], via_nodes[key]), key
