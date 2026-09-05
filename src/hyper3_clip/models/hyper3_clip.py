"""The Hyper3-CLIP model (paper Sec. 3).

A dual encoder whose 512-d projections are lifted onto a Lorentz hyperboloid of
learned curvature ``kappa``, trained with the hierarchy-conditioned objective of
:mod:`hyper3_clip.models.objective`.

Tensor flow of one training step (``B`` images, ``P`` parts, ``Q`` queries,
``d = embed_dim``, ``d_v`` the vision width, ``W`` the DDP world size)::

    image [B,3,H,W] ┐
                    ├─ one vision forward ─→ pooled [B+P, d_v] , tokens [B+P, 1+N, d_v]
    part_images [P] ┘                             │                    │
                                                  │                    └─ whole-image
                                                  │                       tokens [B, 1+N, d_v]
    caption  [B,Lc] ┐                             ▼
    box text [P,Lp] ├─ one text forward ─→ pooled [B+P+Q, d_t]
    queries  [Q,Lq] ┘                             │
                                                  ▼
       image_proj / text_proj → tangent [·, d] → exp_map0(· * exp(alpha), kappa) → [·, d+1]

The query nodes take a third route: the whole-image patch tokens are pooled
*conditioned on each query's text embedding* (Sec. 3.2,
:class:`~hyper3_clip.models.query_pooling.QueryConditionedPooling`) and the
result goes through the same ``image_proj`` and Lorentz lift.  Query pooling is
a **training-time-only** module: :meth:`Hyper3CLIP.encode_image` and
:meth:`Hyper3CLIP.encode_text` never touch it, so at inference the model is a
plain dual encoder.

Retrieval similarity is the **negative Lorentz distance**
``-d_L(x, y) = -acosh(-kappa <x,y>_L) / sqrt(kappa)``.  Ranking by the Lorentz
inner product ``<x,y>_L`` gives the identical ordering, because ``-d_L`` is a
strictly decreasing function of ``-<x,y>_L`` at fixed ``kappa``; the inner
product is the cheaper score for approximate search, the distance is the one
the paper reports.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hyper3_clip.distributed import (
    gather_variable_many_with_grad,
    gather_with_grad,
    local_target_indices,
    rank_offset,
)
from hyper3_clip.geometry.lorentz import entailment_score, exp_map0, pairwise_dist
from hyper3_clip.models.checkpoints import detect_checkpoint_flavour, load_mapped_state_dict
from hyper3_clip.models.encoders import TextEncoder, VisionEncoder
from hyper3_clip.models.objective import Hyper3CLIPObjective, ObjectiveConfig, compute_objective
from hyper3_clip.models.query_pooling import QueryConditionedPooling

__all__ = ["Hyper3CLIP", "Hyper3CLIPConfig"]

#: Upper clamp on every logit scale: ``log(100)``, i.e. temperature >= 0.01.
LOGIT_SCALE_MAX = 4.6052


@dataclass
class Hyper3CLIPConfig:
    """Architecture and objective configuration of :class:`Hyper3CLIP`.

    Defaults are the paper recipe: ViT-B/16 and CLIP-B/32-text, 512-d
    embedding, learned curvature initialised at 1.0, query pooling on, the
    paper's entailment cone (see :class:`ObjectiveConfig`).  **Both towers are
    always randomly initialised** — the paper trains from scratch, so there is
    no "load pretrained weights" switch.

    ``vision_global_pool`` is ``"token"`` for the paper model; ``"avg"`` exists
    because the checkpoint published on the Hub was trained with it.

    ``query_pooling_mode="mean"`` is the unconditioned-pooling control and
    ``query_parent_mode="shuffled"`` the shuffled-parent control of the
    ablation table; both keep the parameter layout identical so an ablation can
    resume from the shared base checkpoint.
    """

    vision_backbone: str = "vit_base_patch16_224"
    vision_global_pool: Literal["token", "avg"] = "token"
    vision_sincos2d_pos: bool = True
    vision_norm_layer: str | None = "layer_norm"
    text_model_name: str = "openai/clip-vit-base-patch32"
    embed_dim: int = 512
    curv_init: float = 1.0
    learn_curv: bool = True
    query_pooling: bool = True
    query_num_heads: int = 8
    query_mlp_ratio: float = 4.0
    query_pooling_mode: Literal["conditioned", "mean"] = "conditioned"
    query_parent_mode: Literal["true", "shuffled"] = "true"
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    image_size: int = 224
    max_text_length: int = 77

    def __post_init__(self) -> None:
        if isinstance(self.objective, Mapping):
            self.objective = ObjectiveConfig(**dict(self.objective))
        if self.vision_global_pool not in ("token", "avg"):
            raise ValueError("vision_global_pool must be 'token' or 'avg'")
        if self.query_pooling_mode not in ("conditioned", "mean"):
            raise ValueError("query_pooling_mode must be 'conditioned' or 'mean'")
        if self.query_parent_mode not in ("true", "shuffled"):
            raise ValueError("query_parent_mode must be 'true' or 'shuffled'")
        if not math.isfinite(self.curv_init) or self.curv_init <= 0:
            raise ValueError("curv_init must be finite and greater than zero")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a plain, YAML-safe dict with the objective nested under ``objective``."""
        payload = asdict(self)
        payload["objective"] = asdict(self.objective)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Hyper3CLIPConfig:
        """Build a config from :meth:`to_dict` output, rejecting unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown Hyper3CLIPConfig keys: {unknown}")
        kwargs = dict(payload)
        objective = kwargs.get("objective")
        if isinstance(objective, Mapping):
            kwargs["objective"] = ObjectiveConfig(**dict(objective))
        return cls(**kwargs)

    def to_yaml(self, path: str | Path) -> None:
        """Write :meth:`to_dict` to ``path`` as YAML."""
        import yaml

        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Hyper3CLIPConfig:
        """Read a config written by :meth:`to_yaml`."""
        import yaml

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} does not contain a YAML mapping")
        return cls.from_dict(payload)


class Hyper3CLIP(nn.Module):
    """Hierarchy-conditioned hyperbolic vision-language model.

    Parameters (state-dict names): ``vision_encoder.backbone.*``,
    ``text_encoder.backbone.*``, ``image_proj.{weight,bias}``,
    ``text_proj.{weight,bias}``, ``query_pooling.*`` (only when
    ``config.query_pooling``), the three logit scales
    ``{global,local,global_local}_logit_scale``, the per-modality scale factors
    ``visual_alpha`` / ``textual_alpha`` and the log-curvature ``log_curv``.
    """

    def __init__(self, config: Hyper3CLIPConfig | None = None) -> None:
        super().__init__()
        self.config = Hyper3CLIPConfig() if config is None else config
        cfg = self.config

        self.vision_encoder = VisionEncoder(
            cfg.vision_backbone,
            global_pool=cfg.vision_global_pool,
            use_sincos2d_pos=cfg.vision_sincos2d_pos,
            norm_layer=cfg.vision_norm_layer,
        )
        self.text_encoder = TextEncoder(cfg.text_model_name)
        self.tokenizer = self.text_encoder.tokenizer

        self.image_proj = nn.Linear(self.vision_encoder.output_dim, cfg.embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.output_dim, cfg.embed_dim)

        self.query_pooling: QueryConditionedPooling | None = None
        if cfg.query_pooling:
            self.query_pooling = QueryConditionedPooling(
                visual_dim=self.vision_encoder.output_dim,
                text_dim=self.text_encoder.output_dim,
                num_heads=cfg.query_num_heads,
                mlp_ratio=cfg.query_mlp_ratio,
                mode=cfg.query_pooling_mode,
            )

        # Contrastive temperatures: 1/0.07 global, 1/0.05 part-part (local),
        # 1/0.06 part-whole (global-local); stored as logs and clamped to
        # LOGIT_SCALE_MAX at the top of every training forward.
        self.global_logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        self.local_logit_scale = nn.Parameter(torch.tensor(1 / 0.05).log())
        self.global_local_logit_scale = nn.Parameter(torch.tensor(1 / 0.06).log())

        self.visual_alpha = nn.Parameter(torch.tensor(cfg.embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(cfg.embed_dim**-0.5).log())
        self.log_curv = nn.Parameter(torch.tensor(float(cfg.curv_init)).log(), requires_grad=cfg.learn_curv)
        self.curv_min = cfg.curv_init / 10.0
        self.curv_max = cfg.curv_init * 10.0

        self.objective = Hyper3CLIPObjective(cfg.objective)

    # -- geometry ---------------------------------------------------------
    @property
    def curvature(self) -> Tensor:
        """Effective curvature ``kappa = clamp(exp(log_curv), curv_min, curv_max)``."""
        return self.log_curv.exp().clamp(min=self.curv_min, max=self.curv_max)

    def project_image_features(self, tangent: Tensor) -> Tensor:
        """Lift 512-d image tangents onto the hyperboloid, shape ``[N, d+1]``."""
        return exp_map0(tangent.float() * self.visual_alpha.exp().float(), self.curvature.float())

    def project_text_features(self, tangent: Tensor) -> Tensor:
        """Lift 512-d text tangents onto the hyperboloid, shape ``[N, d+1]``."""
        return exp_map0(tangent.float() * self.textual_alpha.exp().float(), self.curvature.float())

    # -- inference --------------------------------------------------------
    def encode_image_tangent(self, pixel_values: Tensor) -> Tensor:
        """Return the 512-d Euclidean image projection (before the Lorentz lift)."""
        return self.image_proj(self.vision_encoder(pixel_values))

    def encode_text_tangent(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return the 512-d Euclidean text projection (before the Lorentz lift)."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return self.text_proj(self.text_encoder(input_ids=input_ids, attention_mask=attention_mask))

    def encode_image(self, pixel_values: Tensor) -> Tensor:
        """Encode images as Lorentz points, shape ``[B, embed_dim + 1]``."""
        return self.project_image_features(self.encode_image_tangent(pixel_values))

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Encode text as Lorentz points, shape ``[B, embed_dim + 1]``."""
        return self.project_text_features(self.encode_text_tangent(input_ids, attention_mask))

    def similarity(self, image_feats: Tensor, text_feats: Tensor) -> Tensor:
        """All-pairs retrieval score: the negative Lorentz distance.

        Ranking by ``lorentz_inner(x, y)`` produces the same order, since
        ``-d_L`` is strictly decreasing in ``-<x, y>_L`` at fixed curvature.
        """
        return -pairwise_dist(image_feats, text_feats, self.curvature)

    def entailment_score(self, general: Tensor, specific: Tensor) -> Tensor:
        """Score how far ``specific`` lies inside ``general``'s entailment cone.

        Both arguments are Lorentz points from :meth:`encode_text` /
        :meth:`encode_image`, matched row-wise.  The result is
        ``clamp(1 - 2*O/pi, 0, 1)`` where ``O`` is the exterior angle at the
        general node: ``1`` when the specific node sits on the cone axis, ``0``
        once it is a right angle or more outside.
        """
        return entailment_score(specific, general, self.curvature)

    # -- training ---------------------------------------------------------
    def encode_image_base_with_tokens(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(pooled [N, d_v], tokens [N, 1+P, d_v])`` from the vision tower."""
        return self.vision_encoder.forward_with_tokens(pixel_values)

    def encode_text_base(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Return pooled text-tower features ``[N, d_t]``."""
        return self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

    def forward(self, batch: Mapping[str, Tensor], *, step: int | None = None) -> dict[str, Any]:
        """Run the fused training forward and return the objective's node dict.

        ``batch`` is the collator output (see
        :class:`~hyper3_clip.data.types.GroundedBatch`).  The returned mapping
        is exactly what :func:`~hyper3_clip.models.objective.compute_objective`
        consumes, including the DDP-gathered negative pools and the rank-offset
        targets; on a single process the pools alias the local tensors.
        """
        with torch.no_grad():
            self._clamp_parameters()

        image = batch["image"]
        part_images = batch["part_images"]
        device = image.device
        kappa = self.curvature
        batch_size = int(image.shape[0])
        part_count = int(part_images.shape[0])

        query_ids = batch.get("query_input_ids")
        query_mask = batch.get("query_attention_mask")
        use_queries = self.query_pooling is not None and query_ids is not None and int(query_ids.shape[0]) > 0
        query_count = int(query_ids.shape[0]) if use_queries else 0

        # --- one vision forward over whole images + part crops ---
        pooled_all, tokens_all = self.encode_image_base_with_tokens(torch.cat([image, part_images], dim=0))
        image_euc_all = self.image_proj(pooled_all)
        image_feats_all = self.project_image_features(image_euc_all)
        image_feats, part_image_feats = image_feats_all.split([batch_size, part_count], dim=0)
        patch_tokens = tokens_all[:batch_size].clone()

        # --- one text forward over captions + box texts + queries ---
        text_groups: list[tuple[Tensor, Tensor]] = [
            (batch["text_input_ids"], batch["text_attention_mask"]),
            (batch["part_text_input_ids"], batch["part_text_attention_mask"]),
        ]
        if use_queries:
            text_groups.append((query_ids, query_mask))
        text_ids, text_mask = self._concat_text_batches(text_groups)
        text_base_all = self.encode_text_base(text_ids, text_mask)
        splits = [batch_size, part_count, query_count] if use_queries else [batch_size, part_count]
        parts_of_text = text_base_all.split(splits, dim=0)
        text_base, part_text_base = parts_of_text[0], parts_of_text[1]
        query_base = parts_of_text[2] if use_queries else None

        text_feats = self.project_text_features(self.text_proj(text_base))
        part_text_feats = self.project_text_features(self.text_proj(part_text_base))

        part_owner = batch["part_owner"].to(device=device, dtype=torch.long)
        if part_owner.numel() == 0:
            image_for_parts = image_feats.narrow(0, 0, 0)
            text_for_parts = text_feats.narrow(0, 0, 0)
        else:
            image_for_parts = image_feats[part_owner]
            text_for_parts = text_feats[part_owner]

        nodes: dict[str, Any] = {
            "image_feats": image_feats,
            "text_feats": text_feats,
            "part_image_flat": part_image_feats,
            "part_text_flat": part_text_feats,
            "part_owner": part_owner,
            "image_for_parts": image_for_parts,
            "text_for_parts": text_for_parts,
            "kappa": kappa,
            "logit_scales": self.logit_scales(),
            "entail_weight_scale": torch.ones((), device=device),
            "global_step": 0 if step is None else int(step),
        }
        nodes.update(self._gathered_pools(nodes))
        nodes.update(
            self._query_nodes(
                patch_tokens=patch_tokens,
                query_base=query_base,
                batch=batch,
                device=device,
                use_queries=use_queries,
            )
        )
        return nodes

    def compute_loss(self, batch: Mapping[str, Tensor], *, step: int | None = None) -> dict[str, Tensor]:
        """Run :meth:`forward` and evaluate the objective on its nodes."""
        return self.loss_from_nodes(self.forward(batch, step=step))

    def loss_from_nodes(self, nodes: Mapping[str, Any]) -> dict[str, Tensor]:
        """Evaluate the objective plus the scalars worth logging each step.

        Kept separate from :meth:`forward` so a DDP-wrapped module can produce
        the nodes and the (parameter-free) objective can be evaluated outside
        the wrapper.
        """
        losses = compute_objective(nodes, self.config.objective)
        kappa = nodes["kappa"]
        return {
            **losses,
            "kappa": kappa.detach().reshape(()),
            "global_logit_scale": self.global_logit_scale.exp().detach(),
            "local_logit_scale": self.local_logit_scale.exp().detach(),
            "global_local_logit_scale": self.global_local_logit_scale.exp().detach(),
        }

    def logit_scales(self) -> dict[str, Tensor]:
        """Return the three log-temperatures the contrastive term consumes."""
        return {
            "global": self.global_logit_scale,
            "local": self.local_logit_scale,
            "global_local": self.global_local_logit_scale,
        }

    # -- forward internals ------------------------------------------------
    def _clamp_parameters(self) -> None:
        self.global_logit_scale.clamp_(max=LOGIT_SCALE_MAX)
        self.local_logit_scale.clamp_(max=LOGIT_SCALE_MAX)
        self.global_local_logit_scale.clamp_(max=LOGIT_SCALE_MAX)
        self.visual_alpha.clamp_(max=0.0)
        self.textual_alpha.clamp_(max=0.0)

    def _concat_text_batches(self, groups: Sequence[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """Re-pad independently padded token groups to one length and concatenate."""
        target_length = max(int(input_ids.shape[1]) for input_ids, _ in groups)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        return (
            torch.cat([_pad_sequence_dim(ids, target_length, pad_token_id) for ids, _ in groups], dim=0),
            torch.cat([_pad_sequence_dim(mask, target_length, 0) for _, mask in groups], dim=0),
        )

    def _gathered_pools(self, nodes: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Build the cross-rank negative pools and the rank-offset targets.

        The whole-image / whole-caption rows have the same count on every rank
        so a plain differentiable all-gather suffices and the target of local
        row ``i`` is ``rank * B + i``.  Part rows are *packed*, so their count
        differs per rank: they go through the padded variable gather, and the
        part targets are offset by the number of part rows on lower ranks.  On
        one process both gathers are the identity and the offsets are zero.
        """
        image_feats = nodes["image_feats"]
        text_feats = nodes["text_feats"]
        part_image_flat = nodes["part_image_flat"]
        part_text_flat = nodes["part_text_flat"]

        pools: dict[str, Tensor] = {
            "all_image_feats": gather_with_grad(image_feats),
            "all_text_feats": gather_with_grad(text_feats),
            "targets": local_target_indices(image_feats.size(0), image_feats.device),
        }

        gathered, counts = gather_variable_many_with_grad(
            [part_image_flat, part_text_flat, nodes["image_for_parts"], nodes["text_for_parts"]]
        )
        pools["all_part_image_feats"] = gathered[0]
        pools["all_part_text_feats"] = gathered[1]
        pools["all_image_for_parts"] = gathered[2]
        pools["all_text_for_parts"] = gathered[3]
        offset = rank_offset(counts).to(device=part_image_flat.device)
        pools["part_targets"] = torch.arange(part_image_flat.size(0), device=part_image_flat.device) + offset
        return pools

    def _query_nodes(
        self,
        *,
        patch_tokens: Tensor,
        query_base: Tensor | None,
        batch: Mapping[str, Tensor],
        device: torch.device,
        use_queries: bool,
    ) -> dict[str, Tensor]:
        """Pool the query-conditioned visual nodes and assemble the query hierarchy.

        Runs with autocast disabled: the pooling attention and the Lorentz lift
        are float32 even under fp16 AMP, matching the training run.
        """
        embed_dim = self.config.embed_dim
        if not use_queries or self.query_pooling is None or query_base is None:
            empty_feats = patch_tokens.new_zeros((0, embed_dim + 1))
            zero = self._query_pooling_keepalive(patch_tokens)
            return {
                "query_image_feats": empty_feats + zero,
                "query_text_feats": empty_feats,
                "query_owner": torch.zeros(0, dtype=torch.long, device=device),
                "query_parent": torch.zeros(0, dtype=torch.long, device=device),
                "query_weight": torch.zeros(0, dtype=torch.float32, device=device),
                "query_source_part": torch.zeros(0, dtype=torch.long, device=device),
            }

        owner = batch["query_owner"].to(device=device, dtype=torch.long)
        parent = batch["query_parent"].to(device=device, dtype=torch.long)
        weight = batch["query_weight"].to(device=device, dtype=torch.float32)
        source_part = batch.get("query_source_part")
        source_part = (
            torch.full_like(owner, -1) if source_part is None else source_part.to(device=device, dtype=torch.long)
        )
        if self.config.query_parent_mode == "shuffled":
            parent = _shuffle_valid_parents(parent)

        with torch.autocast(device_type=device.type, enabled=False):
            tokens = patch_tokens.float()
            query_feats = query_base.float()
            pooled = self.query_pooling(tokens, query_feats, owner)
            query_image_feats = self.project_image_features(self.image_proj(pooled))
            query_text_feats = self.project_text_features(self.text_proj(query_feats))

        return {
            "query_image_feats": query_image_feats,
            "query_text_feats": query_text_feats,
            "query_owner": owner,
            "query_parent": parent,
            "query_weight": weight,
            "query_source_part": source_part,
        }

    def _query_pooling_keepalive(self, patch_tokens: Tensor) -> Tensor:
        """Return a zero that touches every pooling parameter.

        A batch with no queries would otherwise leave the pooling block out of
        the autograd graph and break DDP's reduction invariant on that step.
        """
        if self.query_pooling is None:
            return patch_tokens.new_zeros(())
        with torch.autocast(device_type=patch_tokens.device.type, enabled=False):
            empty_query = patch_tokens.new_zeros((0, self.text_encoder.output_dim))
            empty_owner = torch.zeros(0, dtype=torch.long, device=patch_tokens.device)
            return self.query_pooling(patch_tokens.float(), empty_query, empty_owner).sum()

    # -- checkpoints ------------------------------------------------------
    def save_pretrained(self, directory: str | Path) -> Path:
        """Write ``config.yaml`` + ``model.safetensors`` into ``directory``."""
        from safetensors.torch import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(target / "config.yaml")
        state = {key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()}
        save_file(state, str(target / "model.safetensors"))
        return target

    @classmethod
    def from_pretrained(
        cls,
        path_or_hf_id: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        config: Hyper3CLIPConfig | None = None,
        strict: bool = True,
    ) -> Hyper3CLIP:
        """Load either supported checkpoint layout.

        ``path_or_hf_id`` may be a directory (this repository's ``config.yaml``
        + ``model.safetensors``, or the released artifact's ``config.json`` +
        ``model.safetensors``), a training checkpoint ``.pt`` written by this
        repository, or a Hugging Face repo id, which is downloaded with
        ``huggingface_hub`` if it is not a local path.

        Pass ``config`` to override the configuration inferred from the
        checkpoint.  With ``strict=True`` (the default) any key that cannot be
        matched raises, listing the offending names.
        """
        state_dict, stored_config, flavour = _read_checkpoint(path_or_hf_id, map_location=map_location)
        resolved = config if config is not None else _config_for(state_dict, stored_config, flavour)
        model = cls(resolved)
        load_mapped_state_dict(model, state_dict, flavour=flavour, strict=strict)
        model.to(map_location)
        return model


def _pad_sequence_dim(tensor: Tensor, target_length: int, value: int) -> Tensor:
    pad = target_length - int(tensor.shape[1])
    if pad <= 0:
        return tensor
    return F.pad(tensor, (0, pad), value=value)


def _shuffle_valid_parents(parent: Tensor) -> Tensor:
    """Roll the non-root parent assignments by one (the shuffled-parent control).

    Keeps the number of parent edges and the set of parent nodes identical
    while destroying which child each belongs to, so any gain from the true
    hierarchy cannot be explained by the edge count alone.
    """
    shuffled = parent.clone()
    valid = shuffled >= 0
    if int(valid.sum().item()) > 1:
        shuffled[valid] = shuffled[valid].roll(1)
    return shuffled


# ---------------------------------------------------------------------------
# checkpoint discovery
# ---------------------------------------------------------------------------
def _read_checkpoint(
    path_or_hf_id: str | Path,
    *,
    map_location: str | torch.device,
) -> tuple[dict[str, Tensor], Mapping[str, Any] | None, str]:
    path = Path(path_or_hf_id)
    if path.is_file():
        return _read_torch_checkpoint(path, map_location=map_location)
    if not path.is_dir():
        path = Path(_download_repo(str(path_or_hf_id)))

    weights = path / "model.safetensors"
    if not weights.is_file():
        candidates = sorted(p.name for p in path.glob("*.safetensors")) or sorted(p.name for p in path.glob("*.pt"))
        raise FileNotFoundError(f"{path} has no model.safetensors (found: {candidates})")

    from safetensors.torch import load_file

    state_dict = load_file(str(weights), device=str(map_location))
    flavour = detect_checkpoint_flavour(state_dict)
    stored_config: Mapping[str, Any] | None = None
    if flavour == "hub" and (path / "config.json").is_file():
        stored_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    elif (path / "config.yaml").is_file():
        import yaml

        stored_config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
    return state_dict, stored_config, flavour


def _read_torch_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device,
) -> tuple[dict[str, Tensor], Mapping[str, Any] | None, str]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{path} does not contain a checkpoint mapping")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict"))
    if state_dict is None:
        state_dict = checkpoint  # a bare state dict
    stored_config = checkpoint.get("config") if isinstance(checkpoint, Mapping) else None
    return dict(state_dict), stored_config, detect_checkpoint_flavour(state_dict)


def _download_repo(repo_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise FileNotFoundError(
            f"{repo_id!r} is not a local path and huggingface_hub is not installed to download it"
        ) from error
    return snapshot_download(repo_id=repo_id, allow_patterns=["*.json", "*.yaml", "*.safetensors"])


def _config_for(
    state_dict: Mapping[str, Tensor],
    stored_config: Mapping[str, Any] | None,
    flavour: str,
) -> Hyper3CLIPConfig:
    """Build the :class:`Hyper3CLIPConfig` a checkpoint should be loaded with."""
    if flavour == "public" and stored_config is not None:
        payload = stored_config.get("model", stored_config) if "model" in stored_config else stored_config
        if isinstance(payload, Mapping) and "vision_backbone" in payload:
            merged = dict(payload)
            if "objective" not in merged and isinstance(stored_config.get("objective"), Mapping):
                merged["objective"] = stored_config["objective"]
            return Hyper3CLIPConfig.from_dict(merged)

    # Released Hub artifact: config.json carries the towers and the geometry.
    # The pooling flavour is read off the state dict (timm keeps `fc_norm` for
    # average pooling and `norm` for token pooling); the positional-embedding
    # and norm-layer switches are not recorded and are what the artifact was
    # built with (learned positions, timm's own norm layer).
    hub = dict(stored_config or {})
    image_proj_weight = state_dict.get("image_proj.weight")
    return Hyper3CLIPConfig(
        vision_backbone=hub.get("vision_backbone", "vit_base_patch16_224"),
        vision_global_pool="avg" if "vision_encoder.backbone.fc_norm.weight" in state_dict else "token",
        vision_sincos2d_pos=False,
        vision_norm_layer=None,
        text_model_name=hub.get("text_model_name", "openai/clip-vit-base-patch32"),
        embed_dim=int(hub.get("embed_dim", 512 if image_proj_weight is None else image_proj_weight.shape[0])),
        curv_init=float(hub.get("curv_init", 1.0)),
        learn_curv=bool(hub.get("learn_curv", True)),
        query_pooling=any(key.startswith("query_pooling.") for key in state_dict),
        image_size=int(hub.get("image_size", 224)),
        max_text_length=int(hub.get("max_text_length", 77)),
    )
