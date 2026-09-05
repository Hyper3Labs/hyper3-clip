# Training

## Launch

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --config configs/paper/hyper3_clip_vitb_500k.yaml \
    --override data.tarfiles='["/data/grit-processed/shard-{000000..000999}.tar"]' \
    --override output_dir=/scratch/runs/hyper3_clip_vitb_500k
```

`--override` takes dotted paths into the config tree and YAML-typed values, and
may be repeated. The resolved config is written to `<output_dir>/config.yaml`
before the first step. Without torchrun's `RANK` / `WORLD_SIZE` everything
degrades to a single process, which is how the tests run.

## The recipe

| | value |
|---|---|
| towers | ViT-B/16 @224 and CLIP-B/32 text, both random init |
| embedding | 512-d tangent, 513-d Lorentz, curvature 1.0 learned in `[0.1, 10]` |
| objective | entailment-cone violation, `entail_weight = 0.2`, apertures 0.7 / 1.2, calibration alpha 10 |
| query pooling | 8 heads, MLP ratio 4.0, CLS token dropped |
| data | processed GRIT, `max_parts: 5`, tight-crop + jitter + grayscale, ImageNet normalization |
| queries | 5 sentences, 30 shared text-box/phrase budget, 6 kept per image, text boxes on |
| schedule | 500k steps, global batch 768, `grad_accum_steps: 1` |
| optimizer | AdamW, lr 5e-4, weight decay 0.2, betas (0.9, 0.98), eps 1e-8 |
| warmup | 4000 steps linear, then cosine to 0 |
| precision | fp16 autocast + GradScaler, grad clip 1.0 |
| bookkeeping | checkpoint every 10k steps, log every 20 steps, seed 31 (per rank: `31 + rank`) |

## Training data

Training data is **processed GRIT**: the HyCoCLIP-preprocessed shards of the
GRIT grounded image-text corpus, in which every localized box has already been
cropped into its own JPEG at shard-build time.

One WebDataset sample per image group, keyed by `__key__`:

| member | content |
|---|---|
| `child.jpg` | the whole image |
| `child.txt` | its caption (UTF-8) |
| `numparents.txt` | decimal ASCII count `N` of localized parts |
| `parent{i:03d}.jpg` | part `i`, **already cropped** |
| `parent{i:03d}.txt` | the text of box `i` |

There are **no bounding boxes in the shards** and the reader never crops: "box
cropping" means reading `parentNNN.jpg` and applying the same train transform
used for the whole image, with independent randomness per call.

`ProcessedGritDataset` yields
`{"image": Tensor[3,224,224], "caption": str, "part_images": list[Tensor],
"part_texts": list[str], "key": str}`.

* **Part cap.** Every part is kept; when a group has more than `max_parts` (5 in
  the paper recipe) a random subset of that size is drawn and ascending order is
  **restored**, so part order always follows the shard's own. `max_parts: null`
  disables the cap.
* **Sharding.** Shards are split across ranks (`tarfiles[rank::world_size]`)
  and then across DataLoader workers. The shuffle buffer is 4000.
* **No epochs.** `__iter__` loops forever over deep copies of the pipeline, so
  a run is measured in optimizer steps.
* `deterministic_transforms=True` seeds each image's augmentation from the
  sample key, which makes fixtures reproducible; the paper run leaves it off.

`tests/fixtures/make_grit_fixture.py` writes a four-sample shard in exactly this
layout with procedurally generated images, so nothing has to be downloaded to
exercise the pipeline:

```bash
python tests/fixtures/make_grit_fixture.py /tmp/grit-fixture-000000.tar
```

The `data:` section of a run config:

| key | paper value | meaning |
|---|---|---|
| `type` | `processed_grit` | the only reader shipped here |
| `tarfiles` | placeholder | shard paths or brace/glob patterns |
| `image_size`, `max_text_length` | 224, 77 | resolution, CLIP context |
| `max_parts` | 5 | per-image cap on localized parts |
| `train_transform`, `image_normalization` | `tight_crop_color_jitter_gray`, `imagenet` | the augmentation preset |
| `shuffle_buffer`, `num_workers` | 4000, 8 | loader tuning |
| `queries_enabled` | `true` | emit the query tensors |
| `max_sentences`, `max_phrases`, `max_queries_per_image` | 5, 30, 6 | the query budgets |
| `use_text_boxes` | `true` | include text-box queries |

### Optimizer, schedule and checkpoints

AdamW with the usual no-decay group (biases, norms, parameters with fewer than
two dimensions); linear warmup for `warmup_steps`, then cosine decay to zero
over `training.scheduler_total_steps` (defaults to `total_steps`). Training runs
under fp16 autocast with a `GradScaler` and gradient clipping at 1.0; the
objective and every Lorentz operation run in float32.

Rank 0 logs one JSON row per `log_interval` steps to
`<output_dir>/train_log.jsonl` and writes `checkpoint_step_{N}.pt` every
`ckpt_interval` steps. To resume, set `training.resume_from` (or the
`RESUME_FROM_CHECKPOINT` environment variable), or leave `training.resume: true`
to pick up the newest checkpoint under `output_dir`.

## Ablation protocol (paper Sec. 5.1, Table 5)

One shared **base run** trains for 80k steps with all three query relation
weights at zero (`configs/paper/ablations/base_80k.yaml`). Each ablation row
then resumes that single checkpoint and runs to 100k steps, changing exactly
one thing:

| config | what it changes |
|---|---|
| `no_query.yaml` | nothing: the control that isolates the extra 20k steps |
| `query_image_text_only.yaml` | relation (1) `I^q <= T^q` on |
| `visual_hierarchy_only.yaml` | relation (2) `I <= I^q` on |
| `text_hierarchy_only.yaml` | relation (3) `T^pi(q) <= T^q` on |
| `full_objective_100k.yaml` | all three on |
| `mean_pooled_visual.yaml` | all three on, pooling replaced by a plain token mean |
| `reversed_visual_order.yaml` | all three on, visual relation reversed to `I^q <= I` |
| `shuffled_text_parents.yaml` | all three on, parents rolled by one |

Every row uses `scheduler_total_steps: 100000`, so the base run and all rows
share one cosine shape and resuming introduces no LR discontinuity. Every row
keeps `model.query_pooling: true`, so the parameter layout — and therefore the
checkpoint — is identical and the base checkpoint loads strictly into all of
them. `tests/test_configs.py` asserts that each row differs from the base in
its documented keys and nothing else.

Run the base once, then the rows:

```bash
torchrun --nproc_per_node=8 scripts/train.py --config configs/paper/ablations/base_80k.yaml
for row in no_query query_image_text_only visual_hierarchy_only text_hierarchy_only \
           full_objective_100k mean_pooled_visual reversed_visual_order shuffled_text_parents; do
  torchrun --nproc_per_node=8 scripts/train.py --config "configs/paper/ablations/${row}.yaml"
done
```

## Max-parts sweep (paper Sec. 5.3)

`configs/paper/max_parts_sweep/` holds a 10k-step proxy run per cap — 2, 3, 4,
5, 6 and uncapped (`max_parts: null`) — identical to the paper recipe except
for `data.max_parts`, the run name and the shorter budget. It measures how many
localized parts per image the objective can use before the extra rows stop
paying for themselves.
