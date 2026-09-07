# Hyper3-CLIP: Hierarchy-Conditioned Hyperbolic Vision-Language Training

Code for the ECCV 2026 workshop paper *Hyper3-CLIP: Hierarchy-Conditioned
Hyperbolic Vision-Language Training*. Hyper3-CLIP is a dual encoder whose
512-d projections are lifted onto a Lorentz hyperboloid of learned curvature
and trained with hyperbolic entailment terms over a rule-based caption
hierarchy. The visual node of each query is pooled from the image's patch
tokens with cross-attention conditioned on that query's text embedding; this
pooling runs only during training, so at inference the model is a standard dual
encoder and adds no extra computation.

This repository holds the single variant described in the paper and the
evaluation code that produced the paper's numbers.

## Authors

- Matin Mahmood, hyper3labs, Berlin, Germany
- Antonio Rueda-Toicen, Hasso Plattner Institute, University of Potsdam, Germany
- Mohamed ElBassat, Faculty of Computers and Data Science, Alexandria University, Egypt
- Seifeldin Elkerdany, Faculty of Computer Science and Engineering, Alamein International University, Egypt
- Weixing Wang, Hasso Plattner Institute, University of Potsdam, Germany
- Gerard de Melo, Hasso Plattner Institute, University of Potsdam, Germany

## Installation

```bash
git clone https://github.com/hyper3labs/hyper3-clip.git
cd hyper3-clip
uv sync --extra dev          # or: pip install -e ".[dev]"
```

Python 3.10+. The CLIP tokenizer and text config are fetched from the Hugging
Face hub on first use and cached; for offline runs pre-populate the cache and
set `HF_HUB_OFFLINE=1`.

## Quickstart

```python
import torch
from hyper3_clip.models import Hyper3CLIP, build_retrieval_transform, build_tokenizer

model = Hyper3CLIP.from_pretrained("hyper3labs/hyper3-clip").eval()
transform = build_retrieval_transform(model.config)
tokenizer = build_tokenizer(model.config)

from PIL import Image
image = transform(Image.open("photo.jpg").convert("RGB"))[None]
captions = [
    "a red bicycle leaning against a stone wall",
    "a plate of pasta on a kitchen table",
]
tokens = tokenizer(captions, padding=True, truncation=True, max_length=77, return_tensors="pt")

with torch.no_grad():
    image_feats = model.encode_image(image)                       # [1, 513] Lorentz points
    text_feats = model.encode_text(tokens["input_ids"], tokens["attention_mask"])
    scores = model.similarity(image_feats, text_feats)            # -d_L, higher is better

print(captions[int(scores[0].argmax())])
```

Retrieval is ranked by the **negative Lorentz distance**. Ranking by the
Lorentz inner product gives the identical order and is a plain dot product with
the time coordinate negated, which is the cheaper score for a vector database.

Entailment is directional: ask how far a phrase sits inside a caption's cone.
The score is `1` on the cone axis and falls to `0` once the specific node is a
right angle or more outside:

```python
ids = tokenizer(["a red bicycle leaning against a stone wall", "a red bicycle"],
                padding=True, return_tensors="pt")["input_ids"]
with torch.no_grad():
    caption, phrase = model.encode_text(ids)
    print(float(model.entailment_score(caption[None], phrase[None])))
```

## Training

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --config configs/paper/hyper3_clip_vitb_500k.yaml \
    --override data.tarfiles='["/data/grit-processed/shard-{000000..000999}.tar"]' \
    --override output_dir=/scratch/runs/hyper3_clip_vitb_500k
```

The paper recipe is ViT-B/16 and CLIP-B/32-text from scratch, 500k steps at
global batch 768 on processed GRIT, AdamW at lr 5e-4 with 4000 warmup steps and
cosine decay, fp16 autocast and gradient clipping at 1.0. `docs/training.md`
has the full recipe, the shard schema, the resume rules, the ablation protocol
and the max-parts sweep; `docs/method.md` maps the paper's equations onto the
code.

`configs/paper/ablations/` implements the paper's ablation protocol: one shared
80k-step base run with every query relation switched off, then eight rows that
each resume that single checkpoint to 100k steps and change exactly one
control. `configs/paper/max_parts_sweep/` sweeps the per-image part cap (2, 3,
4, 5, 6 and uncapped) as a 10k-step proxy.

## Evaluation

Every number in Tables 2-6 and Sec. 5.3 comes from one command per suite:

```bash
cp configs/eval/local_paths.example.yaml configs/eval/local_paths.yaml   # fill in dataset roots
python scripts/evaluate.py \
    --checkpoint hyper3labs/hyper3-clip \
    --suite configs/eval/all_paper_tables.yaml \
    --paths configs/eval/local_paths.yaml \
    --output-dir results/hyper3-clip
python scripts/summarize_results.py --results-dir results/hyper3-clip --print
```

`configs/eval/` holds the six suites (`paper_table_2`, `paper_table_3`,
`paper_table_4`, `hierarchy_entailment`, `prompt_sensitivity`,
`all_paper_tables`); dataset locations live only in the git-ignored `--paths`
file, so no suite carries a local path. The driver writes one JSON per task with
a cache key, and the summarizer emits long/wide CSVs plus a Markdown table per
paper table in its published column order. `scripts/ablation_table.py` prints
the Table 5 ablation with its deltas, `scripts/build_multilabel_manifest.py`
builds the VOC/COCO manifests, and `scripts/scan_grit_parts.py` reproduces the
Sec. 5.3 parts-per-image statistics.

`docs/evaluation.md` documents each table's protocol, dataset layout, exact
prompt strings (including the Official vs Photo regimes of Table 6) and
reproduction command, plus a synthetic smoke run that exercises every task type
with no downloads.

## Repository layout

```
src/hyper3_clip/
  geometry/lorentz.py        Lorentz primitives: exp map, distance, cones, entailment score
  data/                      query_hierarchy.py, grit.py, transforms.py, collate.py, types.py
  models/                    encoders.py, query_pooling.py, hyper3_clip.py, objective.py,
                             contrastive.py, entailment.py, reduction.py, checkpoints.py,
                             preprocessing.py
  training/                  config.py, optim.py, checkpoint.py, distributed.py, trainer.py
  evaluation/                retrieval.py, imagenet.py, classification.py, multilabel.py,
                             hierarchy_entailment.py, grit_stats.py, metrics.py, prompts.py,
                             class_names.py, encoding.py, config.py, runner.py, reporting.py,
                             assets/ (vendored class names and ImageNet hierarchy)
  distributed.py             rank-aware gathers shared by the forward and the loop
configs/paper/               the recipe, the ablation rows, the max-parts sweep
configs/eval/                the six evaluation suites and the paths template
scripts/                     train.py, evaluate.py, summarize_results.py, ablation_table.py,
                             build_multilabel_manifest.py, scan_grit_parts.py
docs/                        method.md, training.md, evaluation.md
tests/                       unit tests plus tiny end-to-end fixture runs for training and evaluation
```

## Checkpoints

`Hyper3CLIP.from_pretrained` reads this repository's own `config.yaml` +
`model.safetensors` directories (and training `.pt` checkpoints), and the
released Hugging Face artifact at
<https://huggingface.co/hyper3labs/hyper3-clip>. That released artifact is
**not** the paper model: it comes from an earlier run with average pooling, a
single temperature and no query hierarchy. The paper model's checkpoint will be
published on the Hub in this repository's own format.

## License

MIT; see [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{mahmood2026hyper3clip,
  title     = {Hyper3-CLIP: Hierarchy-Conditioned Hyperbolic Vision-Language Training},
  author    = {Mahmood, Matin and Rueda-Toicen, Antonio and ElBassat, Mohamed and Elkerdany, Seifeldin and Wang, Weixing and de Melo, Gerard},
  year      = {2026}
}
```
