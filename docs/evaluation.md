# Evaluation

Every number in the paper's Tables 2–6 and Sec. 5.3 comes from the code under
`src/hyper3_clip/evaluation/`, driven by `scripts/evaluate.py` against a suite
in `configs/eval/`. This page gives, per table, the protocol, the dataset
layout it expects, the exact prompts, and the command to reproduce it.

- [Setup](#setup)
- [Table 2 — retrieval and the ImageNet hierarchy](#table-2--retrieval-and-the-imagenet-hierarchy)
- [Table 3 — zero-shot classification](#table-3--zero-shot-classification)
- [Table 4 — zero-shot multi-label mAP](#table-4--zero-shot-multi-label-map)
- [Table 5 — hierarchy entailment](#table-5--hierarchy-entailment-sec-51)
- [Table 6 — prompt sensitivity](#table-6--prompt-sensitivity-sec-52)
- [Sec. 5.3 — GRIT parts-per-image scan](#sec-53--grit-parts-per-image-scan)
- [Results, caching and summaries](#results-caching-and-summaries)
- [Smoke test](#smoke-test)

## Setup

```bash
uv sync --extra dev
cp configs/eval/local_paths.example.yaml configs/eval/local_paths.yaml   # git-ignored
$EDITOR configs/eval/local_paths.yaml                                    # fill in dataset roots
```

A suite YAML never contains a dataset path: it writes `${HYPER3_...}`
placeholders that are resolved from the `--paths` file, layered over the
process environment. A name the suite uses and the paths file does not define
raises when the suite loads, so a typo fails immediately rather than halfway
through a run.

Every run is one command:

```bash
python scripts/evaluate.py \
    --checkpoint hyper3labs/hyper3-clip \
    --suite configs/eval/all_paper_tables.yaml \
    --paths configs/eval/local_paths.yaml \
    --output-dir results/hyper3-clip
```

`--checkpoint` accepts a directory (`config.yaml` + `model.safetensors`, or the
released artifact's layout), a training `.pt`, or a Hugging Face repo id — it
is passed straight to `Hyper3CLIP.from_pretrained(..., strict=True)`. The
image size, text length and embedding geometry come from the checkpoint's own
config, never from the suite. Other flags: `--device`, `--batch-size`,
`--precision {fp32,bf16,fp16}` (default `fp32`; the mixed-precision modes only
take effect on CUDA), `--max-items N` to truncate every dataset, `--tasks` to
run a subset, and `--force` to ignore the cache.

**Similarity.** Ranking uses `Hyper3CLIP.similarity`, the negative Lorentz
distance `-d_L`. The evaluator that produced the paper's numbers ranked by the
Lorentz inner product `<x,y>_L`; the two give the *identical ordering*, because
`-d_L = -acosh(-κ <x,y>_L)/√κ` is strictly decreasing in `-<x,y>_L` at fixed
curvature. Every top-k, argmax and per-class ranking is therefore unchanged.
The one place the choice is visible is a mean of scores across several prompt
templates (multi-label with more than one prompt), because averaging is not
invariant under a monotone reparametrisation — and both paper multi-label runs
use a single prompt.

**Preprocessing.** Classification, the ImageNet hierarchy and hierarchy
entailment use `build_eval_transform` (resize the short side, then centre
crop). Retrieval and multi-label use `build_retrieval_transform` — a square
resize with **no** crop, which is what the released checkpoint applies inside
its own preprocessing. Normalisation is ImageNet statistics, not CLIP's.

## Table 2 — retrieval and the ImageNet hierarchy

**Protocol.** Encode every image of the split and every caption of every image,
score all pairs, and take recall@{1,5,10} in both directions. *Text retrieval*
(`i2t_*`) ranks captions for an image and counts a hit when **any** of that
image's captions lands in the top k; *image retrieval* (`t2i_*`) ranks images
for a caption against its single correct image. The same run also reports the
five ImageNet WordNet hierarchy metrics.

**Data layout.**

```
${HYPER3_COCO2014_ROOT}/
    karpathy/dataset_coco.json        # the standard Karpathy split release
    train2014/  val2014/              # the directories the split file names
${HYPER3_FLICKR30K_ROOT}/
    dataset_flickr30k.json            # Karpathy-format split file
    flickr30k_images/
${HYPER3_IMAGENET_VAL_ROOT}/
    n01440764/ n01443537/ ...         # ImageFolder of 1000 WNID directories
    imagenet_label_to_wnid.tsv        # optional: "<label_int>\t<wnid>" per line
```

Captions are every `sentences[*].raw`, stripped (5 per COCO image, occasionally
6–7). The Karpathy `test` split is the reported one for both datasets.

**Hierarchy assets.** `all_synsets.pkl` (official 1000-label order),
`all_ancestors_indices.pkl` (ancestor indices per label) and `imagenet_isa.txt`
(space-separated `parent child` WordNet is-a edges) ship inside the package at
`src/hyper3_clip/evaluation/assets/`; see `assets/PROVENANCE.md`. Set
`imagenet_hierarchy_assets_root` in the task's `data` block only to override
them. ImageFolder class indices are mapped to official label indices through
`imagenet_label_to_wnid.tsv` when present, otherwise through the WNID order in
`all_synsets.pkl`.

With `A_p` / `A_t` the ancestor sets of the predicted and true official labels,
`I = A_p ∩ A_t` and `U = A_p ∪ A_t`:

| metric | definition | direction |
|---|---|---|
| TIE | undirected shortest-path length between the predicted and true synset in the is-a graph | lower is better |
| LCA | `\|A_p\| − \|I\| + 1` | lower is better |
| Jaccard | `\|I\| / \|U\|` | higher is better |
| H-P | `\|I\| / \|A_p\|` | higher is better |
| H-R | `\|I\| / \|A_t\|` | higher is better |

Each is summed per image and divided by the image count. The shortest path is
a breadth-first walk over the undirected is-a graph, so no graph library is
needed.

**Prompts.** ImageNet uses the seven-template ensemble:

```
i took a picture : itap of a {}.
pics : a bad photo of the {}.
pics : a origami {}.
pics : a photo of the large {}.
pics : a {} in a video game.
pics : art of the {}.
pics : a photo of the small {}.
```

Class names are torchvision's `_IMAGENET_CATEGORIES`, which is what produced
the paper's rows; `class_names: hycoclip` selects the official HyCoCLIP/UNCHA
1000-name list instead.

**Reproduce.**

```bash
python scripts/evaluate.py --checkpoint <ckpt> \
    --suite configs/eval/paper_table_2.yaml \
    --paths configs/eval/local_paths.yaml --output-dir results/<name>
```

Reported columns: `i2t_r1/r5/r10`, `t2i_r1/r5/r10` per dataset, then `tie`,
`lca`, `jaccard`, `hierarchical_precision`, `hierarchical_recall`.

## Table 3 — zero-shot classification

**Protocol.** Build one text classifier per dataset, predict `argmax` over
classes, and report **mean-per-class accuracy** — the mean over classes with at
least one image of that class's own accuracy. Plain top-1 is emitted next to it
as `top1_pct` but is not the paper's number.

**Prompt ensembling.** For each class, format every template with the class
name (underscores become spaces), encode them, average the 512-d **tangent**
projections, and lift that single average with `project_text_features`
(`exp_map0`). It is *not* a mean of Lorentz points, and there is no L2
renormalisation of the averaged tangent vector. This is the single detail most
likely to be re-implemented differently, and it moves every column.

**Data layout.** ImageNet is the same root as Table 2. Each of the other 15
datasets is an ImageFolder whose class directories are named
`f"{index:04d}_{slug}"`, so sorted folder order equals the order of the
vendored class-name list the task selects with `class_names_key`. A length
mismatch raises rather than silently mis-labelling. Class names come from the
official HyCoCLIP list shipped at `assets/class_names_hycoclip.py`.

```
${HYPER3_CIFAR10_ROOT}/0000_airplane/*.png
                       0001_automobile/*.png
                       ...
```

**Prompts per dataset.** Every set below lives in
`hyper3_clip.evaluation.prompts.PROMPT_SETS`.

| dataset | `prompt_set` | templates |
|---|---|---|
| ImageNet | `imagenet` | the seven above |
| CIFAR-10 / CIFAR-100 | `cifar` | 18: `a {blurry, black and white, low contrast, high contrast, bad, good, —} photo of a/the {}.` plus `a photo of a/the {small,big} {}.` |
| SUN397, STL-10 | `sun397`, `stl10` | `a photo of a {}.` ; `a photo of the {}.` |
| Caltech-101 | `caltech101` | 34: `a photo of a {}.`, `a painting of a {}.`, `a plastic {}.`, `a sculpture of a {}.`, `a sketch of a {}.`, `a tattoo of a {}.`, `a toy {}.`, `a rendition of a {}.`, `a embroidered {}.`, `a cartoon {}.`, `a {} in a video game.`, `a plushie {}.`, `a origami {}.`, `art of a {}.`, `graffiti of a {}.`, `a drawing of a {}.`, `a doodle of a {}.` and the 17 `the`-variants |
| Cars | `cars` | `a photo of a/the/my {}.`, `i love my {}!`, `a photo of my {dirty,clean,new,old} {}.` |
| Aircraft | `aircraft` | `a photo of a/the {}, a type of aircraft.` |
| Pets | `pets` | `a photo of a {}, a type of pet.` |
| DTD | `dtd` | `pics : {} texture./pattern./thing.` and the three `pics : this {} …` variants |
| EuroSAT | `eurosat` | `a centered satellite photo of {} / of a {} / of the {}.` |
| RESISC45 | `resisc45` | 18 = {satellite, aerial} × {imagery, photo, view} of {∅, a, the} `{}` |
| Country211 | `country211` | `a photo i took in {}.`, `a photo i took while visiting {}.`, `a photo from my home country of {}.`, `a photo from my visit to {}.`, `a photo showing the country of {}.` |
| **Food-101, CUB, Flowers-102** | Photo regime | **`a photo of a {}.`** |

The last row is the Table 3 default and is set with `prompt_regime: photo` in
the suite; those three datasets' official sets are `food : {}.` /
`food porn : {}.`, the four `bird pics : {}.` / `birding : {}.` / `birds : {}.`
/ `bird photography : {}.`, and `flowers : {}.`. Table 6 runs both regimes.

**Reproduce.**

```bash
python scripts/evaluate.py --checkpoint <ckpt> \
    --suite configs/eval/paper_table_3.yaml \
    --paths configs/eval/local_paths.yaml --output-dir results/<name>
```

Reported column: `mean_per_class_acc_pct`. The "Avg." column is the unweighted
mean over the 16 datasets and is computed by `scripts/summarize_results.py`.

## Table 4 — zero-shot multi-label mAP

**Protocol.** Score every image against every class prompt, rank each class's
images independently, and average the per-class average precision. AP ranks
descending with ties broken by insertion order and averages `hits(r)/r` over
the positives — no interpolation, no tie averaging. A class with no positive
returns nothing and is excluded from the mean; `valid_classes` reports how many
counted.

**Manifests.** Build both with `scripts/build_multilabel_manifest.py`:

```bash
python scripts/build_multilabel_manifest.py voc \
    --annotations-root /data/VOC2007/Annotations \
    --image-root /data/VOC2007/JPEGImages \
    --output /data/manifests/voc_multilabel.jsonl \
    --class-names-output /data/manifests/voc_classes.txt

python scripts/build_multilabel_manifest.py coco \
    --instances-json /data/coco2017/annotations/instances_val2017.json \
    --image-root /data/coco2017/val2017 \
    --output /data/manifests/coco_multilabel.jsonl \
    --class-names-output /data/manifests/coco_classes.txt
```

Each manifest row is
`{"image_path", "labels": [...], "subset": "all", "source_id"}`. The
class-names file is one class per line and its **order defines the score-matrix
column order**; VOC's is alphabetical, COCO's is category-id order. Labels are
the raw source spellings and images with no annotated object are absent.

**Prompt.** One template, `a photo of a {}.`, for both datasets. Prompt
ensembling here is a **mean of similarity scores** across templates, not of
embeddings — different from Table 3; with one prompt the two coincide.

**Reproduce.**

```bash
python scripts/evaluate.py --checkpoint <ckpt> \
    --suite configs/eval/paper_table_4.yaml \
    --paths configs/eval/local_paths.yaml --output-dir results/<name>
```

Reported column: `mean_average_precision_pct`. "Avg." is the unweighted mean of
the VOC and COCO mAPs, computed by the summarizer.

## Table 5 — hierarchy entailment (Sec. 5.1)

**Protocol.** Each sample is an image plus a caption hierarchy ordered coarse
to fine. Pairs are labelled and ranked by an entailment score, then scored with
average precision ("Hierarchy AP") and AUROC. Both use tie-aware conventions:
AUROC is the rank-sum statistic with average ranks over ties, AP is a step-wise
AP that collapses equal scores into one step. The reported run has 1,000 images
× 4 hierarchy levels = **4,000 positives** and 1,000 × 100 = **100,000
negatives**.

- **Positives** — every caption of the image's own hierarchy, paired with that
  image.
- **Negatives** — the finest caption (`positive_captions[-1]`) of every *other*
  sample in dataset order, with any caption that is also a positive for this
  image removed, then truncated to `max_negatives_per_image` (100). The
  construction is deterministic and takes no seed, so roughly the same 100
  captions serve as negatives for every image; `negatives: sampled` draws them
  at random with a seed instead.

**Scoring direction.** The image is the **specific** node and the caption the
**general** node that roots the entailment cone. The default `score:
entailment_score` is the paper's `p(a ⪯ b) = max(1 − 2φ/π, 0)`, i.e.
`Hyper3CLIP.entailment_score(general=text, specific=image)`. Set `score:
signed_margin` for the unclamped signed cone margin `−(φ − Θ(text))`, which is
the setting that produced the published Table 5 numbers. Every result
carries tie diagnostics
(`distinct_scores`, `modal_score_fraction`, `exactly_zero_fraction`) so a
saturated clamped score is visible rather than hidden inside an AUROC.

**Annotations file.** CSV, TSV, JSON or JSONL. Two shapes are accepted:

```csv
image_id,image_path,positive_captions
42,COCO_val2014_000000000042.jpg,"an animal => a bird => a small brown bird => a wren perched on a fence"
```

— one row per image, hierarchy in a `positive_captions` /
`hierarchical_captions` / `caption_hierarchy` / `captions` column, given as a
JSON list or split on `=>` (then `||`) — **order matters, the last entry is the
finest node**; or one row per `(image, caption, label)` triple with `caption`
and `label` columns, which is regrouped by image key into positives (`label ∈
{1, true, yes, positive, pos}`) and negatives, preserving row order. Image
paths resolve against `image_root`; a row that carries only an `image_url`
(the public HierarCaps release: `id,captions,image_url`) uses the URL's file
name, so `image_root` should point at the COCO `val2014` directory.

**Avg R@10.** The suite also runs two retrieval tasks; the ablation's
"Avg R@10" is the unweighted mean of their four R@10 values. The reported
ablation used COCO **val2017** here, not the Table 2 Karpathy split,
which is why `configs/eval/hierarchy_entailment.yaml` names
`coco_val2017_retrieval`.

**Reproduce.**

```bash
python scripts/evaluate.py --checkpoint <ckpt> \
    --suite configs/eval/hierarchy_entailment.yaml \
    --paths configs/eval/local_paths.yaml --output-dir results/<variant>

python scripts/ablation_table.py \
    --baseline results/base_80k \
    --results results/full_objective results/no_query results/visual_hierarchy_only
```

`ablation_table.py` prints the paper's column order: variant, Hierarchy AP,
AUROC, Avg R@10, ΔAP and ΔR@10 against the named baseline directory.

## Table 6 — prompt sensitivity (Sec. 5.2)

**Protocol.** The Table 3 evaluator, run twice on Food-101, CUB and
Flowers-102: once with each dataset's official prompt set, once with the Photo
prompt `a photo of a {}.`. Reported are the six cells, the three-dataset
average per regime, and the 16-dataset average recomputed with only those three
columns swapped.

```bash
python scripts/evaluate.py --checkpoint <ckpt> \
    --suite configs/eval/prompt_sensitivity.yaml \
    --paths configs/eval/local_paths.yaml --output-dir results/<name>
```

Point `--output-dir` at the **same** directory as the Table 3 run: the
16-dataset average needs the other 13 columns, and the summarizer leaves it
blank when they are absent.

## Sec. 5.3 — GRIT parts-per-image scan

```bash
python scripts/scan_grit_parts.py --tarfiles '/data/grit-processed/*.tar'
```

The scan reads each sample's `numparents.txt` from every shard's tar index and
accumulates the histogram `H` of localized parts per example. Everything Sec.
5.3 quotes is arithmetic on `H`, with `N = Σ_k H[k]`:

| statistic | formula |
|---|---|
| examples | `N` |
| localized parts | `Σ_k k·H[k]` |
| mean parts per example | `Σ_k k·H[k] / N` |
| max parts | `max(k : H[k] > 0)` |
| fraction with exactly one part | `H[1] / N` |
| fraction with one or two parts | `(H[1] + H[2]) / N` |
| part instances retained at cap `m` | `Σ_k min(k, m)·H[k] / Σ_k k·H[k]` |
| examples untruncated at cap `m` | `Σ_{k ≤ m} H[k] / N` |

The paper's full-corpus scan reports 2,051 shards, 13,149,251 examples,
25,185,017 parts, mean 1.9153, max 20, 40.68% with one part, 79.23% with one or
two, and at cap 5 99.38% of part instances retained with 99.20% of examples
untruncated. The cap the sweep varies is the training-side
`ProcessedGritDataset(max_parts=...)`; `configs/paper/max_parts_sweep/` holds
the 10k-step proxy runs for caps 2–6 and uncapped. The scanner emits a row per
cap in 2..6 plus `"all"`, so the retention half of the sweep needs no separate
arithmetic.

## Results, caching and summaries

Each task writes `<output-dir>/<model-id>/<task-id>.json`:

```json
{
  "suite": "paper_table_2",
  "created_at": "...",
  "model": {"id": "...", "group": null, "checkpoint": "...", "checkpoint_signature": {...}},
  "task": {"id": "...", "name": "...", "dataset": "...", "data": {...}, "options": {...}},
  "device": "cuda", "precision": "fp32", "elapsed_seconds": 123.4,
  "cache_key": "<sha256>",
  "results": {"i2t_r5": 61.2, ...}
}
```

The cache key covers the suite, the task spec and options, the checkpoint
signature, the device type, the precision and the `--max-items` override. A
re-run whose key matches an existing record prints `cache_hit` and skips the
work; `--force` re-runs regardless. Nothing in a record is a private path
beyond the dataset roots you supplied.

```bash
python scripts/summarize_results.py --results-dir results/hyper3-clip --print
```

writes `summary_long.csv` (one row per metric), `summary_wide.csv` (one row per
model, `{task_id}.{metric}` columns) and `table_2.md` … `table_6.md` for
whichever tables the directory covers, each in the paper's column order.
Retrieval metrics keep their plain names — read `i2t_r5` / `i2t_r10` for text
retrieval and `t2i_r5` / `t2i_r10` for image retrieval.

## Smoke test

The whole pipeline runs on synthetic data with no downloads, which is what
`tests/test_eval_suite.py` exercises and the fastest way to check an
installation:

```bash
python tests/fixtures/make_eval_fixtures.py /tmp/eval-fixtures --checkpoint

python scripts/evaluate.py \
    --checkpoint /tmp/eval-fixtures/checkpoint/model \
    --suite /tmp/eval-fixtures/smoke_suite.yaml \
    --paths /tmp/eval-fixtures/paths.yaml \
    --output-dir /tmp/eval-fixtures/results \
    --model-id smoke-tiny --device cpu --max-items 4

python scripts/summarize_results.py --results-dir /tmp/eval-fixtures/results --print
```

The fixture generator writes a Karpathy-format COCO split, a Flickr30K split, a
COCO val2017 captions file, an ImageFolder, an ImageNet-style WNID tree, a VOC
`Annotations` directory, a COCO `instances.json`, a multi-label manifest and a
HierarCaps-style CSV, plus the `paths.yaml` and `smoke_suite.yaml` the driver
needs and (with `--checkpoint`) a randomly initialised tiny model. The numbers
are meaningless — what the run proves is that every task type loads its data,
produces the metric keys the tables read, and summarizes.
