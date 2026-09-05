# Method: equations to code

This page maps the paper's Sections 3.1-3.3 onto the modules that implement
them, then summarises the model, its preprocessing and the checkpoint formats.

- [Query construction (Sec. 3.1)](#query-construction-sec-31)
- [Query-conditioned visual pooling (Sec. 3.2)](#query-conditioned-visual-pooling-sec-32)
- [Training objective (Sec. 3.3)](#training-objective-sec-33)
- [Model](#model)
- [Checkpoint formats](#checkpoint-formats)
- [Preprocessing and public API](#preprocessing-and-public-api)

## Query construction (Sec. 3.1)

`hyper3_clip.data.query_hierarchy.build_query_hierarchy(caption, text_boxes)`
implements the rules stated in Sec. 4 of the paper. Candidates are added in a
fixed order — caption root, sentence queries, text-box queries, extracted
phrases — and each is normalized by collapsing whitespace runs and accepted
only if it is at least 3 characters long and its case-folded text has not
already been accepted.

| kind | parent | weight | budget |
|---|---|---|---|
| caption root | none (`-1`) | 0.0 | always first |
| sentence | the caption | 1.0 | first 5 raw split fragments |
| text box | the caption | 0.75 | shares a 30-query budget with phrases, text boxes first |
| phrase | the first accepted sentence, else the caption | 0.5 | the rest of that budget |

The list is then truncated to `max_queries_per_image = 6` in construction
order. Sentences split on `.!?;` followed by whitespace, and on line breaks.
Phrases split on `,;:()` and on the connector words *and, with, near, beside,
behind, under, above, around, next to*; a 2-8 word chunk becomes one phrase, a
longer chunk becomes 6-word windows starting every 4 words, and fragments under
2 words are dropped.

Four behaviours are worth stating because the paper does not:

1. The 5-sentence cap applies to the **raw** split fragments, before dedupe and
   the length filter, so at most 5 are offered and possibly fewer accepted. A
   single-sentence caption yields a sentence identical to the caption, which
   the dedupe then drops — such captions have **zero** sentence queries.
2. Each query carries a **weight** (the table above) and a **type**. The weight
   is handed to the objective as `query_weight` and scales that query's
   contribution.
3. A text-box query records the **part index** it came from, as
   `query_source_part`. Empty box texts are dropped but the surviving ones keep
   their true part index, so the recorded indices can be non-contiguous. The
   objective carries this field through without reading it.
4. If the caption root is rejected (empty, whitespace-only, under 3 characters)
   the fallback parent index is `0`, which then points at whatever query landed
   there — possibly the query itself.

Parent indices are assigned **before** truncation. Because a parent always has
a lower index than its children, truncation never orphans a query except
through case 4. `tests/test_query_hierarchy.py` pins all of this against a
golden file of 40 captions.

### Collation

`collate_grounded(batch, tokenizer, max_text_length, queries=QueryConfig())`
packs a batch. Parts and queries are packed, not padded: each image contributes
its own rows, with `part_owner` / `query_owner` recording ownership, and the
per-image `parent` / `source_part` indices are rewritten to batch-global row
indices (`-1` stays `-1`). Captions, box texts and queries are tokenized
separately with longest-in-batch padding.

## Query-conditioned visual pooling (Sec. 3.2)

`hyper3_clip.models.query_pooling.QueryConditionedPooling`:

```
v~_q = MHA(Q = W_Q h_q, K = Z_I, V = Z_I)
v_q  = v~_q + MLP(LN(v~_q))
```

with 8 heads, MLP hidden dim `4 * d_v` and GELU. `h_q` is the query's text
embedding, `Z_I` the 196 patch tokens of the image that owns the query (the CLS
token is always dropped), and `v_q` the pooled visual node. The implementation
indexes the owner's tokens per query and makes a single
`nn.MultiheadAttention` call.

Two ablation controls keep the parameter layout identical, so a row can resume
the shared base checkpoint strictly:

* `query_pooling_mode="mean"` replaces the conditioned attention with a plain
  mean over the patch tokens, plus a zero-valued call through the pooling
  parameters so they stay in the autograd graph (DDP's reduction invariant).
* `query_parent_mode="shuffled"` rolls the non-root parent assignments by one:
  the same number of hierarchy edges, none of them the true one.

## Training objective (Sec. 3.3)

```
L = L_con + lambda_ent * ( L_ent + L_query )
```

| paper | code (`compute_objective` key, producer) |
|---|---|
| `L` | `["loss"]` |
| `L_con` | `["contrastive"]` — `contrastive.contrastive_losses` |
| `L_ent` | `["entailment_base"]` — `entailment.base_entailment` |
| `L_query` | `["entailment_query"]` — `entailment.query_entailment` |
| `lambda_ent` | `ObjectiveConfig.entail_weight` (0.2) |

The relations are summed with unit weights; there is no norm regularizer and no
aggregate-consistency term.

### Geometry

All nodes are points `x = (x_0, x_bar)` on the Lorentz hyperboloid of curvature
`-kappa`, produced by `geometry.lorentz.exp_map0`.

| quantity | formula | code |
|---|---|---|
| inner product | `<x,y>_L = -x_0 y_0 + <x_bar, y_bar>` | `lorentz_inner` |
| distance | `d_L(x,y) = acosh(-kappa <x,y>_L) / sqrt(kappa)` | `lorentz_distance` |
| half-aperture | `Theta(b) = asin( 0.2 / (||b_bar|| sqrt(kappa)) )` | `half_aperture` (`min_radius = 0.1`) |
| exterior angle | `phi(a,b)`, the MERU "oxy" angle | `exterior_angle` |
| cone violation | `E(a <= b) = max(phi(a,b) - s Theta(b), 0)` | `cone_violation` |
| entailment score | `max(1 - 2 phi / pi, 0)` | `entailment_score` (hierarchy eval) |

Direction convention: in `E(a <= b)` the **second** argument `b` roots the cone
and `a` is penalized when it falls outside it.

### Contrastive term `L_con`

Three cross-entropy groups over `-d_L` logits scaled by
`exp(logit_scale).clamp(max=100)`, each the symmetric average of both
directions: `contrastive_global` (`I` vs. all `T`), `contrastive_local`
(`I^box` vs. all `T^box`) and `contrastive_global_local` (`I^box` vs. the
owner-repeated whole captions, `T^box` vs. the owner-repeated whole images).

In the global-local group each row's distances are first multiplied by a
detached uncertainty temperature `clamp(exp(-0.5 u), 0.1, 10)`,
`u = softplus(-||x_bar||)` (`contrastive.embedding_uncertainty`).

### Inherited entailment `L_ent`

```
L_ent = E(I <= T) + E(I^box <= T^box) + E(I <= I^box) + E(T <= T^box) + C_img + C_txt
```

| relation | aperture scale | code key |
|---|---|---|
| `I <= T` | inter | `entailment_image_text` |
| `I^box <= T^box` | inter | `entailment_part_image_text` |
| `I <= I^box` | intra | `entailment_image_part_image` (+ `calibration_image_part_image`) |
| `T <= T^box` | intra | `entailment_text_part_text` (+ `calibration_text_part_text`) |

### Query entailment `L_query`

For every **non-root** query `q` (`parent >= 0`) with `w(q) > 0`:

```
L_query = ( 1 / N ) * sum_q w(q) * [ E(I^q <= T^q) + E(I <= I^q) + E(T^pi(q) <= T^q) ] + C_vis + C_txt
```

with `N` the number of contributing queries (`["query_count"]`).

| relation | aperture scale | switch | code key |
|---|---|---|---|
| (1) `I^q <= T^q` | inter | `query_image_text_weight` | `entailment_query_image_text` |
| (2) `I <= I^q` | intra | `visual_hierarchy_weight` | `entailment_query_visual` |
| (3) `T^pi(q) <= T^q` | intra | `text_hierarchy_weight` | `entailment_query_text` |

`reverse_visual_order=True` swaps relation (2) to `E(I^q <= I)`, the paper's
direction control. `w(q)` comes from `data.query_hierarchy.QUERY_WEIGHTS`:
caption root 0.0 (roots never contribute), sentence 1.0, box 0.75, phrase 0.5.

### Uncertainty calibration

`u(x) = softplus(-||x_bar||)` is a radius-derived log-uncertainty (a node near
the origin is generic, hence uncertain). For residual `r` and general node `b`,

```
C = mean_q alpha * ( 0.5 * stopgrad(r) / exp(u(b)) + 0.5 * u(b) )
```

(`entailment.calibration_rows`, `alpha = calibration_alpha = 10`,
`stop_grad_calibration = True`) — a Gaussian negative log-likelihood training
the radii to predict how badly a relation is violated; with stop-grad it never
rescales the entailment gradient itself. Calibration is always on, as in the
paper.

The paper does not say which node's radius defines the uncertainty, nor which
relations are calibrated; both follow the inherited UNCHA objective: the
**general** node's radius, on the within-modality relations only (`I <= I^box`,
`T <= T^box`, `I <= I^q`, `T^pi(q) <= T^q`). Unlike that objective, the
batch-level entropy constant `-sum softmax(u) log softmax(u)` is **not** added
to every row: it has no counterpart in the paper and makes the loss depend on
batch composition.

### Config key -> paper symbol

| config key | symbol / role | value |
|---|---|---|
| `entail_weight` | `lambda_ent` | 0.2 |
| `inter_` / `intra_aperture_scale` | `s` in `s Theta(b)`, cross-modal / within-modality | 0.7 / 1.2 |
| `calibration_alpha`, `stop_grad_calibration` | `alpha`, and the `r` detach — inherited UNCHA constants | 10.0, true |
| `query_image_text_weight`, `visual_hierarchy_weight`, `text_hierarchy_weight` | weights of query relations (1), (2), (3) | 1.0 each |
| `reverse_visual_order` | the `I^q <= I` direction control | false |

## Model

`hyper3_clip.models.Hyper3CLIP` is a dual encoder (ViT-B/16 vision tower and a
CLIP-B/32 text tower, both trained from scratch) whose 512-d projections are
lifted onto a Lorentz hyperboloid of learned curvature. Query-conditioned
pooling is a **training-time-only** module: at inference the model is a plain
dual encoder and costs exactly what CLIP costs. Learned scalars are the
contrastive temperatures, the per-modality scales `visual_alpha` /
`textual_alpha`, and `log_curv`.

### Geometry at inference

* `kappa = clamp(exp(log_curv), curv_init/10, curv_init*10)`, i.e. `[0.1, 10]`
  at the default `curv_init=1.0`.
* Lift: `exp_map0(x * exp(alpha), kappa)` maps a 512-d tangent vector to a
  513-d point on the hyperboloid; index 0 is the time coordinate, whose floor is
  `1/sqrt(kappa)`.
* Retrieval score: `similarity(a, b) = -d_L(a, b)`, the negative geodesic
  distance. Ranking by the Lorentz inner product `<a,b>_L` gives the *same*
  order, because `-d_L = -acosh(-kappa <a,b>_L)/sqrt(kappa)` is strictly
  decreasing in `-<a,b>_L` at fixed curvature. The inner product is a plain dot
  product with the time coordinate negated, which is why it is the cheaper
  score for vector databases; the distance is the score the paper reports.
* `entailment_score(general, specific)` scores how far `specific` lies inside
  `general`'s entailment cone: `clamp(1 - 2*phi/pi, 0, 1)`, so `1` on the cone
  axis and `0` once the specific node is a right angle or more outside.

## Checkpoint formats

`Hyper3CLIP.from_pretrained(path_or_hf_id)` reads two layouts and detects which
one it has from the state-dict keys:

* **This repository's format**: a directory with `config.yaml` (a serialized
  `Hyper3CLIPConfig`) and `model.safetensors`, written by
  `model.save_pretrained(dir)`; a training checkpoint `.pt` is also accepted.
* **The released Hugging Face artifact**
  (<https://huggingface.co/hyper3labs/hyper3-clip>): `config.json` +
  `model.safetensors`. Its single `logit_scale` is broadcast to the three
  temperatures (inference-inert) and its pooling flavour is inferred from the
  keys.

**The released artifact is not the paper model.** It comes from an earlier run
with average pooling, one temperature, learned positional embeddings and no
query hierarchy. The paper checkpoint will be published in this repository's
own format.

## Preprocessing and public API

`hyper3_clip.models.preprocessing` reads `image_size` / `text_model_name` off a
config:

* `build_train_transform(config)` — `RandomResizedCrop(224, scale=(0.8,1.0),
  bicubic)`, `ColorJitter(0.4,0.4,0.4,0.1)` with `p=0.8`,
  `RandomGrayscale(p=0.2)`, `ToTensor`, ImageNet `Normalize`. No horizontal
  flip anywhere, and ImageNet statistics rather than CLIP's.
* `build_eval_transform(config)` — short-side `Resize(224)`, `CenterCrop(224)`.
* `build_retrieval_transform(config)` — `Resize((224,224))`, no crop. This is
  what the released artifact's own `preprocess_image` does, so reproducing its
  retrieval numbers uses this one.
* `build_tokenizer(config)` — the CLIP tokenizer from `text_model_name`. It is
  fetched from the hub once and cached; for offline use pre-populate the cache
  and set `HF_HUB_OFFLINE=1`, or point `text_model_name` at a local directory.

```python
from hyper3_clip.models import Hyper3CLIP, Hyper3CLIPConfig, ObjectiveConfig

model = Hyper3CLIP(Hyper3CLIPConfig())          # or Hyper3CLIP.from_pretrained(...)
model.encode_image(pixel_values)                 # -> [B, 513] Lorentz points
model.encode_text(input_ids, attention_mask)     # -> [B, 513]
model.encode_image_tangent(pixel_values)         # -> [B, 512] before the lift
model.encode_text_tangent(input_ids, mask)       # -> [B, 512]
model.similarity(image_feats, text_feats)        # -> [B_i, B_t]  = -d_L
model.entailment_score(general, specific)        # -> [B] in [0, 1], row-wise
model.curvature                                  # -> kappa, clamped
model(batch, step=...)                           # training nodes for compute_objective
model.compute_loss(batch, step=...)              # nodes -> objective, in one call
model.loss_from_nodes(nodes)                     # the objective alone (DDP-friendly)
model.save_pretrained(directory)
```

`forward` and `loss_from_nodes` are split so a DDP-wrapped module can produce
the nodes while the parameter-free objective is evaluated outside the wrapper;
`compute_loss` is the single-process convenience that chains them.
