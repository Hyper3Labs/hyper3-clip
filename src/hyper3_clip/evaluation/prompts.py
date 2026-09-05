"""Zero-shot prompt templates and the two prompt regimes (paper Tables 3 and 6).

Every template below is the one the reported runs used.  ``PROMPT_SETS`` is the
per-dataset *official* prompt set inherited from the baselines the paper
compares against; :data:`PHOTO_PROMPTS` is the single-template *Photo* regime.

Table 3 reports the Photo regime for Food-101, CUB and Flowers-102 and the
official set everywhere else.  Table 6 (paper Sec. 5.2, "prompt sensitivity")
runs both regimes on exactly those three datasets and reports both the
three-dataset and the 16-dataset averages.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "IMAGENET_PROMPTS",
    "PHOTO_PROMPTS",
    "PROMPT_REGIMES",
    "PROMPT_SENSITIVE_DATASETS",
    "PROMPT_SETS",
    "resolve_prompts",
]

#: ImageNet prompt ensemble (paper Table 3, column "IN").
IMAGENET_PROMPTS: tuple[str, ...] = (
    "i took a picture : itap of a {}.",
    "pics : a bad photo of the {}.",
    "pics : a origami {}.",
    "pics : a photo of the large {}.",
    "pics : a {} in a video game.",
    "pics : art of the {}.",
    "pics : a photo of the small {}.",
)

#: The Photo regime: one generic template, used for the three prompt-sensitive
#: datasets in Table 3 and for every dataset in the "Photo" half of Table 6.
PHOTO_PROMPTS: tuple[str, ...] = ("a photo of a {}.",)

#: The two prompt regimes of Table 6.
PROMPT_REGIMES: tuple[str, ...] = ("official", "photo")

#: The three datasets Table 6 varies; Table 3 reports them in the Photo regime.
PROMPT_SENSITIVE_DATASETS: tuple[str, ...] = ("food101", "cub2011", "flowers102")

#: Official per-dataset prompt sets, keyed the same way as ``class_names_key``.
PROMPT_SETS: dict[str, tuple[str, ...]] = {
    "imagenet": IMAGENET_PROMPTS,
    "cifar": (
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a black and white photo of a {}.",
        "a low contrast photo of a {}.",
        "a high contrast photo of a {}.",
        "a bad photo of a {}.",
        "a good photo of a {}.",
        "a photo of a small {}.",
        "a photo of a big {}.",
        "a photo of the {}.",
        "a blurry photo of the {}.",
        "a black and white photo of the {}.",
        "a low contrast photo of the {}.",
        "a high contrast photo of the {}.",
        "a bad photo of the {}.",
        "a good photo of the {}.",
        "a photo of the small {}.",
        "a photo of the big {}.",
    ),
    "cars": (
        "a photo of a {}.",
        "a photo of the {}.",
        "a photo of my {}.",
        "i love my {}!",
        "a photo of my dirty {}.",
        "a photo of my clean {}.",
        "a photo of my new {}.",
        "a photo of my old {}.",
    ),
    "food101": (
        "food : {}.",
        "food porn : {}.",
    ),
    "cub2011": (
        "bird pics : {}.",
        "birding : {}.",
        "birds : {}.",
        "bird photography : {}.",
    ),
    "sun397": (
        "a photo of a {}.",
        "a photo of the {}.",
    ),
    "aircraft": (
        "a photo of a {}, a type of aircraft.",
        "a photo of the {}, a type of aircraft.",
    ),
    "dtd": (
        "pics : {} texture.",
        "pics : {} pattern.",
        "pics : {} thing.",
        "pics : this {} texture.",
        "pics : this {} pattern.",
        "pics : this {} thing.",
    ),
    "pets": ("a photo of a {}, a type of pet.",),
    "caltech101": (
        "a photo of a {}.",
        "a painting of a {}.",
        "a plastic {}.",
        "a sculpture of a {}.",
        "a sketch of a {}.",
        "a tattoo of a {}.",
        "a toy {}.",
        "a rendition of a {}.",
        "a embroidered {}.",
        "a cartoon {}.",
        "a {} in a video game.",
        "a plushie {}.",
        "a origami {}.",
        "art of a {}.",
        "graffiti of a {}.",
        "a drawing of a {}.",
        "a doodle of a {}.",
        "a photo of the {}.",
        "a painting of the {}.",
        "the plastic {}.",
        "a sculpture of the {}.",
        "a sketch of the {}.",
        "a tattoo of the {}.",
        "the toy {}.",
        "a rendition of the {}.",
        "the embroidered {}.",
        "the cartoon {}.",
        "the {} in a video game.",
        "the plushie {}.",
        "the origami {}.",
        "art of the {}.",
        "graffiti of the {}.",
        "a drawing of the {}.",
        "a doodle of the {}.",
    ),
    "flowers102": ("flowers : {}.",),
    "stl10": (
        "a photo of a {}.",
        "a photo of the {}.",
    ),
    "eurosat": (
        "a centered satellite photo of {}.",
        "a centered satellite photo of a {}.",
        "a centered satellite photo of the {}.",
    ),
    "resisc45": (
        "satellite imagery of {}.",
        "aerial imagery of {}.",
        "satellite photo of {}.",
        "aerial photo of {}.",
        "satellite view of {}.",
        "aerial view of {}.",
        "satellite imagery of a {}.",
        "aerial imagery of a {}.",
        "satellite photo of a {}.",
        "aerial photo of a {}.",
        "satellite view of a {}.",
        "aerial view of a {}.",
        "satellite imagery of the {}.",
        "aerial imagery of the {}.",
        "satellite photo of the {}.",
        "aerial photo of the {}.",
        "satellite view of the {}.",
        "aerial view of the {}.",
    ),
    "country211": (
        "a photo i took in {}.",
        "a photo i took while visiting {}.",
        "a photo from my home country of {}.",
        "a photo from my visit to {}.",
        "a photo showing the country of {}.",
    ),
}


def resolve_prompts(
    prompts: Sequence[str] | None = None,
    prompt_set: str | None = None,
    prompt_regime: str | None = None,
) -> tuple[str, ...]:
    """Resolve the prompt templates for one task.

    Precedence: an explicit ``prompts`` list wins, then ``prompt_regime``
    (``"photo"`` short-circuits to :data:`PHOTO_PROMPTS`, ``"official"`` falls
    through to ``prompt_set``), then ``prompt_set``.  With nothing set at all
    the ImageNet ensemble is used, matching the evaluator that produced the
    paper numbers.
    """
    if prompts:
        return tuple(str(prompt) for prompt in prompts)
    if prompt_regime is not None:
        if prompt_regime not in PROMPT_REGIMES:
            raise ValueError(f"prompt_regime must be one of {list(PROMPT_REGIMES)}, got {prompt_regime!r}")
        if prompt_regime == "photo":
            return PHOTO_PROMPTS
    if prompt_set is None:
        return IMAGENET_PROMPTS
    if prompt_set not in PROMPT_SETS:
        raise ValueError(f"Unknown prompt_set {prompt_set!r}; expected one of {sorted(PROMPT_SETS)}")
    return PROMPT_SETS[prompt_set]
