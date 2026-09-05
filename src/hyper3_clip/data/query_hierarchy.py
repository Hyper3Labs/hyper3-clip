"""Rule-based query-hierarchy construction for Hyper3-CLIP (paper Sec. 3.1).

Implements the "Query construction" rules of paper Sec. 4, including the edge
cases the paper does not spell out:

* the 5-sentence cap is applied to the *raw* split fragments (before dedupe /
  length filtering), so a single-sentence caption offers its caption as a
  sentence and the dedupe then drops it;
* text boxes and phrases share a 30-query budget, text boxes first;
* a phrase's parent is the first accepted sentence query, else the caption;
* caption-root rejection falls back to index 0;
* the final list is truncated to ``max_queries`` in construction order.

Each accepted query carries a reliability ``weight`` (caption 0.0, sentence
1.0, text box 0.75, phrase 0.5) and a ``source_part`` index linking a text-box
query to its part image (``-1`` otherwise).  These feed the training objective
as the per-query hierarchy and reliability terms.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Query",
    "QUERY_WEIGHTS",
    "TYPE_CODES",
    "build_query_hierarchy",
    "split_sentences",
    "extract_lightweight_phrases",
]

QueryKind = Literal["caption", "sentence", "text_box", "phrase"]

#: Reliability weight per query kind (paper Sec. 4).
QUERY_WEIGHTS: dict[str, float] = {
    "caption": 0.0,
    "sentence": 1.0,
    "text_box": 0.75,
    "phrase": 0.5,
}

#: Numeric type code -> kind name, in construction order.
TYPE_CODES: dict[int, str] = {0: "caption", 1: "sentence", 2: "text_box", 3: "phrase"}

_KIND_TO_CODE: dict[str, int] = {v: k for k, v in TYPE_CODES.items()}

_CONNECTOR_SPLIT = re.compile(r"[,;:()]|\s+(?:and|with|near|beside|behind|under|above|around|next to)\s+", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


@dataclass(frozen=True)
class Query:
    """A single query node in the image-text hierarchy.

    ``parent`` indexes the parent query (``-1`` for the caption root).
    ``source_part`` indexes the part image row for a text-box query (``-1``
    otherwise).  ``kind`` is one of ``{"caption","sentence","text_box",
    "phrase"}``.
    """

    text: str
    kind: QueryKind
    parent: int
    weight: float
    source_part: int


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences on ``.!?;`` followed by whitespace or newlines."""
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def extract_lightweight_phrases(text: str) -> list[str]:
    """Extract phrases per the lightweight rule.

    Chunks are split on commas/semicolons/colons/parentheses and connector
    words; 2-8 word chunks become one phrase; >8-word chunks are windowed in
    run of 6 words starting every 4 (overlapping by 2).  Sub-2-word chunks are
    dropped.
    """
    phrases: list[str] = []
    for chunk in _CONNECTOR_SPLIT.split(text):
        words = _WORD_RE.findall(chunk)
        if 2 <= len(words) <= 8:
            phrases.append(" ".join(words))
        elif len(words) > 8:
            for start in range(0, len(words) - 1, 4):
                phrase = " ".join(words[start : start + 6])
                if len(phrase.split()) >= 2:
                    phrases.append(phrase)
    return phrases


def build_query_hierarchy(
    caption: str,
    text_boxes: Sequence[str],
    *,
    max_sentences: int = 5,
    max_phrases: int = 30,
    max_queries: int = 6,
    use_text_boxes: bool = True,
) -> list[Query]:
    """Build the query hierarchy for one image.

    ``text_boxes`` holds the per-part box texts in part order; a text-box query
    records its original ``source_part`` index so the caller can recover the
    part-image association after a non-contiguous accepted set.
    """
    queries: list[str] = []
    query_items: list[Query] = []
    seen: set[str] = set()

    def add_query(
        text: str, *, query_type: int, parent: int = 0, weight: float = 1.0, source_part: int = -1
    ) -> int:
        normalized = " ".join(str(text).strip().split())
        key = normalized.casefold()
        if len(normalized) >= 3 and key not in seen:
            seen.add(key)
            queries.append(normalized)
            query_items.append(
                Query(
                    text=normalized,
                    kind=TYPE_CODES[query_type],
                    parent=parent,
                    weight=weight,
                    source_part=source_part,
                )
            )
            return len(queries) - 1
        return -1

    caption_index = add_query(caption, query_type=0, parent=-1, weight=0.0)
    if caption_index < 0:
        caption_index = 0

    sentence_indices: list[int] = []
    for sentence in split_sentences(caption)[: max(0, max_sentences)]:
        sentence_index = add_query(sentence, query_type=1, parent=caption_index, weight=1.0)
        if sentence_index >= 0:
            sentence_indices.append(sentence_index)
    phrase_parent = sentence_indices[0] if sentence_indices else caption_index

    phrase_count = 0
    if use_text_boxes:
        for part_index, part_text in enumerate(text_boxes):
            if phrase_count >= max_phrases:
                break
            before = len(queries)
            add_query(part_text, query_type=2, parent=caption_index, weight=0.75, source_part=part_index)
            phrase_count += int(len(queries) > before)

    if phrase_count < max_phrases:
        for phrase in extract_lightweight_phrases(caption):
            if phrase_count >= max_phrases:
                break
            before = len(queries)
            add_query(phrase, query_type=3, parent=phrase_parent, weight=0.5)
            phrase_count += int(len(queries) > before)

    return query_items[: max(1, max_queries)] if max_queries is not None else query_items
