"""Evaluation code for the paper's tables.

======================================================  ==========================
:mod:`hyper3_clip.evaluation.retrieval`                 Table 2 retrieval, Table 5 Avg R@10
:mod:`hyper3_clip.evaluation.imagenet`                  Table 2 hierarchy, Table 3 "IN"
:mod:`hyper3_clip.evaluation.classification`            Tables 3 and 6 zero-shot classification
:mod:`hyper3_clip.evaluation.multilabel`                Table 4 multi-label mAP
:mod:`hyper3_clip.evaluation.hierarchy_entailment`      Table 5 hierarchy AP / AUROC
:mod:`hyper3_clip.evaluation.grit_stats`                Sec. 5.3 parts-per-image scan
:mod:`hyper3_clip.evaluation.runner`                    the suite driver
:mod:`hyper3_clip.evaluation.reporting`                 CSVs and the paper's tables
======================================================  ==========================

``docs/evaluation.md`` documents the protocol, the dataset layouts and every
place the evaluation that produced the published numbers differs from the
paper's wording.
"""

from __future__ import annotations

from hyper3_clip.evaluation.classification import evaluate_imagefolder_zero_shot
from hyper3_clip.evaluation.config import SuiteSpec, TaskSpec, load_suite, load_variables
from hyper3_clip.evaluation.grit_stats import scan_grit_parts
from hyper3_clip.evaluation.hierarchy_entailment import evaluate_hierarchy_entailment
from hyper3_clip.evaluation.imagenet import evaluate_imagenet
from hyper3_clip.evaluation.multilabel import evaluate_multilabel_zero_shot, load_multilabel_samples
from hyper3_clip.evaluation.prompts import IMAGENET_PROMPTS, PHOTO_PROMPTS, PROMPT_SETS, resolve_prompts
from hyper3_clip.evaluation.retrieval import evaluate_caption_retrieval
from hyper3_clip.evaluation.runner import EvalRunner, ModelSpec

__all__ = [
    "EvalRunner",
    "IMAGENET_PROMPTS",
    "ModelSpec",
    "PHOTO_PROMPTS",
    "PROMPT_SETS",
    "SuiteSpec",
    "TaskSpec",
    "evaluate_caption_retrieval",
    "evaluate_hierarchy_entailment",
    "evaluate_imagefolder_zero_shot",
    "evaluate_imagenet",
    "evaluate_multilabel_zero_shot",
    "load_multilabel_samples",
    "load_suite",
    "load_variables",
    "resolve_prompts",
    "scan_grit_parts",
]
