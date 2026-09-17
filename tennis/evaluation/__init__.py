"""Scoring pipeline output against hand-made labels (spec section 10.3).

One module per ``tennis eval-*`` command. None of them re-runs a stage: they read the
session's Parquet files and re-apply only the cheap decision rule being scored.
"""

from __future__ import annotations

from tennis.evaluation.contacts import (
    DEFAULT_SPEEDS,
    DEFAULT_WINDOWS,
    ContactEval,
    EvalRow,
    evaluate_contacts,
    format_contact_eval,
    write_contact_eval,
)
from tennis.evaluation.labels import (
    LabelEval,
    evaluate_labels,
    format_label_eval,
    write_label_eval,
)
from tennis.evaluation.strokes import (
    ClassifierEval,
    ConfusionMatrix,
    evaluate_classifier,
    format_classifier_eval,
    write_classifier_eval,
)

__all__ = [
    "DEFAULT_SPEEDS",
    "DEFAULT_WINDOWS",
    "ClassifierEval",
    "ConfusionMatrix",
    "ContactEval",
    "EvalRow",
    "LabelEval",
    "evaluate_classifier",
    "evaluate_contacts",
    "evaluate_labels",
    "format_classifier_eval",
    "format_contact_eval",
    "format_label_eval",
    "write_classifier_eval",
    "write_contact_eval",
    "write_label_eval",
]
