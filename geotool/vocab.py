"""Diagnosis vocabulary: a JSON list, not a code enum, since it will grow.

New diagnosis categories get added to vocab_data/diagnoses.json (or a
user-supplied file via GEOTOOL_DIAGNOSIS_VOCAB) without any code change.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from geotool import config

DEFAULT_DIAGNOSIS_VOCAB_PATH = config.PROJECT_ROOT / "geotool" / "vocab_data" / "diagnoses.json"


def diagnosis_vocab_path() -> Path:
    return Path(os.environ.get("GEOTOOL_DIAGNOSIS_VOCAB", DEFAULT_DIAGNOSIS_VOCAB_PATH))


def load_diagnosis_categories() -> list[str]:
    with open(diagnosis_vocab_path(), encoding="utf-8") as f:
        return json.load(f)


def normalize_diagnosis(raw: str, categories: list[str] | None = None) -> tuple[str, bool]:
    """Case/whitespace-insensitive match of `raw` against known categories.

    Returns (matched_category, True) on a hit, or ("other", False) when
    nothing matches -- the caller should keep the raw value in a detail
    field so nothing is silently lost.
    """
    categories = categories if categories is not None else load_diagnosis_categories()
    needle = raw.strip().lower()
    for category in categories:
        if category.strip().lower() == needle:
            return category, True
    return "other", False
