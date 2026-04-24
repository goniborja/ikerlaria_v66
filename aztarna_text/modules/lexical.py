"""lexical.py — M10: Lexical Frequency Profile.

Reimplements text-only logic from AZTARNA's lexical_frequency.py.
Classifies content tokens by frequency band using wordfreq Zipf scores.

Band thresholds (Zipf scale, higher = more frequent):
  band_1k:     zipf >= 5.5  (very frequent: good, think, say)
  band_2k:     4.5 <= zipf < 5.5
  band_3k:     3.5 <= zipf < 4.5
  band_beyond: zipf < 3.5   (sophisticated / rare words)
  (band_unknown omitted — not useful as separate metric)

Columns produced (COLUMNS_PHASE5_LEXFREQ):
    az_lexfreq_band_1k, az_lexfreq_band_2k, az_lexfreq_band_3k,
    az_lexfreq_band_beyond, az_pct_beyond_2k, az_sophistication_ratio

Project: AZTARNA_TEXT
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# POS tags considered "content words"
_CONTENT_POS = frozenset({"NOUN", "VERB", "ADJ", "ADV"})

# Zipf thresholds (calibrated against BNC/COCA via wordfreq)
_ZIPF_BAND_1K = 5.5
_ZIPF_BAND_2K = 4.5
_ZIPF_BAND_3K = 3.5

# Lazy import flag
_wordfreq_available: bool | None = None


def _check_wordfreq() -> bool:
    """Check if wordfreq is importable."""
    global _wordfreq_available
    if _wordfreq_available is None:
        try:
            import wordfreq  # noqa: F401
            _wordfreq_available = True
        except (ImportError, ModuleNotFoundError):
            _wordfreq_available = False
            logger.warning(
                "wordfreq not available — az_lexfreq_* will be NaN. "
                "Install with: pip install wordfreq",
            )
    return _wordfreq_available


def analyze_essay_lexical(text: str, nlp: Any) -> dict[str, object]:
    """Compute lexical frequency profile for a single essay.

    Args:
        text: Full essay text.
        nlp: spaCy Language model.

    Returns:
        Dict with az_lexfreq_* keys.
    """
    nan = float("nan")
    defaults = {
        "az_lexfreq_band_1k": nan,
        "az_lexfreq_band_2k": nan,
        "az_lexfreq_band_3k": nan,
        "az_lexfreq_band_beyond": nan,
        "az_pct_beyond_2k": nan,
        "az_sophistication_ratio": nan,
    }

    if not _check_wordfreq():
        return defaults

    if not text or not text.strip():
        return defaults

    from wordfreq import zipf_frequency

    doc = nlp(text)

    band_1k = 0
    band_2k = 0
    band_3k = 0
    band_beyond = 0

    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if token.pos_ not in _CONTENT_POS:
            continue

        lemma = token.lemma_.lower()
        zipf = zipf_frequency(lemma, "en")

        if zipf == 0:
            # Unknown word — skip (not useful as separate band)
            continue

        if zipf >= _ZIPF_BAND_1K:
            band_1k += 1
        elif zipf >= _ZIPF_BAND_2K:
            band_2k += 1
        elif zipf >= _ZIPF_BAND_3K:
            band_3k += 1
        else:
            band_beyond += 1

    n_content = band_1k + band_2k + band_3k + band_beyond
    if n_content == 0:
        return defaults

    pct_beyond_2k = (band_3k + band_beyond) / n_content
    sophistication_ratio = band_beyond / n_content

    return {
        "az_lexfreq_band_1k": band_1k,
        "az_lexfreq_band_2k": band_2k,
        "az_lexfreq_band_3k": band_3k,
        "az_lexfreq_band_beyond": band_beyond,
        "az_pct_beyond_2k": round(pct_beyond_2k, 4),
        "az_sophistication_ratio": round(sophistication_ratio, 4),
    }
