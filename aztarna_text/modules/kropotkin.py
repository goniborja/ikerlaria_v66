"""kropotkin.py — K1/K2/K3: Sentiment, Empathy, Agency.

Reimplements text-only logic from AZTARNA's kropotkin.py.

K1: Emotional polarity (VADER compound, pos/neg ratio, trend)
K2: Textual empathy (concessive + cognitive verbs → density)
K3: Agency (volitional verbs vs passives → ratio + profile)

Columns produced (COLUMNS_PHASE5_KROPOTKIN):
    az_sentiment_compound, az_sentiment_pos_ratio, az_sentiment_neg_ratio,
    az_sentiment_trend, az_empathy_density, az_empathy_has_perspective,
    az_agency_ratio, az_agency_profile

Project: AZTARNA_TEXT
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Shared sentence splitter
# ============================================================

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (regex heuristic)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


# ============================================================
# K1 — Sentiment (VADER)
# ============================================================

_POSITIVE_WORDS = frozenset({
    "good", "great", "best", "better", "love", "like", "happy",
    "beautiful", "wonderful", "excellent", "amazing", "nice",
    "important", "benefit", "positive", "enjoy", "help", "hope",
    "improve", "success", "advantage", "useful", "interesting",
})

_NEGATIVE_WORDS = frozenset({
    "bad", "worst", "worse", "hate", "ugly", "terrible", "horrible",
    "dangerous", "problem", "negative", "difficult", "hard", "wrong",
    "fail", "destroy", "damage", "harm", "risk", "threat", "suffer",
    "unfortunately", "sadly", "disadvantage", "poor",
})

_vader_available: bool | None = None


def _check_vader() -> bool:
    """Check if VADER is importable."""
    global _vader_available
    if _vader_available is None:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: F401
            _vader_available = True
        except (ImportError, ModuleNotFoundError, LookupError):
            _vader_available = False
            logger.warning(
                "VADER not available — using word-list fallback for sentiment.",
            )
    return _vader_available


def _sentiment_vader(text: str, sentences: list[str]) -> dict[str, object]:
    """Compute sentiment using VADER."""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    scores = sid.polarity_scores(text)
    per_sent = [sid.polarity_scores(s)["compound"] for s in sentences]
    n = len(per_sent)
    pos_count = sum(1 for c in per_sent if c > 0.05)
    neg_count = sum(1 for c in per_sent if c < -0.05)

    # Trend: compare first third vs last third
    trend = _compute_trend_vader(text, sid)

    return {
        "compound": round(scores["compound"], 4),
        "pos_ratio": round(pos_count / n, 3) if n else 0.0,
        "neg_ratio": round(neg_count / n, 3) if n else 0.0,
        "trend": trend,
    }


def _sentiment_fallback(text: str, sentences: list[str]) -> dict[str, object]:
    """Simple word-list polarity fallback."""
    words = text.lower().split()
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
    total = pos + neg
    compound = (pos - neg) / total if total > 0 else 0.0

    n = len(sentences)
    pos_count = 0
    neg_count = 0
    for s in sentences:
        s_words = s.lower().split()
        sp = sum(1 for w in s_words if w in _POSITIVE_WORDS)
        sn = sum(1 for w in s_words if w in _NEGATIVE_WORDS)
        if sp > sn:
            pos_count += 1
        elif sn > sp:
            neg_count += 1

    trend = _compute_trend_fallback(text)

    return {
        "compound": round(compound, 4),
        "pos_ratio": round(pos_count / n, 3) if n else 0.0,
        "neg_ratio": round(neg_count / n, 3) if n else 0.0,
        "trend": trend,
    }


def _compute_trend_vader(text: str, sid: Any) -> str:
    """Compute polarity trend across text thirds using VADER."""
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return "stable"
    third = max(1, len(sentences) // 3)
    chunks = [
        " ".join(sentences[:third]),
        " ".join(sentences[third:2 * third]),
        " ".join(sentences[2 * third:]),
    ]
    scores = [sid.polarity_scores(c)["compound"] for c in chunks]
    if scores[-1] < scores[0] - 0.1:
        return "declining"
    elif scores[-1] > scores[0] + 0.1:
        return "rising"
    return "stable"


def _compute_trend_fallback(text: str) -> str:
    """Compute polarity trend using word-list fallback."""
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return "stable"
    third = max(1, len(sentences) // 3)
    chunks = [
        " ".join(sentences[:third]),
        " ".join(sentences[third:2 * third]),
        " ".join(sentences[2 * third:]),
    ]
    scores = []
    for c in chunks:
        words = c.lower().split()
        pos = sum(1 for w in words if w in _POSITIVE_WORDS)
        neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
        total = pos + neg
        scores.append((pos - neg) / total if total > 0 else 0.0)

    if scores[-1] < scores[0] - 0.1:
        return "declining"
    elif scores[-1] > scores[0] + 0.1:
        return "rising"
    return "stable"


def _analyze_sentiment(text: str) -> dict[str, object]:
    """Compute K1 sentiment metrics."""
    sentences = _split_sentences(text)
    if not sentences:
        return {
            "compound": 0.0,
            "pos_ratio": 0.0,
            "neg_ratio": 0.0,
            "trend": "stable",
        }

    if _check_vader():
        try:
            return _sentiment_vader(text, sentences)
        except Exception as e:
            logger.debug("VADER failed (%s), using fallback", e)

    return _sentiment_fallback(text, sentences)


# ============================================================
# K2 — Empathy (concessives + cognitive verbs)
# ============================================================

COGNITIVE_VERBS = frozenset({
    "think", "believe", "feel", "consider", "suppose", "guess",
    "imagine", "prefer", "hope", "wish", "doubt", "agree", "disagree",
})

_CONCESSIVE_FORMS = frozenset({
    "although", "though", "however", "nevertheless", "nonetheless",
    "despite", "in spite of", "even though", "while", "whereas",
    "on the other hand", "yet", "still", "admittedly", "granted",
})


def _analyze_empathy(text: str, doc: Any) -> dict[str, object]:
    """Compute K2 empathy metrics."""
    # Count concessive connectors from text
    text_lower = text.lower()
    n_concessive = 0
    for form in _CONCESSIVE_FORMS:
        n_concessive += text_lower.count(form)

    # Count cognitive verbs via spaCy
    n_cognitive = 0
    for token in doc:
        if token.lemma_.lower() in COGNITIVE_VERBS and token.pos_ == "VERB":
            n_cognitive += 1

    sentences = _split_sentences(text)
    n_sentences = max(len(sentences), 1)
    empathy_density = (n_concessive + n_cognitive) / n_sentences
    has_perspective = n_concessive >= 1 and n_cognitive >= 1

    return {
        "empathy_density": round(empathy_density, 3),
        "has_perspective": has_perspective,
    }


# ============================================================
# K3 — Agency (volitional vs passive)
# ============================================================

VOLITIONAL_VERBS = frozenset({
    "decide", "choose", "want", "intend", "plan", "aim",
    "determine", "resolve", "prefer", "hope", "wish",
    "demand", "insist", "refuse", "commit",
})


def _analyze_agency(doc: Any, n_passives: int) -> dict[str, object]:
    """Compute K3 agency metrics.

    Args:
        doc: spaCy Doc.
        n_passives: Number of passives from grammar module.

    Returns:
        Dict with agency_ratio and agency_profile.
    """
    n_volitional = 0
    for token in doc:
        if token.lemma_.lower() in VOLITIONAL_VERBS and token.pos_ == "VERB":
            n_volitional += 1

    total = n_volitional + n_passives
    if total == 0:
        return {
            "agency_ratio": float("nan"),
            "agency_profile": "",
        }

    agency_ratio = n_volitional / total
    if agency_ratio > 0.6:
        profile = "agente"
    elif agency_ratio < 0.3:
        profile = "paciente"
    else:
        profile = "equilibrado"

    return {
        "agency_ratio": round(agency_ratio, 3),
        "agency_profile": profile,
    }


# ============================================================
# Public API
# ============================================================

def analyze_essay_kropotkin(
    text: str,
    nlp: Any,
    n_passives: int = 0,
) -> dict[str, object]:
    """Compute K1/K2/K3 metrics for a single essay.

    Args:
        text: Full essay text.
        nlp: spaCy Language model.
        n_passives: Number of passive constructions (from grammar module).

    Returns:
        Dict with az_sentiment_*, az_empathy_*, az_agency_* keys.
    """
    nan = float("nan")
    defaults = {
        "az_sentiment_compound": nan,
        "az_sentiment_pos_ratio": nan,
        "az_sentiment_neg_ratio": nan,
        "az_sentiment_trend": "",
        "az_empathy_density": nan,
        "az_empathy_has_perspective": False,
        "az_agency_ratio": nan,
        "az_agency_profile": "",
    }

    if not text or not text.strip():
        return defaults

    # K1: Sentiment
    k1 = _analyze_sentiment(text)

    # Parse doc once for K2 + K3
    doc = nlp(text)

    # K2: Empathy
    k2 = _analyze_empathy(text, doc)

    # K3: Agency
    k3 = _analyze_agency(doc, n_passives)

    return {
        "az_sentiment_compound": k1["compound"],
        "az_sentiment_pos_ratio": k1["pos_ratio"],
        "az_sentiment_neg_ratio": k1["neg_ratio"],
        "az_sentiment_trend": k1["trend"],
        "az_empathy_density": k2["empathy_density"],
        "az_empathy_has_perspective": k2["has_perspective"],
        "az_agency_ratio": k3["agency_ratio"],
        "az_agency_profile": k3["agency_profile"],
    }
