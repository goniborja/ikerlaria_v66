"""semantics.py — Semantic redundancy analysis.

NEW module (not from AZTARNA). Computes intra-essay semantic redundancy
using sentence-transformers embeddings.

The idea: an essay that says the same thing 5 times with different clothing
(like essay #278) will have high redundancy. An essay that develops diverse
ideas will have low redundancy.

Columns produced (COLUMNS_PHASE4_REDUNDANCY):
    az_semantic_redundancy_mean, az_semantic_redundancy_max_nonadj,
    az_semantic_redundancy_std

Project: AZTARNA_TEXT
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Lazy model loader
# ============================================================

_model: Any = None


def _get_model() -> Any:
    """Load sentence-transformers model (singleton)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformers: all-MiniLM-L6-v2")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(
                "sentence-transformers not available: %s. "
                "Redundancy will be NaN. "
                "Install with: pip install sentence-transformers", e,
            )
            _model = False  # Sentinel for "tried and failed"
    return _model


# ============================================================
# Sentence splitting
# ============================================================

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (regex heuristic)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]


# ============================================================
# Core computation
# ============================================================

def _compute_redundancy(
    sentences: list[str],
    model: Any,
) -> tuple[float, float, float]:
    """Compute semantic redundancy metrics.

    Args:
        sentences: List of sentences (>= 3 words each).
        model: SentenceTransformer model.

    Returns:
        (mean_cosine, max_nonadj_cosine, std_cosine) or NaN tuple.
    """
    nan = float("nan")
    if len(sentences) < 2:
        return nan, nan, nan

    # Encode all sentences
    embeddings = model.encode(sentences, convert_to_numpy=True)

    # Compute pairwise cosine similarities
    # Normalize embeddings for cosine = dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normed = embeddings / norms

    # Full cosine similarity matrix
    sim_matrix = normed @ normed.T

    n = len(sentences)
    all_cosines = []
    nonadj_cosines = []

    for i in range(n):
        for j in range(i + 1, n):
            cos = float(sim_matrix[i, j])
            all_cosines.append(cos)
            # Non-adjacent = distance > 1
            if abs(i - j) > 1:
                nonadj_cosines.append(cos)

    mean_cos = float(np.mean(all_cosines)) if all_cosines else nan
    std_cos = float(np.std(all_cosines)) if all_cosines else nan
    max_nonadj = float(max(nonadj_cosines)) if nonadj_cosines else nan

    return mean_cos, max_nonadj, std_cos


# ============================================================
# Public API
# ============================================================

def analyze_essay_redundancy(text: str) -> dict[str, object]:
    """Compute semantic redundancy metrics for a single essay.

    Args:
        text: Full essay text.

    Returns:
        Dict with az_semantic_redundancy_* keys.
    """
    nan = float("nan")
    model = _get_model()

    if model is False or model is None:
        return {
            "az_semantic_redundancy_mean": nan,
            "az_semantic_redundancy_max_nonadj": nan,
            "az_semantic_redundancy_std": nan,
        }

    if not text or not text.strip():
        return {
            "az_semantic_redundancy_mean": nan,
            "az_semantic_redundancy_max_nonadj": nan,
            "az_semantic_redundancy_std": nan,
        }

    sentences = _split_sentences(text)

    try:
        mean_cos, max_nonadj, std_cos = _compute_redundancy(sentences, model)
        return {
            "az_semantic_redundancy_mean": (
                round(mean_cos, 4) if not np.isnan(mean_cos) else nan
            ),
            "az_semantic_redundancy_max_nonadj": (
                round(max_nonadj, 4) if not np.isnan(max_nonadj) else nan
            ),
            "az_semantic_redundancy_std": (
                round(std_cos, 4) if not np.isnan(std_cos) else nan
            ),
        }
    except Exception as e:
        logger.warning("Redundancy computation failed: %s", e)
        return {
            "az_semantic_redundancy_mean": nan,
            "az_semantic_redundancy_max_nonadj": nan,
            "az_semantic_redundancy_std": nan,
        }
