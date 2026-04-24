"""utils.py — Checkpoint I/O, progress reporting, and shared helpers.

Project: AZTARNA_TEXT
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# Column registry — single source of truth
# ============================================================

# All az_* columns that aztarna_text will produce, grouped by phase.
# Columns are added empty in Phase 1; each module fills its own subset.

COLUMNS_PHASE2_NLP = [
    "az_mean_t_unit_length",
    "az_mean_dep_distance",
    "az_lexical_sophistication",
    "az_mtld",
    "az_spelling_error_count",
    "az_spelling_error_ratio",
    "az_semantic_coherence_mean",
    "az_connector_additive",
    "az_connector_causal",
    "az_connector_contrastive",
    "az_connector_temporal",
    "az_nivel_complejidad_max",
    "az_nivel_complejidad_mean",
]

COLUMNS_PHASE2_DISCOURSE = [
    "az_rhetorical_depth",
    "az_multinuclear_ratio",
    "az_relation_diversity",
    "az_connector_entropy",
    "az_connector_precision",
    "az_explicit_implicit_ratio",
    "az_argument_completeness",
    "az_counterargument_present",
    "az_argumentative_depth",
]

COLUMNS_PHASE3_COREF = [
    "az_coref_chains",
    "az_coref_longest",
    "az_coref_cross_paragraph",
    "az_coref_density",
]

COLUMNS_PHASE3_EDU = [
    "az_edu_total",
    "az_edu_mean_length",
    "az_edu_complexity",
    "az_edu_subordination_ratio",
    "az_edu_coordination_ratio",
    "az_edu_juxtaposition_ratio",
]

COLUMNS_PHASE3_THEMATIC = [
    "az_longest_constant_run",
]

COLUMNS_PHASE3_SEMANTIC_ANCHOR = [
    "az_anchor_distance",
    "az_anchor_variance",
    "az_anchor_max_deviation",
]

COLUMNS_PHASE3_ARG_SUB = [
    "az_arg_sub_ratio",
    "az_arg_sub_argumentative",
    "az_arg_sub_decorative",
]

COLUMNS_PHASE4_PERPLEXITY = [
    "az_surprisal_mean",
    "az_surprisal_std",
]

COLUMNS_PHASE4_REDUNDANCY = [
    "az_semantic_redundancy_mean",
    "az_semantic_redundancy_max_nonadj",
    "az_semantic_redundancy_std",
]

COLUMNS_PHASE5_GRAMMAR = [
    "az_tense_present",
    "az_tense_past",
    "az_tense_future",
    "az_tense_conditional",
    "az_copulative_ratio",
    "az_n_passives",
    "az_n_phrasal_verbs",
    "az_adverb_variety",
    "az_adjective_variety",
    "az_n_complex_prepositions",
]

COLUMNS_PHASE5_LEXFREQ = [
    "az_lexfreq_band_1k",
    "az_lexfreq_band_2k",
    "az_lexfreq_band_3k",
    "az_lexfreq_band_beyond",
    "az_pct_beyond_2k",
    "az_sophistication_ratio",
]

COLUMNS_PHASE5_KROPOTKIN = [
    "az_sentiment_compound",
    "az_sentiment_pos_ratio",
    "az_sentiment_neg_ratio",
    "az_sentiment_trend",
    "az_empathy_density",
    "az_empathy_has_perspective",
    "az_agency_ratio",
    "az_agency_profile",
]

COLUMNS_PHASE5_ERRORS = [
    "az_errant_error_count",
    "az_errant_error_types",
]


def all_az_columns() -> list[str]:
    """Return the full ordered list of az_* output columns."""
    return (
        COLUMNS_PHASE2_NLP
        + COLUMNS_PHASE2_DISCOURSE
        + COLUMNS_PHASE3_COREF
        + COLUMNS_PHASE3_EDU
        + COLUMNS_PHASE3_THEMATIC
        + COLUMNS_PHASE3_SEMANTIC_ANCHOR
        + COLUMNS_PHASE3_ARG_SUB
        + COLUMNS_PHASE4_PERPLEXITY
        + COLUMNS_PHASE4_REDUNDANCY
        + COLUMNS_PHASE5_GRAMMAR
        + COLUMNS_PHASE5_LEXFREQ
        + COLUMNS_PHASE5_KROPOTKIN
        + COLUMNS_PHASE5_ERRORS
    )


# ============================================================
# Checkpoint helpers
# ============================================================

def find_latest_checkpoint(
    checkpoint_dir: str | Path,
    input_ids: list[str] | None = None,
) -> tuple[pd.DataFrame | None, int]:
    """Find the most recent checkpoint CSV and return (df, last_row_index).

    Args:
        checkpoint_dir: Directory containing checkpoint_NNNN.csv files.
        input_ids: Optional list of essay_ids from the current input CSV.
                   If provided, checkpoint is validated against these IDs.
                   A mismatch means stale checkpoint → returns (None, -1).

    Returns:
        (DataFrame, last_processed_index) or (None, -1) if no checkpoint.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None, -1

    ckpt_files = sorted(ckpt_dir.glob("checkpoint_*.csv"))
    if not ckpt_files:
        return None, -1

    latest = ckpt_files[-1]
    # Extract index from filename: checkpoint_0400.csv → 400
    stem = latest.stem  # "checkpoint_0400"
    try:
        last_idx = int(stem.split("_")[1])
    except (IndexError, ValueError):
        return None, -1

    df = pd.read_csv(latest)

    # Validate checkpoint essay_ids match input — prevents stale checkpoint reuse
    if input_ids is not None and "essay_id" in df.columns:
        ckpt_ids = df["essay_id"].tolist()
        expected_ids = input_ids[: len(ckpt_ids)]
        if ckpt_ids != expected_ids:
            logger.warning(
                "Checkpoint essay_ids don't match input. "
                "Stale checkpoint from a different run — ignoring. "
                "Expected first ID: %s, got: %s",
                expected_ids[0] if expected_ids else "N/A",
                ckpt_ids[0] if ckpt_ids else "N/A",
            )
            return None, -1

    return df, last_idx


def save_checkpoint(
    df: pd.DataFrame,
    checkpoint_dir: str | Path,
    last_idx: int,
) -> Path:
    """Save a checkpoint CSV.

    Args:
        df: DataFrame with all rows processed so far.
        checkpoint_dir: Directory for checkpoint files.
        last_idx: Index of last processed row (0-based).

    Returns:
        Path to the saved checkpoint file.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    filename = f"checkpoint_{last_idx:05d}.csv"
    path = ckpt_dir / filename
    df.to_csv(path, index=False)
    return path


# ============================================================
# Progress reporting
# ============================================================

class ProgressTracker:
    """Track processing progress with ETA and memory usage."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.start_time = time.time()
        self.processed = 0

    def update(self, n: int = 1) -> None:
        """Mark n more items as processed."""
        self.processed += n

    def report(self) -> str:
        """Return a progress string with ETA and memory."""
        elapsed = time.time() - self.start_time
        if self.processed == 0:
            eta_str = "?"
        else:
            rate = elapsed / self.processed
            remaining = (self.total - self.processed) * rate
            eta_str = _format_seconds(remaining)

        mem_mb = _get_memory_mb()
        pct = 100.0 * self.processed / self.total if self.total > 0 else 0.0

        return (
            f"[{self.processed}/{self.total}] "
            f"{pct:.1f}% | "
            f"elapsed {_format_seconds(elapsed)} | "
            f"ETA {eta_str} | "
            f"mem {mem_mb:.0f} MB"
        )


def _format_seconds(s: float) -> str:
    """Format seconds as Xm Ys."""
    if s < 60:
        return f"{s:.0f}s"
    m = int(s) // 60
    sec = int(s) % 60
    return f"{m}m {sec}s"


def _get_memory_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0
