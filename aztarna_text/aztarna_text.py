"""
AZTARNA_TEXT — Rhetorical-pragmatic text analysis for any corpus.

Reads a CSV with a text column, computes ~65 linguistic/discourse/pragmatic
metrics per row, and writes an enriched CSV preserving all original columns.

Independent of AZTARNA (keystroke pipeline) and TROJAN (AI detection pipeline).
Extracts text-only logic from AZTARNA modules, reimplemented cleanly.

Usage:
    python aztarna_text.py --input corpus.csv --text_column text --output results.csv
    python aztarna_text.py --input corpus.csv --text_column text --output results.csv \
        --prompt_text "Write an argumentative essay..." --batch_size 200

Project: BOZGORAILUA
Author: Borja Goñi / Claude
Date: 2026-04-08
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from modules.utils import (
    ProgressTracker,
    all_az_columns,
    find_latest_checkpoint,
    save_checkpoint,
)
from modules.nlp_base import NLPResources, analyze_essay_nlp
from modules.discourse import analyze_essay_discourse
from modules.pragmatic import analyze_essay_pragmatic
from modules.perplexity import analyze_essay_perplexity
from modules.semantics import analyze_essay_redundancy
from modules.grammar import analyze_essay_grammar
from modules.lexical import analyze_essay_lexical
from modules.kropotkin import analyze_essay_kropotkin

logger = logging.getLogger("aztarna_text")

# Global NLP resources — initialized once in run()
_resources: NLPResources | None = None


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="AZTARNA_TEXT — Rhetorical-pragmatic text analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", required=True,
        help="Path to input CSV with at least one text column.",
    )
    p.add_argument(
        "--text_column", default="text",
        help="Name of the column containing essay text (default: 'text').",
    )
    p.add_argument(
        "--output", required=True,
        help="Path for the output CSV (original columns + az_* columns).",
    )
    p.add_argument(
        "--prompt_text", default=None,
        help="Prompt text used to generate essays. Activates semantic_anchor "
             "and argument_completeness metrics.",
    )
    p.add_argument(
        "--checkpoint_dir", default="./checkpoints",
        help="Directory for intermediate checkpoint CSVs (default: ./checkpoints).",
    )
    p.add_argument(
        "--batch_size", type=int, default=200,
        help="Save checkpoint every N essays (default: 200).",
    )
    p.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    p.add_argument(
        "--skip_perplexity", action="store_true",
        help="Skip GPT-2 surprisal (slow). Produces 69/71 columns fast.",
    )
    p.add_argument(
        "--only_perplexity", action="store_true",
        help="Only compute GPT-2 surprisal. Merges into existing output CSV.",
    )
    return p


# ============================================================
# Text → paragraph boundaries
# ============================================================

def text_to_paragraphs(text: str) -> list[dict]:
    """Split text into paragraph dicts with char offsets.

    Args:
        text: Full essay text.

    Returns:
        List of dicts with keys: index, char_start, char_end, n_chars, text.
    """
    paragraphs = []
    raw_parts = text.split("\n\n")
    char_pos = 0

    for i, part in enumerate(raw_parts):
        stripped = part.strip()
        if not stripped:
            char_pos += len(part) + 2  # +2 for the \n\n separator
            continue

        # Find actual start in original text
        start = text.find(stripped, char_pos)
        if start == -1:
            start = char_pos
        end = start + len(stripped)

        paragraphs.append({
            "index": len(paragraphs),
            "char_start": start,
            "char_end": end,
            "n_chars": end - start,
            "text": stripped,
        })
        char_pos = end

    # Fallback: if no \n\n splits found, treat whole text as one paragraph
    if not paragraphs and text.strip():
        stripped = text.strip()
        paragraphs.append({
            "index": 0,
            "char_start": 0,
            "char_end": len(stripped),
            "n_chars": len(stripped),
            "text": stripped,
        })

    return paragraphs


# ============================================================
# Process a single essay (stub — modules fill this in later)
# ============================================================

def process_essay(
    text: str,
    prompt_text: str | None = None,
    skip_perplexity: bool = False,
    only_perplexity: bool = False,
) -> dict[str, object]:
    """Run all analysis modules on a single essay.

    Args:
        text: Full essay text.
        prompt_text: Optional prompt text for anchor metrics.
        skip_perplexity: Skip GPT-2 surprisal (fast pass).
        only_perplexity: Only compute GPT-2 surprisal.

    Returns:
        Dict with az_* column names as keys.
    """
    result = {col: np.nan for col in all_az_columns()}

    # Validate input
    if not text or not isinstance(text, str) or not text.strip():
        return result

    # Build paragraph boundaries
    paragraphs = text_to_paragraphs(text)
    if not paragraphs:
        return result

    if only_perplexity:
        # Only compute GPT-2 surprisal
        try:
            perplexity_results = analyze_essay_perplexity(text)
            result.update(perplexity_results)
        except Exception as e:
            logger.warning("Perplexity analysis failed: %s", e)
        return result

    # --- Phase 2: NLP base ---
    connectors_enriched: list[dict] = []
    try:
        nlp_results, connectors_enriched = analyze_essay_nlp(
            text, paragraphs, _resources,
        )
        result.update(nlp_results)
    except Exception as e:
        logger.warning("NLP analysis failed: %s", e)

    # --- Phase 2: Discourse (A9, A10, A11) ---
    try:
        discourse_results = analyze_essay_discourse(
            text, paragraphs, _resources.nlp, prompt_text=prompt_text,
        )
        result.update(discourse_results)
    except Exception as e:
        logger.warning("Discourse analysis failed: %s", e)

    # --- Phase 3: Pragmatic modules ---
    try:
        pragmatic_results = analyze_essay_pragmatic(
            text, paragraphs, _resources.nlp,
            prompt_text=prompt_text,
            connectors_enriched=connectors_enriched,
        )
        result.update(pragmatic_results)
    except Exception as e:
        logger.warning("Pragmatic analysis failed: %s", e)

    # --- Phase 4: Perplexity (GPT-2 surprisal) ---
    if not skip_perplexity:
        try:
            perplexity_results = analyze_essay_perplexity(text)
            result.update(perplexity_results)
        except Exception as e:
            logger.warning("Perplexity analysis failed: %s", e)

    # --- Phase 4: Semantic redundancy ---
    try:
        redundancy_results = analyze_essay_redundancy(text)
        result.update(redundancy_results)
    except Exception as e:
        logger.warning("Redundancy analysis failed: %s", e)

    # --- Phase 5: Grammar + ERRANT errors ---
    n_passives = 0
    try:
        grammar_results = analyze_essay_grammar(text, _resources.nlp)
        result.update(grammar_results)
        n_passives = grammar_results.get("az_n_passives", 0)
    except Exception as e:
        logger.warning("Grammar analysis failed: %s", e)

    # --- Phase 5: Lexical frequency profile ---
    try:
        lexical_results = analyze_essay_lexical(text, _resources.nlp)
        result.update(lexical_results)
    except Exception as e:
        logger.warning("Lexical analysis failed: %s", e)

    # --- Phase 5: Kropotkin (sentiment, empathy, agency) ---
    try:
        kropotkin_results = analyze_essay_kropotkin(
            text, _resources.nlp, n_passives=n_passives,
        )
        result.update(kropotkin_results)
    except Exception as e:
        logger.warning("Kropotkin analysis failed: %s", e)

    return result


# ============================================================
# Main pipeline loop
# ============================================================

def _run_only_perplexity(args: argparse.Namespace) -> None:
    """Second pass: compute only GPT-2 surprisal and merge into existing output."""
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    df = pd.read_csv(input_path)
    logger.info("[only_perplexity] Loaded %d rows from %s", len(df), input_path)

    if args.text_column not in df.columns:
        logger.error("Text column '%s' not found.", args.text_column)
        sys.exit(1)

    total = len(df)
    tracker = ProgressTracker(total)
    t0 = time.time()

    surprisal_means = []
    surprisal_stds = []

    for i in range(total):
        text = df.iloc[i][args.text_column]
        if pd.isna(text) or not isinstance(text, str):
            text = ""

        result = analyze_essay_perplexity(text)
        surprisal_means.append(result["az_surprisal_mean"])
        surprisal_stds.append(result["az_surprisal_std"])

        tracker.update()
        if (i + 1) % 100 == 0 or i == total - 1:
            logger.info(tracker.report())

    df["az_surprisal_mean"] = surprisal_means
    df["az_surprisal_std"] = surprisal_stds
    df.to_csv(args.output, index=False)

    elapsed = time.time() - t0
    per_essay = elapsed / total if total > 0 else 0.0
    logger.info("=" * 60)
    logger.info("DONE [only_perplexity]")
    logger.info("  Essays: %d | Time: %.1fs (%.3fs/essay)", total, elapsed, per_essay)
    logger.info("  Output: %s", args.output)
    logger.info("=" * 60)


def run(args: argparse.Namespace) -> None:
    """Main processing loop: read CSV, process essays, write output."""
    global _resources
    if not args.only_perplexity:
        _resources = NLPResources()
    else:
        # only_perplexity doesn't need spaCy — use lightweight stub
        _resources = None

    # --- Handle --only_perplexity mode ---
    if args.only_perplexity:
        return _run_only_perplexity(args)

    # --- Load input CSV ---
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_path)
    logger.info("Columns: %s", list(df.columns))

    # Validate text column
    if args.text_column not in df.columns:
        logger.error(
            "Text column '%s' not found. Available: %s",
            args.text_column, list(df.columns),
        )
        sys.exit(1)

    # --- Check for checkpoint ---
    input_ids = df["essay_id"].tolist() if "essay_id" in df.columns else None
    ckpt_df, last_idx = find_latest_checkpoint(args.checkpoint_dir, input_ids=input_ids)
    if ckpt_df is not None and last_idx >= 0:
        logger.info(
            "Resuming from checkpoint: %d rows already processed", last_idx + 1,
        )
        # Validate checkpoint matches input
        if len(ckpt_df) != last_idx + 1:
            logger.warning(
                "Checkpoint row count (%d) != expected (%d). Starting fresh.",
                len(ckpt_df), last_idx + 1,
            )
            ckpt_df = None
            last_idx = -1
    else:
        last_idx = -1

    # --- Prepare output columns ---
    az_cols = all_az_columns()
    logger.info("Will generate %d az_* columns", len(az_cols))

    # Initialize result storage
    if ckpt_df is not None:
        results = ckpt_df.to_dict("records")
    else:
        results = []

    # --- Process essays ---
    total = len(df)
    start_from = last_idx + 1
    tracker = ProgressTracker(total)
    tracker.processed = start_from

    logger.info(
        "Processing essays %d to %d (batch_size=%d)",
        start_from, total - 1, args.batch_size,
    )

    t0 = time.time()

    for i in range(start_from, total):
        row = df.iloc[i]
        text = row[args.text_column]

        # Handle missing/non-string text
        if pd.isna(text) or not isinstance(text, str):
            text = ""

        # Process
        az_values = process_essay(
            text,
            prompt_text=args.prompt_text,
            skip_perplexity=args.skip_perplexity,
            only_perplexity=args.only_perplexity,
        )

        # Merge original row + az_* values
        merged = row.to_dict()
        merged.update(az_values)
        results.append(merged)

        tracker.update()

        # Progress report every 100 essays
        if (i + 1) % 100 == 0 or i == total - 1:
            logger.info(tracker.report())

        # Checkpoint every batch_size essays
        if (i + 1) % args.batch_size == 0:
            ckpt_path = save_checkpoint(
                pd.DataFrame(results), args.checkpoint_dir, i,
            )
            logger.info("Checkpoint saved: %s", ckpt_path)

    # --- Write final output ---
    out_df = pd.DataFrame(results)

    # Ensure column order: original columns first, then az_* columns
    original_cols = [c for c in df.columns if c in out_df.columns]
    az_present = [c for c in az_cols if c in out_df.columns]
    ordered_cols = original_cols + az_present
    # Add any extra columns not in either list (safety net)
    extra = [c for c in out_df.columns if c not in ordered_cols]
    ordered_cols += extra

    out_df = out_df[ordered_cols]
    out_df.to_csv(args.output, index=False)

    elapsed = time.time() - t0
    n_processed = total - start_from
    per_essay = elapsed / n_processed if n_processed > 0 else 0.0

    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("  Essays processed: %d", n_processed)
    logger.info("  Total columns: %d (%d original + %d az_*)",
                len(ordered_cols), len(original_cols), len(az_present))
    logger.info("  Time: %.1fs (%.3fs/essay)", elapsed, per_essay)
    logger.info("  Output: %s", args.output)
    logger.info("=" * 60)


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    """Parse args, configure logging, run pipeline."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("AZTARNA_TEXT — Rhetorical-pragmatic text analysis")
    logger.info("Input:  %s", args.input)
    logger.info("Output: %s", args.output)
    if args.prompt_text:
        logger.info("Prompt: %s...", args.prompt_text[:80])

    run(args)


if __name__ == "__main__":
    main()
