"""grammar.py — Grammar profile (M9) and ERRANT error analysis (Step 7).

Reimplements text-only logic from AZTARNA's grammar_profile.py and
the ERRANT-only portion of error.py.

Columns produced:
    COLUMNS_PHASE5_GRAMMAR:
        az_tense_present, az_tense_past, az_tense_future, az_tense_conditional,
        az_copulative_ratio, az_n_passives, az_n_phrasal_verbs,
        az_adverb_variety, az_adjective_variety, az_n_complex_prepositions
    COLUMNS_PHASE5_ERRORS:
        az_errant_error_count, az_errant_error_types

Project: AZTARNA_TEXT
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================
# Constants (from grammar_profile.py)
# ============================================================

_COPULATIVE_LEMMAS = frozenset({"be", "have", "get", "seem"})

_COMPLEX_PREP_PATTERNS = [
    ("in spite of", re.compile(r"\bin spite of\b", re.IGNORECASE)),
    ("according to", re.compile(r"\baccording to\b", re.IGNORECASE)),
    ("due to", re.compile(r"\bdue to\b", re.IGNORECASE)),
    ("because of", re.compile(r"\bbecause of\b", re.IGNORECASE)),
    ("in order to", re.compile(r"\bin order to\b", re.IGNORECASE)),
    ("as a result of", re.compile(r"\bas a result of\b", re.IGNORECASE)),
    ("in addition to", re.compile(r"\bin addition to\b", re.IGNORECASE)),
]


# ============================================================
# Grammar profile (M9)
# ============================================================

def _analyze_grammar(doc: Any, text: str) -> dict[str, object]:
    """Extract grammar profile from spaCy doc."""

    # --- Tense distribution ---
    tense_counts: Counter = Counter()
    n_copulative = 0
    n_lexical = 0

    for token in doc:
        # Finite verb tense
        if token.pos_ in {"VERB", "AUX"}:
            morph_tense = token.morph.get("Tense")
            if morph_tense:
                tense = morph_tense[0]
                if tense == "Pres":
                    tense_counts["present"] += 1
                elif tense == "Past":
                    tense_counts["past"] += 1

        # Future: will/shall + verb
        if (token.lemma_.lower() in ("will", "shall")
                and token.dep_ == "aux"
                and token.head.pos_ == "VERB"):
            tense_counts["future"] += 1

        # Conditional: would + verb
        if (token.lemma_.lower() == "would"
                and token.dep_ == "aux"
                and token.head.pos_ == "VERB"):
            tense_counts["conditional"] += 1

        # Copulative vs lexical
        if token.pos_ == "VERB":
            if token.lemma_.lower() in _COPULATIVE_LEMMAS:
                n_copulative += 1
            else:
                n_lexical += 1

    total_verbs = n_copulative + n_lexical
    copulative_ratio = (
        round(n_copulative / total_verbs, 3) if total_verbs > 0 else 0.0
    )

    # --- Passives ---
    n_passives = 0
    seen_heads: set[int] = set()
    for token in doc:
        if token.dep_ == "nsubjpass" and token.head.i not in seen_heads:
            seen_heads.add(token.head.i)
            n_passives += 1
        elif (token.dep_ in ("aux", "auxpass")
              and token.lemma_.lower() in ("be", "get")
              and token.head.tag_ == "VBN"
              and token.head.i not in seen_heads):
            seen_heads.add(token.head.i)
            n_passives += 1

    # --- Phrasal verbs ---
    n_phrasal = 0
    for token in doc:
        if token.pos_ == "VERB":
            if any(child.dep_ == "prt" for child in token.children):
                n_phrasal += 1

    # --- Adverb variety ---
    adv_forms = [
        t.lemma_.lower() for t in doc
        if t.pos_ == "ADV" and not t.is_stop
    ]
    adv_variety = (
        round(len(set(adv_forms)) / len(adv_forms), 3)
        if adv_forms else 0.0
    )

    # --- Adjective variety ---
    adj_forms = [t.lemma_.lower() for t in doc if t.pos_ == "ADJ"]
    adj_variety = (
        round(len(set(adj_forms)) / len(adj_forms), 3)
        if adj_forms else 0.0
    )

    # --- Complex prepositions ---
    n_complex_prep = 0
    for _, pattern in _COMPLEX_PREP_PATTERNS:
        n_complex_prep += len(pattern.findall(text))

    return {
        "az_tense_present": tense_counts.get("present", 0),
        "az_tense_past": tense_counts.get("past", 0),
        "az_tense_future": tense_counts.get("future", 0),
        "az_tense_conditional": tense_counts.get("conditional", 0),
        "az_copulative_ratio": copulative_ratio,
        "az_n_passives": n_passives,
        "az_n_phrasal_verbs": n_phrasal,
        "az_adverb_variety": adv_variety,
        "az_adjective_variety": adj_variety,
        "az_n_complex_prepositions": n_complex_prep,
    }


# ============================================================
# ERRANT error analysis (Step 7, text-only portion)
# ============================================================

_errant_available: bool | None = None


def _check_errant() -> bool:
    """Check if language_tool_python is importable."""
    global _errant_available
    if _errant_available is None:
        try:
            import language_tool_python  # noqa: F401
            _errant_available = True
        except (ImportError, ModuleNotFoundError):
            _errant_available = False
            logger.warning(
                "language_tool_python not available — az_errant_* will be NaN. "
                "Install with: pip install language-tool-python",
            )
    return _errant_available


_lang_tool: Any = None


def _get_lang_tool() -> Any:
    """Get or create LanguageTool instance (singleton)."""
    global _lang_tool
    if _lang_tool is None:
        import language_tool_python
        _lang_tool = language_tool_python.LanguageTool("en-US")
        logger.info("Loaded LanguageTool (en-US)")
    return _lang_tool


def _analyze_errors(text: str) -> dict[str, object]:
    """Count grammatical errors using LanguageTool.

    Uses language_tool_python to detect grammar, style, and spelling errors.
    Returns total error count and pipe-separated error category list.

    Note: Column names retain the 'errant' prefix for backward compatibility
    with existing analysis pipelines.
    """
    nan = float("nan")
    if not _check_errant():
        return {"az_errant_error_count": nan, "az_errant_error_types": ""}

    try:
        tool = _get_lang_tool()
        matches = tool.check(text)

        error_count = len(matches)
        type_counts: Counter = Counter()
        for match in matches:
            type_counts[match.category] += 1

        # Top 5 error categories, pipe-separated
        top_types = "|".join(
            f"{t}:{c}" for t, c in type_counts.most_common(5)
        )

        return {
            "az_errant_error_count": error_count,
            "az_errant_error_types": top_types,
        }
    except Exception as e:
        logger.debug("LanguageTool analysis failed: %s", e)
        return {"az_errant_error_count": nan, "az_errant_error_types": ""}


# ============================================================
# Public API
# ============================================================

def analyze_essay_grammar(
    text: str,
    nlp: Any,
) -> dict[str, object]:
    """Compute Phase 5 grammar + error metrics for a single essay.

    Args:
        text: Full essay text.
        nlp: spaCy Language model.

    Returns:
        Dict with az_* keys for grammar and error columns.
    """
    doc = nlp(text)
    result = _analyze_grammar(doc, text)
    errors = _analyze_errors(text)
    result.update(errors)
    return result
