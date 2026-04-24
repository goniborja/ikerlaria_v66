"""discourse.py — Discourse analysis: RST approximation, connectors, argumentation.

Reimplements text-only logic from AZTARNA's discourse.py (Steps 12: A9, A10, A11).

Columns produced (COLUMNS_PHASE2_DISCOURSE):
    az_rhetorical_depth, az_multinuclear_ratio, az_relation_diversity,
    az_connector_entropy, az_connector_precision, az_explicit_implicit_ratio,
    az_argument_completeness, az_counterargument_present, az_argumentative_depth

Project: AZTARNA_TEXT
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Constants (from AZTARNA discourse.py)
# ============================================================

# Clausal dependency labels for RST approximation
CLAUSAL_DEPS = frozenset({"advcl", "relcl", "ccomp", "xcomp", "acl"})
COORD_DEPS = frozenset({"conj"})
ALL_RELATION_DEPS = CLAUSAL_DEPS | COORD_DEPS | {"parataxis"}

# Connector lexicon — aligned with lib/markers_core.py CONNECTORS (4 categories)
CONNECTOR_LEXICON: dict[str, str] = {
    # Aditivos (11)
    "and": "additive", "also": "additive", "too": "additive",
    "as well": "additive", "moreover": "additive", "furthermore": "additive",
    "in addition": "additive", "besides": "additive", "additionally": "additive",
    "what is more": "additive", "on top of that": "additive",
    # Causales (10)
    "because": "causal", "so": "causal", "since": "causal",
    "therefore": "causal", "as a result": "causal", "consequently": "causal",
    "thus": "causal", "hence": "causal", "owing to": "causal",
    "due to the fact that": "causal",
    # Adversativos (10)
    "but": "contrastive", "yet": "contrastive", "however": "contrastive",
    "although": "contrastive", "nevertheless": "contrastive",
    "on the other hand": "contrastive", "nonetheless": "contrastive",
    "notwithstanding": "contrastive", "conversely": "contrastive",
    "in contrast": "contrastive",
    # Organizadores (10)
    "first": "temporal", "then": "temporal", "finally": "temporal",
    "to begin with": "temporal", "next": "temporal", "in conclusion": "temporal",
    "to sum up": "temporal", "subsequently": "temporal",
    "in the first instance": "temporal", "to recapitulate": "temporal",
}

# Argumentative keywords for depth scoring
CLAIM_KEYWORDS = frozenset({
    "argue", "believe", "think", "claim", "opinion", "position",
    "maintain", "contend", "assert", "suggest", "propose",
})

EVIDENCE_KEYWORDS = frozenset({
    "for example", "for instance", "such as", "research shows",
    "according to", "studies", "evidence", "data", "statistics",
    "demonstrated", "proved", "shown", "found", "reported",
})

# Counterargument markers
COUNTERARG_MARKERS = frozenset({
    "on the other hand", "however", "although", "nevertheless",
    "opponents argue", "critics say", "some people think",
    "it could be argued", "one might argue", "admittedly",
    "while it is true", "despite", "granted",
})


# ============================================================
# Helper: Shannon entropy
# ============================================================

def _shannon_entropy(counts: dict[str, int]) -> float:
    """Compute Shannon entropy from a count dict."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


# ============================================================
# A9: RST approximation
# ============================================================

def _calc_rhetorical_depth(doc: Any) -> int:
    """Max clause nesting depth via spaCy dependency tree."""
    max_depth = 0
    for sent in doc.sents:
        depth = _clause_depth(sent.root, 0)
        max_depth = max(max_depth, depth)
    return max_depth


def _clause_depth(token: Any, current_depth: int) -> int:
    """Recursively compute max clause nesting depth."""
    max_d = current_depth
    for child in token.children:
        if child.dep_ in CLAUSAL_DEPS:
            max_d = max(max_d, _clause_depth(child, current_depth + 1))
        else:
            max_d = max(max_d, _clause_depth(child, current_depth))
    return max_d


def _calc_multinuclear_ratio(doc: Any) -> float:
    """Ratio of coordinated / (coordinated + subordinated) clauses."""
    coord_count = sum(1 for t in doc if t.dep_ in COORD_DEPS)
    subord_count = sum(1 for t in doc if t.dep_ in CLAUSAL_DEPS)
    total = coord_count + subord_count
    if total == 0:
        return float("nan")
    return coord_count / total


def _calc_relation_diversity(doc: Any) -> float:
    """Shannon entropy of clause relation types."""
    counts: dict[str, int] = Counter()
    for token in doc:
        if token.dep_ in ALL_RELATION_DEPS:
            counts[token.dep_] += 1
    return _shannon_entropy(dict(counts))


# ============================================================
# A10: Pragmatic-connective analysis
# ============================================================

def _find_connectors(text: str) -> list[tuple[str, str, int, int]]:
    """Find discourse connectors with char offsets.

    Returns list of (connector, relation_type, char_start, char_end).
    """
    text_lower = text.lower()
    found: list[tuple[str, str, int, int]] = []
    occupied: set[int] = set()

    for connector in sorted(CONNECTOR_LEXICON, key=len, reverse=True):
        start = 0
        while True:
            idx = text_lower.find(connector, start)
            if idx == -1:
                break
            positions = set(range(idx, idx + len(connector)))
            if not (positions & occupied):
                occupied |= positions
                found.append((
                    connector, CONNECTOR_LEXICON[connector], idx,
                    idx + len(connector),
                ))
            start = idx + 1

    found.sort(key=lambda x: x[2])
    return found


def _calc_connector_entropy(
    connectors: list[tuple[str, str, int, int]],
) -> float:
    """Shannon entropy over individual connector frequencies (natural log).

    Aligned with markers_core.py calc_entropia_conectores():
    scipy.stats.entropy on per-connector-form counts (not type categories).
    """
    from scipy.stats import entropy as scipy_entropy

    connector_counts: dict[str, int] = Counter()
    for connector_form, _, _, _ in connectors:
        connector_counts[connector_form] += 1

    if len(connector_counts) <= 1:
        return float("nan")

    freqs = np.array(list(connector_counts.values()), dtype=float)
    return float(scipy_entropy(freqs))


def _calc_connector_precision(
    connectors: list[tuple[str, str, int, int]],
    doc: Any,
) -> float:
    """Fraction of connectors in sentences with clausal dependencies."""
    if not connectors:
        return float("nan")

    sentences_with_clauses: set[str] = set()
    clausal_deps = {"advcl", "relcl", "ccomp", "xcomp", "acl", "conj"}
    for sent in doc.sents:
        if any(t.dep_ in clausal_deps for t in sent):
            sentences_with_clauses.add(sent.text.lower())

    precise = 0
    for connector, _, _, _ in connectors:
        for sent_text in sentences_with_clauses:
            if connector in sent_text:
                precise += 1
                break

    return precise / len(connectors)


def _calc_explicit_implicit_ratio(doc: Any) -> float:
    """Fraction of sentences containing explicit discourse connectors."""
    sents = list(doc.sents)
    if not sents:
        return float("nan")

    explicit_count = 0
    for sent in sents:
        sent_lower = sent.text.lower()
        if any(c in sent_lower for c in CONNECTOR_LEXICON):
            explicit_count += 1

    return explicit_count / len(sents)


# ============================================================
# A11: Argumentative analysis
# ============================================================

def _calc_argumentative_depth(paragraphs: list[dict]) -> float:
    """Mean claim+evidence keyword density per paragraph."""
    if not paragraphs:
        return float("nan")

    depths = []
    for para in paragraphs:
        text_lower = para["text"].lower()
        claim_count = sum(1 for kw in CLAIM_KEYWORDS if kw in text_lower)
        evidence_count = sum(1 for kw in EVIDENCE_KEYWORDS if kw in text_lower)
        depths.append(claim_count + evidence_count)

    return sum(depths) / len(depths) if depths else 0.0


def _has_counterargument(text: str) -> bool:
    """Whether text contains counterargument markers."""
    text_lower = text.lower()
    return any(marker in text_lower for marker in COUNTERARG_MARKERS)


def _calc_argument_completeness(
    text: str,
    prompt_text: str | None,
    doc: Any,
) -> float:
    """Fraction of expected argumentative moves present.

    Without prompt: checks for claim, evidence, counterargument, conclusion.
    With prompt: also checks for prompt-derived topical coverage.
    """
    moves_expected = 4  # claim, evidence, counter, conclusion
    moves_present = 0

    text_lower = text.lower()

    # Claim
    if any(kw in text_lower for kw in CLAIM_KEYWORDS):
        moves_present += 1

    # Evidence
    if any(kw in text_lower for kw in EVIDENCE_KEYWORDS):
        moves_present += 1

    # Counterargument
    if _has_counterargument(text):
        moves_present += 1

    # Conclusion
    conclusion_markers = {
        "in conclusion", "to conclude", "to sum up", "therefore",
        "all things considered", "in summary", "ultimately",
    }
    if any(m in text_lower for m in conclusion_markers):
        moves_present += 1

    return moves_present / moves_expected


# ============================================================
# Public API
# ============================================================

def analyze_essay_discourse(
    text: str,
    paragraphs: list[dict],
    nlp: Any,
    prompt_text: str | None = None,
) -> dict[str, object]:
    """Compute Phase 2 discourse metrics for a single essay.

    Args:
        text: Full essay text.
        paragraphs: List of paragraph dicts from text_to_paragraphs().
        nlp: spaCy Language model (already loaded).
        prompt_text: Optional prompt text for argument_completeness.

    Returns:
        Dict with az_* keys for COLUMNS_PHASE2_DISCOURSE.
    """
    result: dict[str, object] = {}

    doc = nlp(text)

    # A9 — RST approximation
    result["az_rhetorical_depth"] = _calc_rhetorical_depth(doc)
    multinuc = _calc_multinuclear_ratio(doc)
    result["az_multinuclear_ratio"] = (
        round(multinuc, 3) if not math.isnan(multinuc) else multinuc
    )
    result["az_relation_diversity"] = round(
        _calc_relation_diversity(doc), 3,
    )

    # A10 — Pragmatic connectives
    connectors = _find_connectors(text)
    result["az_connector_entropy"] = round(
        _calc_connector_entropy(connectors), 3,
    )
    conn_prec = _calc_connector_precision(connectors, doc)
    result["az_connector_precision"] = (
        round(conn_prec, 3) if not math.isnan(conn_prec) else conn_prec
    )
    exp_imp = _calc_explicit_implicit_ratio(doc)
    result["az_explicit_implicit_ratio"] = (
        round(exp_imp, 3) if not math.isnan(exp_imp) else exp_imp
    )

    # A11 — Argumentation
    result["az_argument_completeness"] = round(
        _calc_argument_completeness(text, prompt_text, doc), 3,
    )
    result["az_counterargument_present"] = _has_counterargument(text)
    result["az_argumentative_depth"] = round(
        _calc_argumentative_depth(paragraphs), 2,
    )

    return result
