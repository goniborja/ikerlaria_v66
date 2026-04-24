"""pragmatic.py — Pragmatic analysis modules.

Reimplements text-only logic from AZTARNA's discourse.py:
  - Coreference (A13): pronoun-antecedent chains
  - EDU segmentation (A14): Elementary Discourse Units
  - Thematic progression (A15): only az_longest_constant_run (rest in Trojan)
  - Semantic anchor (A12): paragraph-prompt distance (requires prompt_text)
  - Argumentative subordination (M7): argumentative vs decorative connectors

Columns produced:
    az_coref_chains, az_coref_longest, az_coref_cross_paragraph, az_coref_density,
    az_edu_total, az_edu_mean_length, az_edu_complexity,
    az_edu_subordination_ratio, az_edu_coordination_ratio, az_edu_juxtaposition_ratio,
    az_longest_constant_run,
    az_anchor_distance, az_anchor_variance, az_anchor_max_deviation,
    az_arg_sub_ratio, az_arg_sub_argumentative, az_arg_sub_decorative

Project: AZTARNA_TEXT
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

# Coreference: resolvable pronoun dependencies
RESOLVABLE_DEPS = frozenset({
    "nsubj", "nsubjpass", "dobj", "pobj", "poss", "attr",
})

PLURAL_PRONOUNS = frozenset({
    "they", "them", "their", "theirs", "themselves",
})
MASCULINE_PRONOUNS = frozenset({"he", "him", "his", "himself"})
FEMININE_PRONOUNS = frozenset({"she", "her", "hers", "herself"})
NEUTRAL_PRONOUNS = frozenset({"it", "its", "itself", "this", "that"})
ALL_RESOLVABLE = PLURAL_PRONOUNS | MASCULINE_PRONOUNS | FEMININE_PRONOUNS | NEUTRAL_PRONOUNS

SALIENCE_WEIGHTS = {
    "nsubj": 3.0, "nsubjpass": 2.5, "dobj": 2.0,
    "pobj": 1.5, "attr": 1.5, "ROOT": 1.0,
}

# EDU segmentation
SUBORDINATION_DEPS = frozenset({"advcl", "relcl", "ccomp", "xcomp", "acl"})
COORDINATION_DEPS = frozenset({"conj"})
JUXTAPOSITION_DEPS = frozenset({"parataxis"})
ALL_CLAUSE_DEPS = SUBORDINATION_DEPS | COORDINATION_DEPS | JUXTAPOSITION_DEPS
MIN_EDU_TOKENS = 2

# Thematic progression
CONTENT_POS = frozenset({"NOUN", "VERB", "ADJ", "ADV", "PROPN"})
SPLIT_RHEME_THRESHOLD = 3

# Argumentative subordination categories (from markers_core YAML)
ARGUMENTATIVE_CATEGORIES = frozenset({
    "causales", "adversativos", "concesivo_argumentativos",
})


# ============================================================
# COREFERENCE (A13)
# ============================================================

def _is_noun_candidate(token: Any) -> bool:
    return token.pos_ in {"NOUN", "PROPN"} and token.is_alpha


def _number_compatible(pron_lower: str, candidate: Any) -> bool:
    if pron_lower in NEUTRAL_PRONOUNS:
        return True
    is_plural_pron = pron_lower in PLURAL_PRONOUNS
    is_plural_noun = candidate.tag_ in {"NNS", "NNPS"}
    return is_plural_pron == is_plural_noun


def _find_antecedent(
    pronoun_token: Any,
    current_nouns: list,
    prev_nouns: list,
) -> Any | None:
    pron_lower = pronoun_token.text.lower()
    if pron_lower not in ALL_RESOLVABLE:
        return None

    candidates = []
    for noun in current_nouns:
        if noun.i < pronoun_token.i and _number_compatible(pron_lower, noun):
            dist = pronoun_token.i - noun.i
            score = SALIENCE_WEIGHTS.get(noun.dep_, 1.0) / max(dist, 1)
            candidates.append((score, noun))

    for noun in prev_nouns:
        if _number_compatible(pron_lower, noun):
            dist = pronoun_token.i - noun.i
            score = SALIENCE_WEIGHTS.get(noun.dep_, 1.0) / max(dist, 1) * 0.8
            candidates.append((score, noun))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _analyze_coreference(doc: Any, n_paragraphs: int) -> dict[str, object]:
    """Build coreference chains and compute global metrics."""
    sents = list(doc.sents)
    if not sents:
        return {"az_coref_chains": 0, "az_coref_longest": 0,
                "az_coref_cross_paragraph": 0, "az_coref_density": 0.0}

    # Precompute nouns per sentence
    sent_nouns = [[t for t in s if _is_noun_candidate(t)] for s in sents]

    # Resolve pronouns → chains
    chains_dict: dict[str, list[int]] = defaultdict(list)  # lemma → [sent_indices]

    for sent_idx, sent in enumerate(sents):
        prev = sent_nouns[sent_idx - 1] if sent_idx > 0 else []
        curr = sent_nouns[sent_idx]

        for token in sent:
            if token.pos_ != "PRON" or token.dep_ not in RESOLVABLE_DEPS:
                continue
            if token.text.lower() not in ALL_RESOLVABLE:
                continue

            antecedent = _find_antecedent(token, curr, prev)
            if antecedent is not None:
                lemma = antecedent.lemma_.lower()
                chains_dict[lemma].append(sent_idx)
                # Also register antecedent's sentence
                ant_sent = next(
                    (i for i, s in enumerate(sents)
                     if s.start <= antecedent.i < s.end), 0,
                )
                chains_dict[lemma].append(ant_sent)

    # Filter chains with >= 2 unique mentions
    chains = {
        lemma: sorted(set(indices))
        for lemma, indices in chains_dict.items()
        if len(set(indices)) >= 2
    }

    total_chains = len(chains)
    longest = max(
        (len(indices) for indices in chains.values()), default=0,
    )

    # Cross-paragraph chains
    n_sents = len(sents)
    sents_per_para = max(1, n_sents // max(1, n_paragraphs))
    cross_para = 0
    for indices in chains.values():
        paras_touched = {min(si // sents_per_para, n_paragraphs - 1)
                         for si in indices}
        if len(paras_touched) > 1:
            cross_para += 1

    # Referential density
    n_words = sum(1 for t in doc if not t.is_space and not t.is_punct)
    total_mentions = sum(len(v) for v in chains.values())
    density = total_mentions / n_words if n_words > 0 else 0.0

    return {
        "az_coref_chains": total_chains,
        "az_coref_longest": longest,
        "az_coref_cross_paragraph": cross_para,
        "az_coref_density": round(density, 4),
    }


# ============================================================
# EDU SEGMENTATION (A14)
# ============================================================

def _segment_sentence_edus(sent: Any) -> list[dict]:
    """Segment a sentence into EDUs. Returns list of {n_words, role, relation}."""
    tokens = list(sent)
    if not tokens:
        return []

    # Find clausal subtrees
    satellite_spans: list[tuple[int, int, str]] = []
    for token in tokens:
        if token.dep_ in ALL_CLAUSE_DEPS:
            subtree = sorted(t.i for t in token.subtree)
            if subtree:
                satellite_spans.append((subtree[0], subtree[-1], token.dep_))

    if not satellite_spans:
        n_words = sum(1 for t in tokens if not t.is_space and not t.is_punct)
        return [{"n_words": max(1, n_words), "role": "nucleus", "relation": "root"}]

    # Remove overlapping spans (keep longer)
    satellite_spans.sort(key=lambda x: x[0])
    merged: list[tuple[int, int, str]] = []
    for span in satellite_spans:
        if merged and span[0] <= merged[-1][1]:
            if (span[1] - span[0]) > (merged[-1][1] - merged[-1][0]):
                merged[-1] = span
        else:
            merged.append(span)

    edus = []
    assigned: set[int] = set()

    for start_i, end_i, dep_label in merged:
        sat_tokens = [t for t in tokens if start_i <= t.i <= end_i]
        n_words = sum(1 for t in sat_tokens if not t.is_space and not t.is_punct)
        if n_words < MIN_EDU_TOKENS:
            continue
        for t in sat_tokens:
            assigned.add(t.i)
        edus.append({"n_words": n_words, "role": "satellite", "relation": dep_label})

    # Nucleus from unassigned tokens
    nuc_tokens = [t for t in tokens if t.i not in assigned]
    n_words_nuc = sum(1 for t in nuc_tokens if not t.is_space and not t.is_punct)
    if n_words_nuc >= MIN_EDU_TOKENS:
        edus.append({"n_words": n_words_nuc, "role": "nucleus", "relation": "root"})

    return edus


def _analyze_edus(doc: Any) -> dict[str, object]:
    """Segment document into EDUs and compute metrics."""
    all_edus = []
    for sent in doc.sents:
        all_edus.extend(_segment_sentence_edus(sent))

    total = len(all_edus)
    if total == 0:
        return {
            "az_edu_total": 0, "az_edu_mean_length": 0.0,
            "az_edu_complexity": 0.0, "az_edu_subordination_ratio": 0.0,
            "az_edu_coordination_ratio": 0.0, "az_edu_juxtaposition_ratio": 0.0,
        }

    mean_len = float(np.mean([e["n_words"] for e in all_edus]))
    n_sat = sum(1 for e in all_edus if e["role"] == "satellite")

    n_sub = sum(1 for e in all_edus if e["relation"] in SUBORDINATION_DEPS)
    n_coord = sum(1 for e in all_edus if e["relation"] in COORDINATION_DEPS)
    n_juxt = sum(1 for e in all_edus if e["relation"] in JUXTAPOSITION_DEPS)
    total_rel = n_sub + n_coord + n_juxt

    return {
        "az_edu_total": total,
        "az_edu_mean_length": round(mean_len, 2),
        "az_edu_complexity": round(n_sat / total, 4),
        "az_edu_subordination_ratio": round(n_sub / total_rel, 4) if total_rel else 0.0,
        "az_edu_coordination_ratio": round(n_coord / total_rel, 4) if total_rel else 0.0,
        "az_edu_juxtaposition_ratio": round(n_juxt / total_rel, 4) if total_rel else 0.0,
    }


# ============================================================
# THEMATIC PROGRESSION — az_longest_constant_run only
# ============================================================

def _extract_theme(sent: Any) -> set[str]:
    """Theme = content lemmas in nsubj subtree."""
    theme: set[str] = set()
    for token in sent:
        if token.dep_ in {"nsubj", "nsubjpass"}:
            for t in token.subtree:
                if (t.pos_ in CONTENT_POS and t.is_alpha
                        and not t.is_stop and len(t.lemma_) > 1):
                    theme.add(t.lemma_.lower())
            break
    return theme


def _extract_rheme(sent: Any, theme: set[str]) -> set[str]:
    """Rheme = content lemmas NOT in theme."""
    rheme: set[str] = set()
    for token in sent:
        if (token.pos_ in CONTENT_POS and token.is_alpha
                and not token.is_stop and len(token.lemma_) > 1):
            lemma = token.lemma_.lower()
            if lemma not in theme:
                rheme.add(lemma)
    return rheme


def _classify_progression(
    theme_curr: set[str],
    theme_prev: set[str],
    rheme_prev: set[str],
) -> str:
    """Classify thematic progression type."""
    if not theme_curr:
        return "derived"
    if theme_curr & rheme_prev and len(rheme_prev) >= SPLIT_RHEME_THRESHOLD:
        return "split"
    if theme_curr & rheme_prev:
        return "linear"
    if theme_curr & theme_prev:
        return "constant"
    return "derived"


def _calc_longest_constant_run(doc: Any) -> int:
    """Compute longest consecutive run of 'constant' thematic progression.

    This is the metric that catches essays like #278 — repeating the same
    theme sentence after sentence without development.
    """
    sents = list(doc.sents)
    if len(sents) < 2:
        return 0

    themes = []
    rhemes = []
    for sent in sents:
        t = _extract_theme(sent)
        r = _extract_rheme(sent, t)
        themes.append(t)
        rhemes.append(r)

    max_run = 0
    current_run = 0

    for i in range(1, len(sents)):
        prog = _classify_progression(themes[i], themes[i - 1], rhemes[i - 1])
        if prog == "constant":
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return max_run


# ============================================================
# SEMANTIC ANCHOR (A12) — requires prompt_text
# ============================================================

def _calc_semantic_anchor(
    paragraphs: list[dict],
    prompt_text: str | None,
    nlp: Any,
) -> dict[str, object]:
    """Compute paragraph-to-prompt semantic distance."""
    nan = float("nan")
    if not prompt_text or not prompt_text.strip():
        return {
            "az_anchor_distance": nan,
            "az_anchor_variance": nan,
            "az_anchor_max_deviation": nan,
        }

    task_doc = nlp(prompt_text)
    if not task_doc.has_vector:
        return {
            "az_anchor_distance": nan,
            "az_anchor_variance": nan,
            "az_anchor_max_deviation": nan,
        }

    distances = []
    for para in paragraphs:
        para_doc = nlp(para["text"])
        if para_doc.has_vector and task_doc.has_vector:
            sim = para_doc.similarity(task_doc)
            distances.append(1.0 - sim)

    if not distances:
        return {
            "az_anchor_distance": nan,
            "az_anchor_variance": nan,
            "az_anchor_max_deviation": nan,
        }

    return {
        "az_anchor_distance": round(float(np.mean(distances)), 4),
        "az_anchor_variance": round(float(np.var(distances)), 6),
        "az_anchor_max_deviation": round(float(max(distances)), 4),
    }


# ============================================================
# ARGUMENTATIVE SUBORDINATION (M7)
# ============================================================

def _calc_arg_subordination(
    connectors_enriched: list[dict],
) -> dict[str, object]:
    """Ratio of argumentative vs decorative connectors.

    Uses the N-level enriched connectors from nlp_base.
    Argumentative categories: causales, adversativos, concesivo_argumentativos.
    """
    if not connectors_enriched:
        return {
            "az_arg_sub_ratio": float("nan"),
            "az_arg_sub_argumentative": 0,
            "az_arg_sub_decorative": 0,
        }

    argumentative = 0
    decorative = 0
    for conn in connectors_enriched:
        cat = conn.get("category", "")
        if cat in ARGUMENTATIVE_CATEGORIES:
            argumentative += 1
        else:
            decorative += 1

    total = argumentative + decorative
    ratio = argumentative / total if total > 0 else 0.0

    return {
        "az_arg_sub_ratio": round(ratio, 4),
        "az_arg_sub_argumentative": argumentative,
        "az_arg_sub_decorative": decorative,
    }


# ============================================================
# Public API
# ============================================================

def analyze_essay_pragmatic(
    text: str,
    paragraphs: list[dict],
    nlp: Any,
    prompt_text: str | None = None,
    connectors_enriched: list[dict] | None = None,
) -> dict[str, object]:
    """Compute Phase 3 pragmatic metrics for a single essay.

    Args:
        text: Full essay text.
        paragraphs: List of paragraph dicts from text_to_paragraphs().
        nlp: spaCy Language model (already loaded).
        prompt_text: Optional prompt text for semantic anchor.
        connectors_enriched: Enriched connector list from NLP phase
            (for argumentative subordination).

    Returns:
        Dict with az_* keys for Phase 3 columns.
    """
    doc = nlp(text)
    result: dict[str, object] = {}

    # Coreference (A13)
    coref = _analyze_coreference(doc, len(paragraphs))
    result.update(coref)

    # EDU segmentation (A14)
    edu = _analyze_edus(doc)
    result.update(edu)

    # Thematic progression — only longest_constant_run
    result["az_longest_constant_run"] = _calc_longest_constant_run(doc)

    # Semantic anchor (A12)
    anchor = _calc_semantic_anchor(paragraphs, prompt_text, nlp)
    result.update(anchor)

    # Argumentative subordination (M7)
    arg_sub = _calc_arg_subordination(connectors_enriched or [])
    result.update(arg_sub)

    return result
