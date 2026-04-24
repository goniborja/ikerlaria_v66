"""nlp_base.py — Core NLP analysis per paragraph.

Reimplements text-only logic from AZTARNA's linguistics.py + markers_core.py.
Computes: T-unit length, dependency distance, lexical sophistication, MTLD,
spelling errors, coherence, connector breakdown by type, N-level complexity.

Columns produced (COLUMNS_PHASE2_NLP):
    az_mean_t_unit_length, az_mean_dep_distance, az_lexical_sophistication,
    az_mtld, az_spelling_error_count, az_spelling_error_ratio,
    az_semantic_coherence_mean, az_connector_additive, az_connector_causal,
    az_connector_contrastive, az_connector_temporal, az_connector_conditional,
    az_nivel_complejidad_max, az_nivel_complejidad_mean

Project: AZTARNA_TEXT
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

MIN_WORDS_FOR_ANALYSIS = 3
MIN_TOKENS_FOR_MTLD = 10
MTLD_TTR_THRESHOLD = 0.72

# Subordination deps (from markers_core.py SUB_DEPS)
SUB_DEPS = frozenset({"advcl", "acl", "relcl", "ccomp", "xcomp", "csubj"})

# Content POS for lexical metrics
CONTENT_POS = frozenset({"NOUN", "VERB", "ADJ", "ADV"})

# Connector type mapping — aligned with lib/markers_core.py CONNECTORS
# 4 categories matching AZTARNA's lexicon exactly.
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


# ============================================================
# Lazy-loaded shared resources
# ============================================================

class NLPResources:
    """Lazy-loaded NLP resources shared across all essay analyses."""

    def __init__(self, spacy_model: str = "en_core_web_md") -> None:
        self._nlp: Any = None
        self._spell: Any = None
        self._common_words: set[str] | None = None
        self._connector_inventory: list[dict] | None = None
        self.spacy_model_name = spacy_model

    @property
    def nlp(self) -> Any:
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.spacy_model_name)
            logger.info("Loaded spaCy model: %s", self.spacy_model_name)
        return self._nlp

    @property
    def spell(self) -> Any:
        if self._spell is None:
            from spellchecker import SpellChecker
            self._spell = SpellChecker()
        return self._spell

    @property
    def common_words(self) -> set[str]:
        if self._common_words is None:
            from wordfreq import top_n_list
            self._common_words = set(top_n_list("en", 2000))
        return self._common_words

    @property
    def connector_inventory(self) -> list[dict]:
        """N1-N5 connector inventory from YAML (for nivel_complejidad)."""
        if self._connector_inventory is None:
            self._connector_inventory = _load_connector_inventory()
        return self._connector_inventory


def _load_connector_inventory() -> list[dict]:
    """Load the N1-N5 connector YAML from ingelesa/konfigurazioa/.

    YAML structure: {conectores: {category: {tier: [{form, n}]}}}
    Tier names: basicos, intermedios, avanzados
    """
    import yaml

    # Candidatos en orden: portable (copia junto a aztarna_text/), dev Windows, dev WSL.
    _here = Path(__file__).resolve()
    yaml_paths = [
        _here.parent.parent.parent / "conectores.yaml",          # <portable_root>/conectores.yaml
        _here.parent.parent / "conectores.yaml",                  # <aztarna_text>/conectores.yaml
        Path("D:/bozgorailua/ingelesa/konfigurazioa/conectores.yaml"),
        Path("D:/bozgorailua/lib/konfigurazioa/conectores.yaml"),
        Path("/mnt/d/bozgorailua/ingelesa/konfigurazioa/conectores.yaml"),
    ]

    for p in yaml_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)

            # Navigate nested structure
            categories = raw.get("conectores", raw)
            if not isinstance(categories, dict):
                logger.warning("Unexpected YAML structure in %s", p)
                return []

            inventory = []
            for category, tiers in categories.items():
                if not isinstance(tiers, dict):
                    continue
                for tier_name, items in tiers.items():
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        form = item.get("form", "").lower().strip()
                        n_level = item.get("n", 0)
                        if not form:
                            continue
                        inventory.append({
                            "form": form,
                            "n_level": n_level,
                            "category": category,
                            "tier": tier_name,
                            "n_words": len(form.split()),
                        })

            # Sort multiword first (longest) for correct matching
            inventory.sort(key=lambda x: x["n_words"], reverse=True)
            logger.info("Loaded %d connectors from %s", len(inventory), p)
            return inventory

    logger.warning("conectores.yaml not found — nivel_complejidad disabled")
    return []


# ============================================================
# Per-paragraph computations
# ============================================================

def _calc_t_unit_length(doc: Any) -> float:
    """Mean T-unit length: total_tokens / num_ROOT."""
    root_count = sum(1 for t in doc if t.dep_ == "ROOT")
    if root_count == 0:
        return float("nan")
    total_tokens = sum(1 for t in doc if not t.is_space)
    return total_tokens / root_count


def _calc_dependency_distance(doc: Any) -> float:
    """Mean abs(token.i - token.head.i) excluding ROOT/space/punct."""
    distances = [
        abs(token.i - token.head.i)
        for token in doc
        if token.dep_ != "ROOT" and not token.is_space and not token.is_punct
    ]
    if not distances:
        return float("nan")
    return float(np.mean(distances))


def _calc_lexical_sophistication(doc: Any, common_words: set[str]) -> float:
    """% of content lemmas NOT in the 2000 most frequent words."""
    total_content = 0
    sophisticated = 0
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        if token.pos_ not in CONTENT_POS:
            continue
        lemma = token.lemma_.lower()
        if not lemma.isalpha() or len(lemma) <= 1:
            continue
        total_content += 1
        if lemma not in common_words:
            sophisticated += 1
    if total_content == 0:
        return float("nan")
    return sophisticated / total_content


def _calc_mtld_one_direction(tokens: list[str]) -> float:
    """Run MTLD in one direction."""
    n_factors = 0
    factor_start = 0

    for i in range(1, len(tokens) + 1):
        segment = tokens[factor_start:i]
        ttr = len(set(segment)) / len(segment)
        if ttr <= MTLD_TTR_THRESHOLD:
            n_factors += 1
            factor_start = i

    remaining = tokens[factor_start:]
    if remaining:
        ttr = len(set(remaining)) / len(remaining)
        if ttr < 1.0:
            partial = (1.0 - ttr) / (1.0 - MTLD_TTR_THRESHOLD)
            n_factors += partial

    if n_factors == 0:
        return float(len(tokens)) if tokens else 0.0
    return len(tokens) / n_factors


def _calc_mtld(doc: Any) -> float | None:
    """MTLD (McCarthy & Jarvis 2010). None if too few tokens."""
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    if len(tokens) < MIN_TOKENS_FOR_MTLD:
        return None
    forward = _calc_mtld_one_direction(tokens)
    backward = _calc_mtld_one_direction(tokens[::-1])
    return (forward + backward) / 2.0


def _count_spelling_errors(doc: Any, spell: Any) -> int:
    """Count spelling errors (conservative method)."""
    candidates = []
    for token in doc:
        if not token.is_alpha or len(token.text) <= 1:
            continue
        if token.text[0].isupper():
            continue
        if token.is_stop:
            continue
        candidates.append(token.text.lower())

    if not candidates:
        return 0

    unknown = spell.unknown(candidates)
    errors = 0
    for word in unknown:
        correction = spell.correction(word)
        if correction is not None and correction != word:
            errors += 1
    return errors


def _calc_cross_paragraph_coherence(prev_doc: Any, curr_doc: Any) -> float | None:
    """Cosine similarity between last sentence of prev and first of curr."""
    prev_sents = list(prev_doc.sents)
    curr_sents = list(curr_doc.sents)

    if not prev_sents or not curr_sents:
        return None

    v_prev = prev_sents[-1].vector
    v_curr = curr_sents[0].vector

    norm_prev = np.linalg.norm(v_prev)
    norm_curr = np.linalg.norm(v_curr)

    if norm_prev == 0 or norm_curr == 0:
        return None

    return float(np.dot(v_prev, v_curr) / (norm_prev * norm_curr))


def _count_connector_types(text: str) -> dict[str, int]:
    """Count connectors by discourse relation type (additive, causal, etc.)."""
    text_lower = text.lower()
    counts: dict[str, int] = {
        "additive": 0, "causal": 0, "contrastive": 0,
        "temporal": 0,
    }
    occupied: set[int] = set()

    # Match multi-word connectors first (longest first)
    for connector in sorted(CONNECTOR_LEXICON, key=len, reverse=True):
        start = 0
        while True:
            idx = text_lower.find(connector, start)
            if idx == -1:
                break
            positions = set(range(idx, idx + len(connector)))
            if not (positions & occupied):
                occupied |= positions
                counts[CONNECTOR_LEXICON[connector]] += 1
            start = idx + 1

    return counts


def _find_connectors_enriched(doc: Any, text: str, inventory: list[dict]) -> list[dict]:
    """Find N1-N5 connectors with levels (reimplemented from markers_core)."""
    if not text.strip() or not inventory:
        return []

    text_lower = text.lower()
    found = []
    used_spans: list[tuple[int, int]] = []

    for item in inventory:
        form = item["form"]

        if item["n_words"] > 1:
            start = 0
            while True:
                idx = text_lower.find(form, start)
                if idx == -1:
                    break
                end = idx + len(form)
                if not any(s <= idx < e or s < end <= e for s, e in used_spans):
                    before_ok = (idx == 0 or not text_lower[idx - 1].isalpha())
                    after_ok = (end == len(text_lower) or not text_lower[end].isalpha())
                    if before_ok and after_ok:
                        found.append({
                            "form": form,
                            "category": item["category"],
                            "n_level": item["n_level"],
                        })
                        used_spans.append((idx, end))
                start = idx + 1
        else:
            for token in doc:
                if token.text.lower() != form:
                    continue
                cs = token.idx
                ce = token.idx + len(token.text)
                if any(s <= cs < e for s, e in used_spans):
                    continue
                if form == "that" and token.dep_ not in ("mark", "relcl", "ccomp"):
                    continue
                found.append({
                    "form": form,
                    "category": item["category"],
                    "n_level": item["n_level"],
                })
                used_spans.append((cs, ce))

    return found


# ============================================================
# Public API
# ============================================================

def analyze_essay_nlp(
    text: str,
    paragraphs: list[dict],
    resources: NLPResources,
) -> tuple[dict[str, object], list[dict]]:
    """Compute Phase 2 NLP metrics for a single essay.

    Args:
        text: Full essay text.
        paragraphs: List of paragraph dicts from text_to_paragraphs().
        resources: Shared NLPResources instance.

    Returns:
        Tuple of (dict with az_* keys, list of enriched connectors for M7).
    """
    nlp = resources.nlp
    result: dict[str, object] = {}

    # Per-paragraph accumulators
    t_units: list[float] = []
    dep_dists: list[float] = []
    sophistications: list[float] = []
    mtld_values: list[float] = []
    total_spelling_errors = 0
    total_alpha_tokens = 0
    coherence_values: list[float] = []
    all_enriched: list[dict] = []

    prev_doc = None

    for para in paragraphs:
        doc = nlp(para["text"])
        n_words = sum(1 for t in doc if not t.is_space and not t.is_punct)

        if n_words < MIN_WORDS_FOR_ANALYSIS:
            prev_doc = doc
            continue

        # T-unit length
        t_unit = _calc_t_unit_length(doc)
        if not np.isnan(t_unit):
            t_units.append(t_unit)

        # Dependency distance
        dep_dist = _calc_dependency_distance(doc)
        if not np.isnan(dep_dist):
            dep_dists.append(dep_dist)

        # Lexical sophistication
        soph = _calc_lexical_sophistication(doc, resources.common_words)
        if not np.isnan(soph):
            sophistications.append(soph)

        # MTLD
        mtld = _calc_mtld(doc)
        if mtld is not None:
            mtld_values.append(mtld)

        # Spelling errors
        errors = _count_spelling_errors(doc, resources.spell)
        total_spelling_errors += errors
        total_alpha_tokens += sum(
            1 for t in doc if t.is_alpha and len(t.text) > 1
        )

        # Inter-paragraph coherence
        if prev_doc is not None:
            coh = _calc_cross_paragraph_coherence(prev_doc, doc)
            if coh is not None:
                coherence_values.append(coh)

        # Enriched connectors (N-level)
        enriched = _find_connectors_enriched(
            doc, para["text"], resources.connector_inventory,
        )
        all_enriched.extend(enriched)

        prev_doc = doc

    # Aggregate per-paragraph values to essay level
    result["az_mean_t_unit_length"] = (
        round(float(np.mean(t_units)), 3) if t_units else float("nan")
    )
    result["az_mean_dep_distance"] = (
        round(float(np.mean(dep_dists)), 3) if dep_dists else float("nan")
    )
    result["az_lexical_sophistication"] = (
        round(float(np.mean(sophistications)), 4)
        if sophistications else float("nan")
    )
    result["az_mtld"] = (
        round(float(np.mean(mtld_values)), 2) if mtld_values else float("nan")
    )
    result["az_spelling_error_count"] = total_spelling_errors
    result["az_spelling_error_ratio"] = (
        round(total_spelling_errors / total_alpha_tokens, 4)
        if total_alpha_tokens > 0 else float("nan")
    )
    result["az_semantic_coherence_mean"] = (
        round(float(np.mean(coherence_values)), 4)
        if coherence_values else float("nan")
    )

    # Connector type breakdown
    conn_types = _count_connector_types(text)
    result["az_connector_additive"] = conn_types["additive"]
    result["az_connector_causal"] = conn_types["causal"]
    result["az_connector_contrastive"] = conn_types["contrastive"]
    result["az_connector_temporal"] = conn_types["temporal"]
    # az_connector_conditional removed — markers_core.py uses 4 categories

    # N-level complexity
    if all_enriched:
        levels = [c["n_level"] for c in all_enriched]
        result["az_nivel_complejidad_max"] = max(levels)
        result["az_nivel_complejidad_mean"] = round(
            sum(levels) / len(levels), 2,
        )
    else:
        result["az_nivel_complejidad_max"] = 0
        result["az_nivel_complejidad_mean"] = 0.0

    return result, all_enriched
