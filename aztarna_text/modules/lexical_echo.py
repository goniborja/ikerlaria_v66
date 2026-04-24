"""
lexical_echo.py — Modulo de eco lexico prompt->output AZTARNA (T3)

Tres familias de features por par (prompt, output):
  1. Eco LITERAL: coverage/density/recurrence de lemas del prompt en el output.
  2. Eco PARAFRASTICO via WordNet (synonym/hyponym/hypernym distancia 1).
  3. Eco BPE FRAGMENTADO: rate de keywords rare-BPE (bpe_per_word>1.5).

Decisiones Rol B (BPE):
  - Tokenizer: GPT-2 `gpt2` de HuggingFace transformers.
  - Prepend single leading space antes de tokenizar cada palabra.
  - bpe_per_word = n_bpe_tokens / n_palabras_whitespace.
  - Umbral rare: bpe_per_word > 1.5 (consistente con T1.3.c).

Decisiones Rol C (WordNet):
  - POS: n (noun), v (verb), a (adj), r (adv).
  - Relaciones: synonym (lemma_names en mismo synset), hyponym dist=1,
    hypernym dist=1.
  - Multi-sense: TODOS los synsets del lema cuentan (sin WSD). Si alguno
    conecta con un lema del prompt, match.
  - Stopwords: NLTK english stopwords filtradas de lemas del prompt y output.

Decisiones Rol A (implementacion):
  - Lematizacion: spaCy en_core_web_trf (mismo modelo que T2).
  - Solo lemas de POS contentful: NOUN, VERB, ADJ, ADV, PROPN.
  - Matching por lemma en minuscula.
  - Output: features por ensayo + matches detallados para JSONL (trazabilidad).

Autor: Borja Goni + Claude Code
Fecha: 2026-04-19
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

def _ensure_nltk_resources():
    """Auto-descarga recursos NLTK si faltan (robustez pre-import)."""
    import nltk
    for resource_path, name in [("corpora/stopwords", "stopwords"),
                                  ("corpora/wordnet", "wordnet"),
                                  ("corpora/omw-1.4", "omw-1.4")]:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(name, quiet=True, raise_on_error=False)
            except Exception:
                pass


_ensure_nltk_resources()

from nltk.corpus import wordnet as wn  # noqa: E402
from nltk.corpus import stopwords as nltk_stopwords  # noqa: E402

# --- Constantes ---
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
try:
    STOPWORDS_EN = set(nltk_stopwords.words("english"))
except LookupError:
    STOPWORDS_EN = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "and", "or", "but", "if",
        "because", "as", "of", "at", "by", "for", "with", "to", "from", "in",
        "out", "on", "off", "over", "under", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
        "them", "their", "this", "that", "these", "those", "not", "no",
        "so", "too", "very", "can", "will", "just", "should",
    }
    import logging
    logging.getLogger(__name__).warning(
        "nltk 'stopwords' no disponible; usando set minimo fallback"
    )
BPE_RARE_THRESHOLD = 1.5

# --- Nombres de features producidos ---
FEATURE_NAMES = [
    # Familia 1 - Literal
    "lit_coverage", "lit_density_per_100w", "lit_first_position_norm",
    "lit_recurrence_max",
    "lit_coverage_anchors", "lit_coverage_zone_b_strict",
    "lit_coverage_micdrop",
    # Familia 2 - Parafrastico WordNet
    "para_synonym_density", "para_hyponym_density", "para_hypernym_density",
    # Familia 3 - BPE fragmentado
    "bpe_echo_rate_mean", "bpe_echo_rate_max",
    # Meta
    "n_prompt_lemmas", "n_essay_words", "n_essay_lemmas_content",
]


# ---- Helpers lematizacion ----

def content_lemmas(doc, lowercase: bool = True, drop_stop: bool = True) -> list[str]:
    """Lista de lemas de POS contentful (NOUN/VERB/ADJ/ADV/PROPN),
    en minuscula, sin stopwords ni puntuacion."""
    out = []
    for t in doc:
        if t.is_punct or t.is_space:
            continue
        if t.pos_ not in CONTENT_POS:
            continue
        lem = t.lemma_
        if lowercase:
            lem = lem.lower()
        if drop_stop and lem in STOPWORDS_EN:
            continue
        if not lem.strip():
            continue
        out.append(lem)
    return out


def pos_to_wn_pos(pos: str) -> str | None:
    """Mapeo POS spaCy -> POS WordNet. None si no aplica."""
    return {"NOUN": "n", "PROPN": "n", "VERB": "v",
            "ADJ": "a", "ADV": "r"}.get(pos)


# ---- Familia 1 — Literal ----

def compute_literal_echo(prompt_lemmas: list[str],
                          essay_doc,
                          anchor_phrases: list[str] | None = None,
                          zone_b_strict: list[str] | None = None,
                          micdrop: str | None = None) -> dict:
    """Eco literal prompt -> output.

    Returns dict con features Familia 1 + lista de matches para trazabilidad.
    """
    # Lemas del essay contentful
    essay_lemmas = content_lemmas(essay_doc, lowercase=True, drop_stop=True)
    # Lemas del essay EN ORDEN (para first_position). Con indice en el doc.
    essay_lemma_positions = []
    for t in essay_doc:
        if t.is_punct or t.is_space or t.pos_ not in CONTENT_POS:
            continue
        lem = t.lemma_.lower()
        if lem in STOPWORDS_EN or not lem.strip():
            continue
        essay_lemma_positions.append((t.i, lem, t.text))

    prompt_set = set(prompt_lemmas)
    # Palabras totales del essay (non-punct, non-space) para densidad
    n_essay_words = sum(1 for t in essay_doc if not t.is_punct and not t.is_space)

    # Coverage
    essay_lemma_set = set(essay_lemmas)
    if prompt_set:
        matched = prompt_set & essay_lemma_set
        lit_coverage = round(len(matched) / len(prompt_set), 4)
    else:
        matched = set()
        lit_coverage = None

    # Density: matches totales (contando cada aparicion) / (words/100)
    lemma_occurrences = Counter(essay_lemmas)
    # "matches literales" = total apariciones de lemas del prompt en el output
    n_matches_total = sum(lemma_occurrences[lem] for lem in matched)
    lit_density_per_100w = round(n_matches_total / max(1, n_essay_words / 100), 4)

    # First position norm: posicion token del primer match / len doc
    first_pos_norm = None
    first_match_lemma = None
    for tok_i, lem, surface in essay_lemma_positions:
        if lem in prompt_set:
            first_pos_norm = round(tok_i / max(1, len(essay_doc) - 1), 4)
            first_match_lemma = lem
            break

    # Recurrence max: maximo apariciones de un mismo lema del prompt
    if matched:
        lit_recurrence_max = max(lemma_occurrences[lem] for lem in matched)
    else:
        lit_recurrence_max = 0

    # Subanalisis por subconjuntos
    def _coverage_subset(subset: list[str] | None) -> float | None:
        if not subset:
            return None
        # Lematizar subset por split simple (ya vienen del map strict)
        # Para subsets multi-word, bajar a lemas simples es mas robusto
        subset_lemmas = set()
        for term in subset:
            for w in term.split():
                w_clean = re.sub(r"[^\w-]", "", w.lower())
                if w_clean and w_clean not in STOPWORDS_EN:
                    subset_lemmas.add(w_clean)
        if not subset_lemmas:
            return None
        match_sub = subset_lemmas & essay_lemma_set
        return round(len(match_sub) / len(subset_lemmas), 4)

    lit_coverage_anchors = _coverage_subset(anchor_phrases)
    lit_coverage_zone_b_strict = _coverage_subset(zone_b_strict)

    # Micdrop: binario (aparece o no la frase literal — case-insensitive
    # substring search sobre el texto original)
    if micdrop:
        essay_text = essay_doc.text
        lit_coverage_micdrop = int(micdrop.lower() in essay_text.lower())
    else:
        lit_coverage_micdrop = None

    # Matches para trazabilidad
    matches_trace = {
        "n_prompt_lemmas_unique": len(prompt_set),
        "n_essay_lemmas_content": len(essay_lemmas),
        "matched_lemmas": sorted(matched),
        "n_matches_total": n_matches_total,
        "first_match_lemma": first_match_lemma,
        "first_match_pos_norm": first_pos_norm,
    }

    return {
        "lit_coverage": lit_coverage,
        "lit_density_per_100w": lit_density_per_100w,
        "lit_first_position_norm": first_pos_norm,
        "lit_recurrence_max": lit_recurrence_max,
        "lit_coverage_anchors": lit_coverage_anchors,
        "lit_coverage_zone_b_strict": lit_coverage_zone_b_strict,
        "lit_coverage_micdrop": lit_coverage_micdrop,
        "_literal_trace": matches_trace,
        "_n_essay_words": n_essay_words,
        "_n_essay_lemmas_content": len(essay_lemmas),
    }


# ---- Familia 2 — Parafrastico WordNet ----

def _wn_related_lemmas(token, max_hop: int = 1) -> set[str]:
    """Para un token spaCy, devolver set de lemmas WordNet relacionados
    (synonyms + hyponyms dist=1 + hypernyms dist=1) sobre todos los senses.
    No WSD.
    """
    wn_pos = pos_to_wn_pos(token.pos_)
    if wn_pos is None:
        return set()
    related = set()
    # Buscar synsets por lemma spaCy
    lem = token.lemma_.lower()
    synsets = wn.synsets(lem, pos=wn_pos)
    if not synsets:
        # fallback: buscar por surface
        synsets = wn.synsets(token.text.lower(), pos=wn_pos)
    for syn in synsets:
        # Sinonimos: otros lemma_names del mismo synset
        for lname in syn.lemma_names():
            related.add(lname.lower().replace("_", " "))
        # Hyponyms distancia 1
        for hypo in syn.hyponyms():
            for lname in hypo.lemma_names():
                related.add(lname.lower().replace("_", " "))
        # Hypernyms distancia 1
        for hyper in syn.hypernyms():
            for lname in hyper.lemma_names():
                related.add(lname.lower().replace("_", " "))
    related.discard(lem)
    return related


def _wn_sense_types(token) -> dict[str, set[str]]:
    """Devuelve dict con tres sets separados: synonyms, hyponyms_d1, hypernyms_d1.
    Para poder atribuir por relacion."""
    syns, hypos, hypers = set(), set(), set()
    wn_pos = pos_to_wn_pos(token.pos_)
    if wn_pos is None:
        return {"syn": syns, "hypo": hypos, "hyper": hypers}
    lem = token.lemma_.lower()
    synsets = wn.synsets(lem, pos=wn_pos) or wn.synsets(token.text.lower(), pos=wn_pos)
    for syn in synsets:
        for lname in syn.lemma_names():
            syns.add(lname.lower().replace("_", " "))
        for hypo in syn.hyponyms():
            for lname in hypo.lemma_names():
                hypos.add(lname.lower().replace("_", " "))
        for hyper in syn.hypernyms():
            for lname in hyper.lemma_names():
                hypers.add(lname.lower().replace("_", " "))
    syns.discard(lem)
    return {"syn": syns, "hypo": hypos, "hyper": hypers}


def compute_paraphrastic_echo(prompt_lemmas: set[str],
                               essay_doc,
                               already_literal: set[str] | None = None) -> dict:
    """Por cada token contentful del essay, mirar sus relacionados WordNet
    y ver si hay interseccion con prompt_lemmas. Excluir tokens cuyo lema
    ya coincide literalmente (para no doble contar)."""
    if already_literal is None:
        already_literal = set()

    prompt_set = set(prompt_lemmas)

    # Contadores por relacion (a nivel de essay)
    n_syn_hits = 0
    n_hypo_hits = 0
    n_hyper_hits = 0

    # Trazabilidad
    examples_syn = []
    examples_hypo = []
    examples_hyper = []

    # Total tokens contentful considerados
    n_content = 0

    for t in essay_doc:
        if t.is_punct or t.is_space:
            continue
        if t.pos_ not in CONTENT_POS:
            continue
        lem = t.lemma_.lower()
        if lem in STOPWORDS_EN or not lem.strip():
            continue
        n_content += 1
        if lem in already_literal:
            continue  # ya contado como literal
        if lem in prompt_set:
            continue  # coincidencia literal, no parafrastica
        sense_types = _wn_sense_types(t)
        hit_any = False
        # Priorizamos attribucion: primero syn, luego hypo, luego hyper
        syn_overlap = sense_types["syn"] & prompt_set
        if syn_overlap:
            n_syn_hits += 1
            if len(examples_syn) < 10:
                examples_syn.append({"essay_lemma": lem,
                                      "prompt_matches": sorted(syn_overlap)[:3]})
            hit_any = True
        else:
            hypo_overlap = sense_types["hypo"] & prompt_set
            if hypo_overlap:
                n_hypo_hits += 1
                if len(examples_hypo) < 10:
                    examples_hypo.append({"essay_lemma": lem,
                                           "prompt_matches": sorted(hypo_overlap)[:3]})
                hit_any = True
            else:
                hyper_overlap = sense_types["hyper"] & prompt_set
                if hyper_overlap:
                    n_hyper_hits += 1
                    if len(examples_hyper) < 10:
                        examples_hyper.append({"essay_lemma": lem,
                                                "prompt_matches": sorted(hyper_overlap)[:3]})
                    hit_any = True

    # Densidad: hits / content tokens del essay
    def dens(k):
        return round(k / max(1, n_content), 4)

    return {
        "para_synonym_density": dens(n_syn_hits),
        "para_hyponym_density": dens(n_hypo_hits),
        "para_hypernym_density": dens(n_hyper_hits),
        "_paraphrase_trace": {
            "n_content_tokens_examined": n_content,
            "n_syn_hits": n_syn_hits,
            "n_hypo_hits": n_hypo_hits,
            "n_hyper_hits": n_hyper_hits,
            "examples_syn": examples_syn,
            "examples_hypo": examples_hypo,
            "examples_hyper": examples_hyper,
        },
    }


# ---- Familia 3 — BPE fragmentado ----

def _compile_keyword_pattern(keyword: str) -> re.Pattern:
    """Compila regex case-insensitive para keyword; maneja espacios."""
    parts = keyword.strip().split()
    # escape por palabra + \s+
    pat = r"\b" + r"\s+".join(re.escape(p) for p in parts) + r"\b"
    return re.compile(pat, re.IGNORECASE)


def compute_bpe_echo(essay_text: str,
                      rare_keywords_bpe: dict[str, dict]) -> dict:
    """Para cada keyword rare-BPE (bpe_per_word > 1.5) del map del prompt,
    buscar si aparece en el essay_text.

    rare_keywords_bpe: dict {keyword: {'bpe_tokens': int, 'bpe_per_word': float, 'chars_per_token': float}}
                      (precomputado en T1.3.c, cargado desde map_v2).

    Returns dict con:
      - per_keyword: {kw: {count, first_pos_norm, positions_norm}}
      - agregados: bpe_echo_rate_mean, bpe_echo_rate_max,
      - bpe_echo_rate_per_keyword (dict)
      - bpe_echo_position_mean_per_keyword (dict)
    """
    n_chars = max(1, len(essay_text))
    per_kw = {}
    rates = []
    for kw, bpe_info in rare_keywords_bpe.items():
        if bpe_info.get("bpe_per_word", 0) <= BPE_RARE_THRESHOLD:
            continue
        pat = _compile_keyword_pattern(kw)
        matches = list(pat.finditer(essay_text))
        count = len(matches)
        if count:
            first_pos_norm = round(matches[0].start() / n_chars, 4)
            positions_norm = [round(m.start() / n_chars, 4) for m in matches]
            position_mean = round(sum(positions_norm) / len(positions_norm), 4)
        else:
            first_pos_norm = None
            positions_norm = []
            position_mean = None
        per_kw[kw] = {
            "count": count,
            "rate": int(count > 0),  # binario: aparece o no (rate ensayo)
            "first_pos_norm": first_pos_norm,
            "position_mean": position_mean,
            "positions_norm": positions_norm,
        }
        rates.append(int(count > 0))

    if rates:
        bpe_echo_rate_mean = round(sum(rates) / len(rates), 4)
        bpe_echo_rate_max = max(rates)
    else:
        bpe_echo_rate_mean = None
        bpe_echo_rate_max = None

    return {
        "bpe_echo_rate_mean": bpe_echo_rate_mean,
        "bpe_echo_rate_max": bpe_echo_rate_max,
        "_bpe_per_keyword": per_kw,
    }


# ---- API principal ----

def compute_lexical_echo(prompt_doc,
                          essay_doc,
                          prompt_meta: dict) -> dict:
    """Orquesta las 3 familias para un (prompt, essay).

    prompt_meta (obtenido del map_v2 o de la anotacion retorica):
      - anchor_phrases: list[str] | None (textos)
      - zone_b_strict: list[str] | None
      - micdrop: str | None
      - rare_keywords_bpe: dict {kw: bpe_info} (solo las con bpe_per_word>1.5)

    Returns dict con las features + traces para JSONL.
    """
    prompt_lemmas_list = content_lemmas(prompt_doc, lowercase=True, drop_stop=True)
    prompt_lemmas = set(prompt_lemmas_list)

    literal = compute_literal_echo(
        prompt_lemmas_list,
        essay_doc,
        anchor_phrases=prompt_meta.get("anchor_phrases"),
        zone_b_strict=prompt_meta.get("zone_b_strict"),
        micdrop=prompt_meta.get("micdrop"),
    )
    matched_literal = set(literal["_literal_trace"]["matched_lemmas"])
    para = compute_paraphrastic_echo(prompt_lemmas, essay_doc,
                                      already_literal=matched_literal)
    bpe = compute_bpe_echo(essay_doc.text,
                            prompt_meta.get("rare_keywords_bpe", {}))

    out = {
        # Literal
        "lit_coverage": literal["lit_coverage"],
        "lit_density_per_100w": literal["lit_density_per_100w"],
        "lit_first_position_norm": literal["lit_first_position_norm"],
        "lit_recurrence_max": literal["lit_recurrence_max"],
        "lit_coverage_anchors": literal["lit_coverage_anchors"],
        "lit_coverage_zone_b_strict": literal["lit_coverage_zone_b_strict"],
        "lit_coverage_micdrop": literal["lit_coverage_micdrop"],
        # Parafrastico
        "para_synonym_density": para["para_synonym_density"],
        "para_hyponym_density": para["para_hyponym_density"],
        "para_hypernym_density": para["para_hypernym_density"],
        # BPE
        "bpe_echo_rate_mean": bpe["bpe_echo_rate_mean"],
        "bpe_echo_rate_max": bpe["bpe_echo_rate_max"],
        # Meta
        "n_prompt_lemmas": len(prompt_lemmas),
        "n_essay_words": literal["_n_essay_words"],
        "n_essay_lemmas_content": literal["_n_essay_lemmas_content"],
        # Traces para JSONL
        "_traces": {
            "literal": literal["_literal_trace"],
            "paraphrase": para["_paraphrase_trace"],
            "bpe_per_keyword": bpe["_bpe_per_keyword"],
        },
    }
    return out
