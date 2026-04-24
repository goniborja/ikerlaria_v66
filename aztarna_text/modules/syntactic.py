"""
syntactic.py — Modulo sintactico AZTARNA v1 (T2)

Extrae features `syn_*` por ensayo a partir de un Doc spaCy ya parseado
con `en_core_web_trf`. Diseñado para recibir el doc como argumento, sin
re-parseo (single-pass nlp.pipe externo).

Especificacion: `recon_syntactic_order.md` §3 (spec embebido).

Taxonomia de constituyentes funcionales (decidida por Rol C, fija):
  S      — subject (nsubj, nsubjpass, csubj, csubjpass)
  V      — main verb (ROOT si es VERB/AUX)
  DO     — direct object (dobj)
  IO     — indirect object (dative, iobj)
  PP     — prep phrase (pobj de una prep)
  ADV    — adverbial (advmod, npadvmod)
  COMP   — complement clause (ccomp, xcomp)
  SUB    — subordinate adverbial clause (advcl, mark)
  COORD  — coordination (conj, cc)
  OTHER  — el resto

Alcance: solo matriz principal. `has_embedded_clause` marca recursion.

Autor: Borja Goni + Claude Code
Fecha: 2026-04-19
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

# --- Mapeo dep_ -> constituent label ---
DEP_TO_CONSTITUENT = {
    "nsubj": "S", "nsubjpass": "S", "csubj": "S", "csubjpass": "S",
    "dobj": "DO",
    "dative": "IO", "iobj": "IO",
    "pobj": "PP",
    "advmod": "ADV", "npadvmod": "ADV",
    "ccomp": "COMP", "xcomp": "COMP",
    "advcl": "SUB",
    "conj": "COORD", "cc": "COORD",
}

# Deps que indican embebido / subordinado
EMBEDDED_DEPS = {"ccomp", "xcomp", "advcl", "acl", "relcl"}

# --- Schema de feature names (para utils y consumidores) ---
FEATURE_NAMES = [
    "syn_n_sentences",
    "syn_func_seq_top1_frac",
    "syn_func_seq_entropy",
    "syn_func_seq_entropy_norm",
    "syn_func_seq_simpson_d",
    "syn_skeleton_top1_frac",
    "syn_skeleton_entropy",
    "syn_skeleton_entropy_norm",
    "syn_skeleton_simpson_d",
    "syn_opening_trigram_top1_frac",
    "syn_subj_pos_tokens_mean", "syn_subj_pos_tokens_cv",
    "syn_subj_pos_norm_mean", "syn_subj_pos_norm_cv",
    "syn_main_verb_pos_tokens_mean", "syn_main_verb_pos_tokens_cv",
    "syn_main_verb_pos_norm_mean", "syn_main_verb_pos_norm_cv",
    "syn_passive_rate",
    "syn_fronted_adv_rate",
    "syn_sub_first_rate",
    "syn_embedded_clause_rate",
    "syn_i_initial_rate",
    "syn_i_pos_norm_mean", "syn_i_pos_norm_cv",
    # Extra para criterio sanity Fase 2B
    "syn_i_present_rate",
]


def _constituent_label(token) -> str:
    """dep_ -> etiqueta funcional. Fallback: OTHER."""
    return DEP_TO_CONSTITUENT.get(token.dep_, "OTHER")


def _detect_passive(sent) -> bool:
    """True si sent tiene auxpass o nsubjpass."""
    return any(t.dep_ in {"auxpass", "nsubjpass"} for t in sent)


def _detect_fronted_adv(sent) -> bool:
    """True si el primer token non-punct/non-det es ADV/advmod."""
    for t in sent:
        if t.is_punct or t.dep_ == "det":
            continue
        return t.pos_ == "ADV" or t.dep_ == "advmod"
    return False


def _detect_sub_first(sent) -> bool:
    """Aproximacion: primer token contentful es SCONJ (although, because, etc.)
    o comienza con un `mark` que cuelga de advcl."""
    for t in sent:
        if t.is_punct or t.dep_ == "det":
            continue
        if t.pos_ == "SCONJ":
            return True
        if t.dep_ == "mark":
            return True
        return False
    return False


def _detect_embedded_clause(sent) -> bool:
    return any(t.dep_ in EMBEDDED_DEPS for t in sent)


def _root_verb(sent):
    """Devuelve el token ROOT si es VERB/AUX, si no None."""
    for t in sent:
        if t.dep_ == "ROOT" and t.pos_ in {"VERB", "AUX"}:
            return t
    return None


def _subj_head(sent):
    """Devuelve primer nsubj/nsubjpass/csubj/csubjpass head (NOUN/PRON/PROPN)."""
    for t in sent:
        if t.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}:
            return t
    return None


def _func_seq(sent) -> list[str]:
    """Secuencia de etiquetas funcionales en orden lineal (una por token relevante).

    Colapsa repeticiones consecutivas del mismo label para capturar el esqueleto.
    Excluye: puntuacion, det, aux (no auxpass), punct.
    """
    labels = []
    for t in sent:
        if t.is_punct or t.is_space or t.dep_ in {"det", "punct"}:
            continue
        # Saltar aux (no auxpass) — no aporta al constituente
        if t.dep_ == "aux":
            continue
        # Solo emitir si es cabeza de constituyente relevante
        label = _constituent_label(t)
        if label == "OTHER":
            continue
        # Colapsar repeticiones consecutivas
        if labels and labels[-1] == label:
            continue
        labels.append(label)
    # Insertar V justo despues del sujeto si no aparecio (a veces el ROOT no se etiqueta)
    root = _root_verb(sent)
    if root and "V" not in labels:
        # insertar despues del primer S si hay, si no al principio
        if "S" in labels:
            idx = labels.index("S") + 1
            labels.insert(idx, "V")
        else:
            labels.insert(0, "V")
    return labels


def _skeleton_seq(func_seq: list[str]) -> tuple[str, str, str]:
    """(first_constituent, V, first_complement).

    first_constituent: el primer elemento de func_seq.
    V: 'V' si aparece en func_seq, si no ''.
    first_complement: primer DO/PP/COMP/SUB despues de V, si existe.
    """
    first_c = func_seq[0] if func_seq else ""
    v = "V" if "V" in func_seq else ""
    first_comp = ""
    if v:
        v_idx = func_seq.index("V")
        for lab in func_seq[v_idx + 1:]:
            if lab in {"DO", "PP", "COMP", "SUB"}:
                first_comp = lab
                break
    return (first_c, v, first_comp)


def _opening_trigram(sent) -> tuple[str, ...]:
    """POS de los primeros 3 tokens no-punct y no-space.

    Saltar is_space es crítico: el dataset ELLIPSE tiene saltos de línea
    tokenizados como POS=SPACE que producían trigrams truncados de 1 elemento.
    """
    pos = []
    for t in sent:
        if t.is_punct or t.is_space:
            continue
        pos.append(t.pos_)
        if len(pos) == 3:
            break
    return tuple(pos)


def _i_positions(sent) -> tuple[list[int], list[float]]:
    """Posiciones (tokens relativos al sent.start) y normalizadas [0,1]
    de tokens 'I' (first person singular)."""
    tok_pos, norm_pos = [], []
    n = len(sent)
    for t in sent:
        # Aceptamos "I" y "I'm"/"I've" etc. (primera letra I y PRON)
        if t.pos_ == "PRON" and t.text == "I":
            rel = t.i - sent.start
            tok_pos.append(rel)
            norm_pos.append(round(rel / max(1, n - 1), 4))
    return tok_pos, norm_pos


def compute_syntactic_order(doc, source: str = "essay",
                             model: str | None = None,
                             id_: str | None = None) -> dict:
    """Produce dict con schema JSONL-ready (per-sentence + metadata).

    Args:
      doc: spaCy Doc ya parseado (en_core_web_trf).
      source: "essay" | "prompt".
      model: modelo IA que genero (si aplica).
      id_: identificador (essay_id | prompt_id).

    Returns:
      dict con campos:
        id, source, model, n_sentences, sentences (list of sent-dicts).
    """
    # Filtrar sentences que no contienen tokens contentful (solo puntuación/espacios)
    sents_raw = list(doc.sents)
    sents = [s for s in sents_raw
             if any(not (t.is_punct or t.is_space) for t in s)]
    n_dropped = len(sents_raw) - len(sents)
    sentence_dicts = []

    for si, sent in enumerate(sents):
        n_tokens = len(sent)
        func_seq = _func_seq(sent)
        skel = _skeleton_seq(func_seq)
        trigram = _opening_trigram(sent)

        subj = _subj_head(sent)
        if subj is not None:
            subj_pos_tokens = subj.i - sent.start
            subj_pos_norm = round(subj_pos_tokens / max(1, n_tokens - 1), 4)
            subj_null = None
        else:
            subj_pos_tokens = None
            subj_pos_norm = None
            subj_null = "no_nsubj_detected_in_sent"

        root = _root_verb(sent)
        if root is not None:
            mv_pos_tokens = root.i - sent.start
            mv_pos_norm = round(mv_pos_tokens / max(1, n_tokens - 1), 4)
            mv_null = None
        else:
            mv_pos_tokens = None
            mv_pos_norm = None
            mv_null = "no_root_verb_detected_in_sent"

        i_tok, i_norm = _i_positions(sent)

        sd = {
            "sent_index": si,
            "n_tokens": n_tokens,
            "func_seq": func_seq,
            "skeleton_seq": list(skel),
            "opening_trigram": list(trigram),
            "subj_pos_tokens": subj_pos_tokens,
            "subj_pos_norm": subj_pos_norm,
            "subj_null_reason": subj_null,
            "main_verb_pos_tokens": mv_pos_tokens,
            "main_verb_pos_norm": mv_pos_norm,
            "main_verb_null_reason": mv_null,
            "i_positions_tokens": i_tok,
            "i_positions_norm": i_norm,
            "has_embedded_clause": _detect_embedded_clause(sent),
            "passive": _detect_passive(sent),
            "fronted_adv": _detect_fronted_adv(sent),
            "sub_first": _detect_sub_first(sent),
        }
        sentence_dicts.append(sd)

    return {
        "id": id_,
        "source": source,
        "model": model,
        "n_sentences": len(sents),
        "sentences": sentence_dicts,
    }


def _entropy(freqs: Iterable[int], base: float = 2.0) -> float:
    """Shannon entropy en base `base` (default log2)."""
    total = sum(freqs)
    if total == 0:
        return 0.0
    h = 0.0
    for c in freqs:
        if c == 0:
            continue
        p = c / total
        h -= p * (math.log(p) / math.log(base))
    return h


def _simpson(freqs: Iterable[int]) -> float:
    """Simpson diversity D = 1 - sum(p_i^2)."""
    total = sum(freqs)
    if total == 0:
        return 0.0
    return 1.0 - sum((c / total) ** 2 for c in freqs)


def _cv(values: list[float]) -> float:
    """Coeficiente de variacion (std/mean)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    if mean == 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = var ** 0.5
    return round(std / abs(mean), 4)


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def aggregate_essay_sequences(sentence_dicts: list[dict]) -> dict:
    """Toma la lista de dicts por sentence y devuelve las 26 features `syn_*`."""
    if not sentence_dicts:
        return {k: None for k in FEATURE_NAMES}

    n = len(sentence_dicts)

    # --- func_seq: contar tuplas (secuencias completas por sent) ---
    seqs = [tuple(sd["func_seq"]) for sd in sentence_dicts]
    seq_counts = Counter(seqs)
    total_seqs = sum(seq_counts.values())
    n_unique_seqs = len(seq_counts)
    top1_frac = max(seq_counts.values()) / total_seqs if total_seqs else 0.0
    entropy_raw = _entropy(seq_counts.values(), base=2.0)
    if n_unique_seqs > 1:
        entropy_norm = entropy_raw / math.log2(n_unique_seqs)
    else:
        entropy_norm = 0.0
    simpson_d = _simpson(seq_counts.values())

    # --- skeleton_seq ---
    skels = [tuple(sd["skeleton_seq"]) for sd in sentence_dicts]
    skel_counts = Counter(skels)
    n_unique_skels = len(skel_counts)
    skel_top1 = max(skel_counts.values()) / sum(skel_counts.values()) if skel_counts else 0.0
    skel_entropy_raw = _entropy(skel_counts.values(), base=2.0)
    skel_entropy_norm = skel_entropy_raw / math.log2(n_unique_skels) if n_unique_skels > 1 else 0.0
    skel_simpson = _simpson(skel_counts.values())

    # --- opening trigram ---
    trigrams = [tuple(sd["opening_trigram"]) for sd in sentence_dicts]
    trig_counts = Counter(trigrams)
    trig_top1 = max(trig_counts.values()) / sum(trig_counts.values()) if trig_counts else 0.0

    # --- subj_pos ---
    subj_tok = [sd["subj_pos_tokens"] for sd in sentence_dicts]
    subj_norm = [sd["subj_pos_norm"] for sd in sentence_dicts]

    # --- main_verb_pos ---
    mv_tok = [sd["main_verb_pos_tokens"] for sd in sentence_dicts]
    mv_norm = [sd["main_verb_pos_norm"] for sd in sentence_dicts]

    # --- booleans rate ---
    passive_rate = sum(1 for sd in sentence_dicts if sd["passive"]) / n
    fronted_adv_rate = sum(1 for sd in sentence_dicts if sd["fronted_adv"]) / n
    sub_first_rate = sum(1 for sd in sentence_dicts if sd["sub_first"]) / n
    embedded_rate = sum(1 for sd in sentence_dicts if sd["has_embedded_clause"]) / n

    # --- "I" features ---
    # i_initial_rate: frac de apariciones de 'I' en posicion 0-1 / total de apariciones de I
    all_i_tok = []
    all_i_norm = []
    sents_with_i = 0
    for sd in sentence_dicts:
        tok = sd["i_positions_tokens"] or []
        norm = sd["i_positions_norm"] or []
        all_i_tok.extend(tok)
        all_i_norm.extend(norm)
        if tok:
            sents_with_i += 1
    n_i = len(all_i_tok)
    i_initial_rate = (sum(1 for p in all_i_tok if p <= 1) / n_i) if n_i else None
    i_pos_norm_mean = _mean(all_i_norm) if all_i_norm else None
    i_pos_norm_cv = _cv(all_i_norm) if all_i_norm else None
    i_present_rate = sents_with_i / n  # extra feature

    return {
        "syn_n_sentences": n,
        "syn_func_seq_top1_frac": round(top1_frac, 4),
        "syn_func_seq_entropy": round(entropy_raw, 4),
        "syn_func_seq_entropy_norm": round(entropy_norm, 4),
        "syn_func_seq_simpson_d": round(simpson_d, 4),
        "syn_skeleton_top1_frac": round(skel_top1, 4),
        "syn_skeleton_entropy": round(skel_entropy_raw, 4),
        "syn_skeleton_entropy_norm": round(skel_entropy_norm, 4),
        "syn_skeleton_simpson_d": round(skel_simpson, 4),
        "syn_opening_trigram_top1_frac": round(trig_top1, 4),
        "syn_subj_pos_tokens_mean": _mean(subj_tok),
        "syn_subj_pos_tokens_cv": _cv(subj_tok),
        "syn_subj_pos_norm_mean": _mean(subj_norm),
        "syn_subj_pos_norm_cv": _cv(subj_norm),
        "syn_main_verb_pos_tokens_mean": _mean(mv_tok),
        "syn_main_verb_pos_tokens_cv": _cv(mv_tok),
        "syn_main_verb_pos_norm_mean": _mean(mv_norm),
        "syn_main_verb_pos_norm_cv": _cv(mv_norm),
        "syn_passive_rate": round(passive_rate, 4),
        "syn_fronted_adv_rate": round(fronted_adv_rate, 4),
        "syn_sub_first_rate": round(sub_first_rate, 4),
        "syn_embedded_clause_rate": round(embedded_rate, 4),
        "syn_i_initial_rate": round(i_initial_rate, 4) if i_initial_rate is not None else None,
        "syn_i_pos_norm_mean": i_pos_norm_mean,
        "syn_i_pos_norm_cv": i_pos_norm_cv,
        "syn_i_present_rate": round(i_present_rate, 4),
    }
