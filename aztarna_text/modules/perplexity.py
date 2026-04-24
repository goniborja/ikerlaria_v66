"""perplexity.py — GPT-2 surprisal computation.

Reimplements text-only logic from AZTARNA's linguistics.py (Step 11).
Computes mean and std surprisal in bits across all GPT-2 tokens.

Columns produced (COLUMNS_PHASE4_PERPLEXITY):
    az_surprisal_mean, az_surprisal_std

Project: AZTARNA_TEXT
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MAX_GPT2_TOKENS = 1024


# ============================================================
# Lazy GPT-2 engine
# ============================================================

class GPT2Engine:
    """Lazy-loaded GPT-2 model for surprisal computation."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._available: bool | None = None

    @property
    def is_available(self) -> bool:
        if self._available is None:
            self._try_load()
        return self._available

    @property
    def model(self) -> Any:
        if self._available is None:
            self._try_load()
        return self._model

    @property
    def tokenizer(self) -> Any:
        if self._available is None:
            self._try_load()
        return self._tokenizer

    @property
    def device(self) -> str:
        if self._available is None:
            self._try_load()
        return self._device

    def _try_load(self) -> None:
        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            self._model = GPT2LMHeadModel.from_pretrained("gpt2").to(self._device)
            self._model.eval()
            self._available = True
            logger.info("GPT-2 loaded on %s", self._device)
        except (ImportError, ModuleNotFoundError) as e:
            self._available = False
            logger.warning(
                "torch/transformers not available: %s. "
                "Surprisal will be NaN. "
                "Install with: pip install torch transformers", e,
            )
        except Exception as e:
            self._available = False
            logger.warning("GPT-2 init failed: %s", e)


# Singleton
_engine: GPT2Engine | None = None


def _get_engine() -> GPT2Engine:
    global _engine
    if _engine is None:
        _engine = GPT2Engine()
    return _engine


# ============================================================
# Core computation
# ============================================================

def _compute_surprisal(text: str, engine: GPT2Engine) -> tuple[float, float]:
    """Compute mean and std WORD-level surprisal in bits.

    Matches AZTARNA methodology: sub-token surprisals are SUMMED per word,
    then mean/std computed across words.  This makes values comparable
    with the main pipeline's run_perplexity().

    Returns (mean_bits, std_bits) or (NaN, NaN) if text too short.
    """
    import re
    import torch

    encoding = engine.tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_GPT2_TOKENS,
    )
    input_ids = encoding.input_ids.to(engine.device)
    offsets = encoding.offset_mapping[0].tolist()
    n_tokens = input_ids.shape[1]

    if n_tokens < 2:
        return float("nan"), float("nan")

    with torch.no_grad():
        outputs = engine.model(input_ids, labels=input_ids)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    # Per-token surprisal in bits (first token = 0, no prior context)
    ln2 = math.log(2)
    token_surprisals = [0.0] + [float(l) / ln2 for l in per_token_loss.cpu().numpy()]

    # Align sub-tokens to whitespace-delimited words (SUM per word)
    word_surprisals: list[float] = []
    for m in re.finditer(r"\S+", text):
        ws, we = m.start(), m.end()
        word_surp = 0.0
        for i, (cs, ce) in enumerate(offsets):
            if ce <= ws or cs >= we:
                continue
            word_surp += token_surprisals[i]
        word_surprisals.append(word_surp)

    if not word_surprisals:
        return float("nan"), float("nan")

    mean_bits = float(np.mean(word_surprisals))
    std_bits = float(np.std(word_surprisals))

    return mean_bits, std_bits


# ============================================================
# Public API
# ============================================================

def analyze_essay_perplexity(text: str) -> dict[str, object]:
    """Compute GPT-2 surprisal metrics for a single essay.

    Args:
        text: Full essay text.

    Returns:
        Dict with az_surprisal_mean and az_surprisal_std.
    """
    nan = float("nan")
    engine = _get_engine()

    if not engine.is_available:
        return {"az_surprisal_mean": nan, "az_surprisal_std": nan}

    if not text or not text.strip():
        return {"az_surprisal_mean": nan, "az_surprisal_std": nan}

    try:
        mean_bits, std_bits = _compute_surprisal(text, engine)
        return {
            "az_surprisal_mean": round(mean_bits, 4) if not np.isnan(mean_bits) else nan,
            "az_surprisal_std": round(std_bits, 4) if not np.isnan(std_bits) else nan,
        }
    except Exception as e:
        logger.warning("Surprisal computation failed: %s", e)
        return {"az_surprisal_mean": nan, "az_surprisal_std": nan}
