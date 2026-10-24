"""From letter scores to answers: permutation TTA and letter-prior debiasing.

Notation. For one question scored under option permutation `perm`, the model
returns log-probabilities over *display positions* (letters A..D). Display
position i shows original option `perm[i]`. `to_original` maps those scores
back to the original option order, where everything is aggregated.

Two cheap fixes for the A/B/C/D selection bias (no labels needed):

* Permutation TTA. Score each question under the 4 cyclic shifts (every option
  visits every letter once) and average the log-probs in the original frame.
  4x inference cost.
* PriDe (Zheng et al., ICLR 2024, arXiv 2309.03882). Under the cyclic shifts,
  the content term cancels in the per-letter mean log-prob, leaving the
  model's prior over letters. Estimate it on a small subset of questions, then
  divide it out of single-pass scores on the rest:
      log p_debiased(display i) = log p_obs(display i) - log prior(i)
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.labels import LETTERS


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


def to_original(display_scores: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    """Reorder per-display-position scores `(..., 4)` into original option order."""
    display_scores = np.asarray(display_scores)
    out = np.empty_like(display_scores)
    for i, p in enumerate(perm):
        out[..., p] = display_scores[..., i]
    return out


def aggregate_tta(
    display_logits: Sequence[np.ndarray], perms: Sequence[Sequence[int]]
) -> np.ndarray:
    """Mean log-softmax over permutations, in the original option order.

    display_logits: one `(4,)` (or `(N, 4)`) array per permutation.
    """
    if len(display_logits) != len(perms) or not perms:
        raise ValueError("need one score array per permutation")
    stacked = [to_original(log_softmax(s), p) for s, p in zip(display_logits, perms)]
    return log_softmax(np.mean(stacked, axis=0))


def estimate_letter_prior(display_logits_cyclic: np.ndarray) -> np.ndarray:
    """PriDe prior from questions scored under all cyclic shifts.

    display_logits_cyclic: `(N, P, 4)`, scores per display position for each
    of the P cyclic permutations of N questions. Returns the `(4,)` log prior.
    """
    x = np.asarray(display_logits_cyclic, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"expected (N, P, 4), got shape {x.shape}")
    per_question = log_softmax(log_softmax(x).mean(axis=1))  # (N, 4)
    # Average the per-question priors in probability space, as in PriDe.
    return np.log(np.exp(per_question).mean(axis=0))


def debias(display_logits: np.ndarray, log_prior: np.ndarray) -> np.ndarray:
    """Divide the letter prior out of single-pass display scores `(N, 4)`."""
    return log_softmax(log_softmax(display_logits) - np.asarray(log_prior))


def predict_letters(scores: np.ndarray) -> List[str]:
    """argmax over the original option order -> official letters."""
    return [LETTERS[i] for i in np.asarray(scores).argmax(axis=-1)]


def letter_distribution(letters: Sequence[str]) -> Dict[str, float]:
    n = max(len(letters), 1)
    return {letter: sum(x == letter for x in letters) / n for letter in LETTERS}


def kl_from_uniform(dist: Dict[str, float]) -> float:
    """KL(pred letter distribution || uniform); 0 means no position skew."""
    p = np.array([dist[k] for k in LETTERS], dtype=np.float64)
    p = p[p > 0]
    return float((p * np.log(p * len(LETTERS))).sum())


def split_for_prior(
    n: int, fraction: float, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Indices `(estimate, rest)` for PriDe's small estimation subset."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = max(1, int(round(n * fraction))) if n else 0
    return np.sort(idx[:k]), np.sort(idx[k:])
