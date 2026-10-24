import numpy as np
import pytest

from src.data import cyclic_perms, permute
from src.decoding import (
    aggregate_tta,
    debias,
    estimate_letter_prior,
    kl_from_uniform,
    letter_distribution,
    log_softmax,
    predict_letters,
    split_for_prior,
    to_original,
)
from tests.conftest import make_example


def test_to_original_inverts_permutation(example):
    perm = (2, 0, 3, 1)
    shown = permute(example, perm)
    display = np.zeros(4)
    display[shown.label] = 1.0  # model is right in display space
    orig = to_original(display, shown.perm)
    assert orig.argmax() == example.label


def biased_logits(label, perms, bias=2.0, signal=1.0):
    """Display logits: correct answer +signal, letter A +bias."""
    ex = make_example(label=label)
    out = []
    for perm in perms:
        shown = permute(ex, perm)
        z = np.zeros(4)
        z[shown.label] += signal
        z[0] += bias
        out.append(z)
    return np.array(out)


def test_cyclic_tta_cancels_letter_bias():
    perms = cyclic_perms()
    for label in range(4):
        d = biased_logits(label, perms)
        assert d[0].argmax() == 0  # single pass picks the biased letter
        assert aggregate_tta(list(d), perms).argmax() == label


def test_pride_recovers_prior_and_debiases():
    perms = cyclic_perms()
    data = np.stack([biased_logits(i % 4, perms) for i in range(40)])  # (N, P, 4)
    prior = estimate_letter_prior(data)
    assert prior.argmax() == 0 and np.exp(prior).sum() == pytest.approx(1.0)
    fixed = debias(data[:, 0], prior)
    assert (fixed.argmax(axis=1) == np.arange(40) % 4).all()


def test_log_softmax_and_helpers():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert np.exp(log_softmax(x)).sum() == pytest.approx(1.0)
    assert predict_letters(x) == ["D"]
    assert kl_from_uniform(letter_distribution(list("ABCD"))) == pytest.approx(0.0)
    assert kl_from_uniform(letter_distribution(list("AAAA"))) == pytest.approx(
        np.log(4)
    )
    est, rest = split_for_prior(100, 0.05)
    assert len(est) == 5 and len(rest) == 95 and not set(est) & set(rest)
    with pytest.raises(ValueError):
        aggregate_tta([np.zeros(4)], [])
