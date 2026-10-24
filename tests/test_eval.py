import numpy as np
import pytest

from src.data import cyclic_perms
from src.pipelines.eval import decode, format_report, per_task_table, reva_metrics
from tests.conftest import make_example


def test_reva_metrics_per_task_and_weighting():
    exs = [
        make_example(0, label=0, task="Temporal Grounding"),
        make_example(1, label=1, task="Temporal Grounding"),
        make_example(2, label=2, task="Change Detection"),
        make_example(3, label=-1, task="Change Detection"),
    ]  # unlabelled: ignored
    res = reva_metrics(["A", "C", "C", "A"], exs)
    assert res["n"] == 3
    assert res["accuracy"] == pytest.approx(2 / 3)
    assert res["per_task"] == {"TU_Temporal": 0.5, "FP_Change": 1.0}
    # test counts: Temporal 640, Change 500
    assert res["test_weighted_accuracy"] == pytest.approx(
        (640 * 0.5 + 500 * 1.0) / 1140
    )
    assert res["per_category"]["Temporal Understanding"] == 0.5
    assert "TU_Temporal" in format_report(res)
    assert per_task_table({"run": res})[2].startswith("| run | 66.7 |")


def test_reva_metrics_validates_lengths():
    with pytest.raises(ValueError):
        reva_metrics(["A"], [])
    assert reva_metrics(["A"], [make_example(label=-1)])["n"] == 0


def test_decode_modes_shape_and_identity():
    rng = np.random.default_rng(0)
    perms = cyclic_perms()
    display = rng.normal(size=(10, 4, 4))
    none = decode(display[:, :1], perms[:1], "none")
    assert none.shape == (10, 4)
    assert (none.argmax(1) == display[:, 0].argmax(1)).all()  # identity perm
    assert decode(display, perms, "cyclic").shape == (10, 4)
    assert decode(display, perms, "pride", pride_fraction=0.2).shape == (10, 4)
    with pytest.raises(ValueError):
        decode(display, perms, "bogus")
