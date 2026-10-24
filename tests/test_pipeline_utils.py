import pytest

from src.pipelines.config import load_training_config
from src.pipelines.train import total_optimizer_steps
from src.utils.io_utils import write_yaml
from scripts.submission.make_submission import render, validate


def test_load_training_config_yaml_and_overrides(tmp_path):
    write_yaml(
        {"epochs": 3, "arch": {"backbone": "Qwen/Qwen3.5-4B"}, "tta": "pride"},
        tmp_path / "c.yaml",
    )
    cfg = load_training_config(tmp_path / "c.yaml", lr=5e-5, epochs=None)
    assert cfg.epochs == 3 and cfg.lr == 5e-5 and cfg.tta == "pride"
    assert cfg.eval_tta == "none"
    assert cfg.arch["backbone"] == "Qwen/Qwen3.5-4B"
    write_yaml({"bogus": 1}, tmp_path / "bad.yaml")
    with pytest.raises(ValueError, match="unknown"):
        load_training_config(tmp_path / "bad.yaml")
    with pytest.raises(ValueError, match="tta"):
        load_training_config(None, tta="sometimes")
    with pytest.raises(ValueError, match="eval_tta"):
        load_training_config(None, eval_tta="sometimes")


def test_repo_configs_load():
    from pathlib import Path

    from src.config import ModelConfig

    for path in [
        Path("configs/train.yaml"),
    ]:
        cfg = load_training_config(path)
        ModelConfig(**(cfg.arch or {}))


def test_total_optimizer_steps():
    cfg = load_training_config(None, batch_size=2, accum_steps=4, epochs=3)
    # 17 examples -> 9 batches -> ceil(9 / 4) = 3 steps per epoch
    assert total_optimizer_steps(17, cfg) == 9


def test_submission_validation():
    ids = ["test_000001", "test_000002"]
    validate({"test_000001": "A", "test_000002": "D"}, ids)
    with pytest.raises(ValueError, match="missing"):
        validate({"test_000001": "A"}, ids)
    with pytest.raises(ValueError, match="unknown"):
        validate({"test_000001": "A", "test_000002": "B", "x": "C"}, ids)
    with pytest.raises(ValueError, match="not in A-D"):
        validate({"test_000001": "A", "test_000002": "E"}, ids)
    test = {"metadata": {}, "QA": [{"qa_id": i, "correct_answer": ""} for i in ids]}
    pred = {"test_000001": "A", "test_000002": "D"}
    assert render(pred, test, "dict") == pred
    assert render(pred, test, "list")[1] == {"qa_id": "test_000002", "answer": "D"}
    assert render(pred, test, "testjson")["QA"][0]["correct_answer"] == "A"
