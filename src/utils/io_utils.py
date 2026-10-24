import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import yaml


def load_env(env_file: Union[str, Path] = ".env") -> None:
    """Load `KEY=VALUE` lines from `env_file` into os.environ (does not
    override already-set variables). Used by the scripts that need HF/W&B
    tokens, mirroring the reference codebase's helper.
    """
    path = Path(env_file)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Only the parameters with `requires_grad` (the LoRA adapter, for a
    frozen VLM), moved to CPU. A full 9B state dict would be ~18 GB."""
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    path: Union[str, Path],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save {"model", "optimizer", "step", "extra"} — the schema
    src.pipelines.eval / predict expect to load. "model" holds the trainable
    tensors only; `optimizer=None` skips the (larger) optimizer state.
    `extra` carries run metadata (model config, epoch, best metric, wandb id).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "extra": extra or {},
    }
    torch.save(checkpoint, path)


def load_trainable(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """Load a `trainable_state_dict` into a freshly built model. Every saved
    tensor must find a parameter, so a LoRA-config mismatch fails loudly."""
    missing = model.load_state_dict(state, strict=False)
    unexpected = list(missing.unexpected_keys)
    if unexpected:
        raise KeyError(
            f"{len(unexpected)} checkpoint tensors match no parameter, e.g. {unexpected[:3]}"
        )


def read_adapter(path: Union[str, Path]) -> dict:
    """Read a checkpoint as {"model", "extra"}: either our `.pt` file or a
    PEFT folder (adapter_model.safetensors + adapter_config.json, as released
    on HF) with the ModelConfig in `peekaboo_config.json`. PEFT keys are mapped
    back to the `trainable_state_dict` names."""
    path = Path(path)
    if not path.is_dir():
        return torch.load(path, map_location="cpu", weights_only=False)
    from safetensors.torch import load_file

    state = {
        "backbone." + k.replace(".weight", ".default.weight"): v
        for k, v in load_file(path / "adapter_model.safetensors").items()
    }
    with open(path / "peekaboo_config.json") as f:
        cfg = json.load(f)
    return {"model": state, "extra": {"cfg": cfg}}


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    load_trainable(model, checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def save_json(data: Any, filename: Union[str, Path], save_pretty: bool = True) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=2))
        else:
            json.dump(data, f)


def load_json(filename: Union[str, Path]) -> Any:
    with open(filename, "r") as f:
        return json.load(f)


def read_jsonl(filename: Union[str, Path]) -> List[Any]:
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(
    rows: Iterable[Any], filename: Union[str, Path], append: bool = False
) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "a" if append else "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data if data is not None else {}


def write_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )
