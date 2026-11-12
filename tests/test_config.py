"""Config loader tests. These do not download data or construct a model."""

from pathlib import Path

import pytest
import yaml

from codeslm.config import (
    ConfigError,
    count_parameters,
    launch_blockers,
    load_data_config,
    load_model_config,
    load_training_config,
)
from codeslm.config.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_configs_load_and_match_the_design() -> None:
    base = load_model_config(ROOT / "configs/model/base.yaml")
    smoke = load_model_config(ROOT / "configs/model/smoke.yaml")
    training = load_training_config(ROOT / "configs/training/smoke_test.yaml", ROOT)
    assert base.parameter_count == 168_059_904
    assert base.model.head_dim == 64
    assert base.model.tie_embeddings is True
    assert base.model.bias is False
    assert base.tokenizer.add_prefix_space is False
    assert len(base.tokenizer.special_tokens) == 8
    assert smoke.parameter_count == 40_509_952
    assert training.model.name == "codeslm-smoke-40m"
    assert training.batch.seq_len == 512
    assert training.budget.max_tokens == 20_000
    assert training.budget.max_steps == 20
    assert training.batch.tokens_per_update == 512 * 2 * 2
    assert training.data.download_enabled is False
    assert training.data.quarantine_unknown is True
    assert training.data.mixture["python"] == 1.0
    assert training.data.mixture["javascript"] == 0.0
    assert training.data.train_repo == "codeparrot/codeparrot-clean-train"
    assert training.data.validation_repo == "codeparrot/codeparrot-clean-valid"
    assert [source.id for source in training.data.enabled_sources("train")] == ["codeparrot_clean_train"]
    assert launch_blockers(training) == (
        "data.download_enabled is false. Training stays off until a download "
        "is approved or a future synthetic-corpus flag is set.",
    )
    assert count_parameters(
        vocab_size=base.model.vocab_size,
        d_model=base.model.d_model,
        n_layers=base.model.n_layers,
        n_heads=base.model.n_heads,
        n_kv_heads=base.model.n_kv_heads,
        intermediate_size=base.model.intermediate_size,
        tie_embeddings=base.model.tie_embeddings,
        bias=base.model.bias,
    ) == base.parameter_count


def test_validate_configs_script_exits_clean() -> None:
    assert main([]) == 0


def test_rejects_head_count_that_does_not_divide_width(tmp_path: Path) -> None:
    path = _mutate_model(tmp_path, n_heads=12)
    with pytest.raises(ConfigError, match="not divisible"):
        load_model_config(path)


def test_rejects_parameter_count_that_disagrees_with_the_formula(tmp_path: Path) -> None:
    path = _mutate_model(tmp_path, parameter_count=1)
    with pytest.raises(ConfigError, match="architecture formula"):
        load_model_config(path)


def test_rejects_prefix_space_for_code(tmp_path: Path) -> None:
    raw = _model_raw()
    raw["tokenizer"]["add_prefix_space"] = True
    path = _dump(tmp_path / "model.yaml", raw)
    with pytest.raises(ConfigError, match="add_prefix_space"):
        load_model_config(path)


def test_rejects_unknown_model_key(tmp_path: Path) -> None:
    raw = _model_raw()
    raw["extra"] = True
    path = _dump(tmp_path / "model.yaml", raw)
    with pytest.raises(ConfigError, match="unknown keys"):
        load_model_config(path)


def test_rejects_mixture_that_does_not_sum_to_one(tmp_path: Path) -> None:
    raw = _data_raw()
    raw["mixture"]["python"] = 0.5
    path = _dump(tmp_path / "data.yaml", raw)
    with pytest.raises(ConfigError, match="mixture weights sum"):
        load_data_config(path)


def test_rejects_positive_weight_without_a_train_source(tmp_path: Path) -> None:
    raw = _data_raw()
    raw["mixture"]["python"] = 0.8
    raw["mixture"]["sql"] = 0.2
    path = _dump(tmp_path / "data.yaml", raw)
    with pytest.raises(ConfigError, match="mixture.sql"):
        load_data_config(path)


def test_rejects_enabling_a_blocked_source(tmp_path: Path) -> None:
    raw = _data_raw()
    raw["sources"][2]["enabled"] = True
    path = _dump(tmp_path / "data.yaml", raw)
    with pytest.raises(ConfigError, match="cannot be enabled"):
        load_data_config(path)


def test_rejects_sequence_longer_than_the_model(tmp_path: Path) -> None:
    raw = _training_raw()
    raw["batch"]["seq_len"] = 4096
    path = _dump(tmp_path / "train.yaml", raw)
    with pytest.raises(ConfigError, match="exceeds model.max_seq_len"):
        load_training_config(path, ROOT)


def test_rejects_config_path_that_escapes_the_project(tmp_path: Path) -> None:
    raw = _training_raw()
    raw["model_config"] = "../outside.yaml"
    path = _dump(tmp_path / "train.yaml", raw)
    with pytest.raises(ConfigError, match="relative path"):
        load_training_config(path, ROOT)


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_model_config(ROOT / "configs/model/missing.yaml")


def _model_raw() -> dict:
    return yaml.safe_load((ROOT / "configs/model/base.yaml").read_text(encoding="utf-8"))


def _data_raw() -> dict:
    return yaml.safe_load((ROOT / "configs/data/initial.yaml").read_text(encoding="utf-8"))


def _training_raw() -> dict:
    return yaml.safe_load((ROOT / "configs/training/smoke_test.yaml").read_text(encoding="utf-8"))


def _mutate_model(tmp_path: Path, **changes: object) -> Path:
    raw = _model_raw()
    for key, value in changes.items():
        if key in raw["model"]:
            raw["model"][key] = value
        else:
            raw[key] = value
    return _dump(tmp_path / "model.yaml", raw)


def _dump(path: Path, raw: dict) -> Path:
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path
