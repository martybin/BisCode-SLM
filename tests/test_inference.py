"""CPU generation tests. No training run and no download."""

from pathlib import Path

import torch
import yaml
from torch import nn

from codeslm.config.schema import ModelArch, TokenizerSpec, load_model_config
from codeslm.config.sizing import count_parameters
from codeslm.inference import InferenceError, format_prompt, generate, generate_stream, load_model
from codeslm.inference.cli import main
from codeslm.inference.sampling import apply_repetition_penalty, apply_top_k, apply_top_p
from codeslm.models import CodeModel
from codeslm.models.model import ModelOutput
from codeslm.tokenizer import train_tokenizer
from codeslm.training.trainer import CHECKPOINT_FORMAT

ROOT = Path(__file__).resolve().parents[1]


def test_sampling_filters() -> None:
    penalized = apply_repetition_penalty(torch.tensor([2.0, -2.0, 0.5]), [0, 1], 2.0)
    torch.testing.assert_close(penalized, torch.tensor([1.0, -4.0, 0.5]))
    limited = apply_top_k(torch.tensor([0.0, 1.0, 5.0, 2.0]), 1)
    assert torch.isfinite(limited[2])
    assert not torch.isfinite(limited[0])
    nucleus = apply_top_p(torch.tensor([0.0, 10.0, 0.0]), 0.5)
    assert torch.isfinite(nucleus[1])
    assert not torch.isfinite(nucleus[0])


def test_generation_stops_on_eos_and_respects_the_token_limit() -> None:
    tokenizer = _tokenizer()
    eos_model = _FixedModel(tokenizer.vocab_size, tokenizer.token_id("eos"))
    eos_result = generate(eos_model, tokenizer, "def ", max_new_tokens=5, temperature=0.0)
    assert eos_result.stopped_on_eos is True
    assert eos_result.token_ids == (tokenizer.token_id("eos"),)
    assert eos_result.text == ""

    letter = tokenizer.encode("a")[0]
    fixed = _FixedModel(tokenizer.vocab_size, letter)
    limited = generate(fixed, tokenizer, "def ", max_new_tokens=3, temperature=0.0)
    assert limited.stopped_on_eos is False
    assert limited.token_ids == (letter, letter, letter)
    streamed = "".join(generate_stream(fixed, tokenizer, "def ", max_new_tokens=3, temperature=0.0))
    assert streamed == limited.text


def test_greedy_generation_is_deterministic_and_checkpoint_reloads(tmp_path: Path) -> None:
    tokenizer = _tokenizer()
    model = _small_model()
    tokenizer.save(tmp_path / "tokenizer")
    checkpoint = tmp_path / "model.pt"
    torch.save({"format": CHECKPOINT_FORMAT, "model": model.state_dict(), "step": 0}, checkpoint)
    config_path = _write_config(tmp_path / "model.yaml")
    loaded = load_model(config_path, checkpoint, torch.device("cpu"))
    torch.testing.assert_close(loaded.embed.weight, model.embed.weight)
    first = generate(loaded, tokenizer, "def ", max_new_tokens=2, temperature=0.0)
    second = generate(loaded, tokenizer, "def ", max_new_tokens=2, temperature=0.0)
    assert first.token_ids == second.token_ids
    assert len(first.token_ids) <= 2
    assert format_prompt("def ") == "def "
    try:
        format_prompt("")
    except InferenceError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected an empty prompt to fail")
    try:
        generate(_FixedModel(32, 1), tokenizer, "def ", max_new_tokens=1)
    except InferenceError as exc:
        assert "vocabulary" in str(exc)
    else:
        raise AssertionError("expected a vocabulary mismatch to fail")
    (tmp_path / "broken.pt").write_bytes(b"not a checkpoint")
    try:
        load_model(config_path, tmp_path / "broken.pt", torch.device("cpu"))
    except InferenceError as exc:
        assert "could not read checkpoint" in str(exc)
    else:
        raise AssertionError("expected a corrupt checkpoint to fail")
    status = main(
        [
            "--model-config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tmp_path / "tokenizer"),
            "--prompt",
            "def ",
            "--max-new-tokens",
            "2",
            "--device",
            "cpu",
        ]
    )
    assert status == 0


def _tokenizer():
    shipped = load_model_config(ROOT / "configs/model/smoke.yaml").tokenizer
    spec = TokenizerSpec(
        type=shipped.type,
        vocab_size=300,
        add_prefix_space=False,
        special_tokens=dict(shipped.special_tokens),
    )
    return train_tokenizer(["def add(value):\n    return value\n", "a = 1\n"], spec, min_frequency=1)


def _small_model() -> CodeModel:
    return CodeModel(
        ModelArch(
            architecture="decoder_only",
            vocab_size=300,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=64,
            max_seq_len=32,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            tie_embeddings=True,
            bias=False,
            dropout=0.0,
            init_std=0.02,
        )
    )


def _write_config(path: Path) -> Path:
    count = count_parameters(
        vocab_size=300,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        tie_embeddings=True,
        bias=False,
    )
    shipped = load_model_config(ROOT / "configs/model/smoke.yaml")
    document = {
        "schema_version": 1,
        "name": "codeslm-infer-test",
        "model": {
            "architecture": "decoder_only",
            "vocab_size": 300,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "n_kv_heads": 2,
            "intermediate_size": 64,
            "max_seq_len": 32,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1.0e-6,
            "tie_embeddings": True,
            "bias": False,
            "dropout": 0.0,
            "init_std": 0.02,
        },
        "tokenizer": {
            "type": shipped.tokenizer.type,
            "vocab_size": 300,
            "add_prefix_space": False,
            "special_tokens": dict(shipped.tokenizer.special_tokens),
        },
        "parameter_count": count,
    }
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


class _FixedModel(nn.Module):
    def __init__(self, vocab_size: int, token_id: int) -> None:
        super().__init__()
        self.arch = ModelArch(
            architecture="decoder_only",
            vocab_size=vocab_size,
            d_model=32,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=64,
            max_seq_len=32,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            tie_embeddings=True,
            bias=False,
            dropout=0.0,
            init_std=0.02,
        )
        self.token_id = token_id
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None, **_: object) -> ModelOutput:
        del position_ids
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], self.arch.vocab_size), -20.0)
        logits[..., self.token_id] = 20.0
        return ModelOutput(logits=logits)
