"""CPU tests for the decoder-only model. No training run."""

import math
from pathlib import Path

import torch

from codeslm.config.schema import ModelArch, load_model_config
from codeslm.config.sizing import count_parameters
from codeslm.data.packing import causal_block_mask
from codeslm.models import CodeModel
from codeslm.models.mask import build_attention_mask
from codeslm.models.norm import RMSNorm
from codeslm.models.rope import apply_rope, rope_angles

ROOT = Path(__file__).resolve().parents[1]


def test_rmsnorm_matches_the_definition() -> None:
    norm = RMSNorm(2, eps=1e-6)
    hidden = torch.tensor([[3.0, 4.0]])
    expected = hidden / math.sqrt((9.0 + 16.0) / 2 + 1e-6)
    torch.testing.assert_close(norm(hidden), expected)


def test_rope_is_identity_at_position_zero_and_depends_on_position() -> None:
    states = torch.randn(1, 2, 1, 8)
    cos, sin = rope_angles(torch.tensor([[0]]), head_dim=8, theta=10_000.0)
    torch.testing.assert_close(apply_rope(states, cos, sin), states)
    moved = apply_rope(states, *rope_angles(torch.tensor([[3]]), head_dim=8, theta=10_000.0))
    assert not torch.allclose(moved, states)


def test_document_mask_matches_the_packer_and_blocks_pads() -> None:
    blocks = [0, 0, 1, -1]
    expected = torch.tensor(causal_block_mask(blocks))
    built = build_attention_mask(sequence_length=4, block_ids=torch.tensor([blocks]))
    assert built is not None
    # The packer mask leaves a pad row fully blocked. The model mask keeps the
    # pad diagonal so a softmax over that row stays finite.
    assert expected[3, 3].item() is False
    expected[3, 3] = True
    torch.testing.assert_close(built[0, 0], expected)
    assert built[0, 0, 2, 0].item() is False
    assert built[0, 0, 3, 2].item() is False
    assert build_attention_mask(sequence_length=4) is None


def test_parameter_counts_match_the_formula_and_the_configs() -> None:
    base = CodeModel.from_config(load_model_config(ROOT / "configs/model/base.yaml"))
    smoke = CodeModel.from_config(load_model_config(ROOT / "configs/model/smoke.yaml"))
    assert base.num_parameters() == 168_059_904
    assert smoke.num_parameters() == 40_509_952
    assert base.lm_head.weight is base.embed.weight
    assert smoke.lm_head.weight is smoke.embed.weight
    for model in (base, smoke):
        arch = model.arch
        assert model.num_parameters() == count_parameters(
            vocab_size=arch.vocab_size,
            d_model=arch.d_model,
            n_layers=arch.n_layers,
            n_heads=arch.n_heads,
            n_kv_heads=arch.n_kv_heads,
            intermediate_size=arch.intermediate_size,
            tie_embeddings=True,
            bias=False,
        )
    attention = base.layers[0].attention
    assert attention.q_proj.weight.shape == (1024, 1024)
    assert attention.k_proj.weight.shape == (256, 1024)
    assert attention.v_proj.weight.shape == (256, 1024)
    assert attention.out_proj.weight.shape == (1024, 1024)
    assert base.layers[0].mlp.gate_proj.weight.shape == (2816, 1024)
    assert base.layers[0].mlp.down_proj.weight.shape == (1024, 2816)
    untied = _tiny(tie_embeddings=False, bias=True)
    assert untied.num_parameters() == count_parameters(
        vocab_size=untied.arch.vocab_size,
        d_model=untied.arch.d_model,
        n_layers=untied.arch.n_layers,
        n_heads=untied.arch.n_heads,
        n_kv_heads=untied.arch.n_kv_heads,
        intermediate_size=untied.arch.intermediate_size,
        tie_embeddings=False,
        bias=True,
    )
    assert untied.lm_head.weight is not untied.embed.weight


def test_initialization_scales_residual_projections() -> None:
    torch.manual_seed(0)
    model = _tiny(d_model=128, n_heads=4, n_kv_heads=2, intermediate_size=256, n_layers=2)
    expected_residual = 0.02 / math.sqrt(2 * 2)
    residual = model.layers[0].attention.out_proj.weight.detach()
    query = model.layers[0].attention.q_proj.weight.detach()
    assert abs(residual.std().item() - expected_residual) / expected_residual < 0.1
    assert abs(query.std().item() - 0.02) / 0.02 < 0.1
    assert torch.equal(model.norm.weight, torch.ones_like(model.norm.weight))
    assert model.layers[0].attention.q_proj.bias is None


def test_forward_backward_and_causal_isolation() -> None:
    torch.manual_seed(1)
    model = _tiny()
    model.eval()
    original = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 9, 8]])
    first = model(original).logits
    second = model(changed_future).logits
    torch.testing.assert_close(first[:, :2], second[:, :2])
    assert not torch.allclose(first[:, 2], second[:, 2])

    blocks = torch.tensor([[0, 0, 1]])
    positions = torch.tensor([[0, 1, 0]])
    left = model(torch.tensor([[4, 5, 6]]), block_ids=blocks, position_ids=positions).logits
    right = model(torch.tensor([[4, 5, 7]]), block_ids=blocks, position_ids=positions).logits
    torch.testing.assert_close(left[:, :2], right[:, :2])
    changed_first_document = model(torch.tensor([[1, 2, 6]]), block_ids=blocks, position_ids=positions).logits
    torch.testing.assert_close(changed_first_document[:, 2], left[:, 2])

    model.train()
    labels = torch.tensor([[-100, 3, 4, -100]])
    output = model(original, labels=labels)
    assert output.logits.shape == (1, 4, model.arch.vocab_size)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert model.embed.weight.grad is not None
    assert torch.isfinite(model.embed.weight.grad).all()


def test_padding_loss_and_checkpointed_backward_stay_finite() -> None:
    torch.manual_seed(2)
    model = _tiny()
    model.gradient_checkpointing = True
    input_ids = torch.tensor([[4, 5, 0]])
    labels = torch.tensor([[5, -100, -100]])
    mask = torch.tensor([[1, 1, 0]])
    output = model(input_ids, labels=labels, attention_mask=mask)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert torch.isfinite(model.embed.weight.grad).all()


def test_sequence_longer_than_the_config_is_rejected() -> None:
    model = _tiny(max_seq_len=4)
    try:
        model(torch.zeros(1, 5, dtype=torch.long))
    except ValueError as exc:
        assert "max_seq_len" in str(exc)
    else:
        raise AssertionError("expected a context-length error")


def _tiny(**overrides: object) -> CodeModel:
    values: dict[str, object] = dict(
        architecture="decoder_only",
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        max_seq_len=16,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        tie_embeddings=True,
        bias=False,
        dropout=0.0,
        init_std=0.02,
    )
    values.update(overrides)
    return CodeModel(ModelArch(**values))  # type: ignore[arg-type]
