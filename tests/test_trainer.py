"""CPU training-step tests. No dataset download and no GPU run."""

import math
from pathlib import Path

import torch
from torch import nn

from codeslm.config.schema import ModelArch
from codeslm.models.model import ModelOutput
from codeslm.training import Batch, TrainSettings, Trainer, TrainingError, estimate_flops, learning_rate
from codeslm.training.cli import main
from codeslm.training.trainer import parameter_groups
from codeslm.models import CodeModel

ROOT = Path(__file__).resolve().parents[1]


def test_schedule_warms_up_and_decays_to_the_floor() -> None:
    peak = 1.0
    assert learning_rate(0, total_steps=10, warmup_steps=4, peak=peak, min_ratio=0.1) == 0.25
    assert learning_rate(3, total_steps=10, warmup_steps=4, peak=peak, min_ratio=0.1) == 1.0
    end = learning_rate(9, total_steps=10, warmup_steps=4, peak=peak, min_ratio=0.1)
    assert abs(end - 0.1) < 1e-12
    previous = 1.0
    for step in range(4, 10):
        current = learning_rate(step, total_steps=10, warmup_steps=4, peak=peak, min_ratio=0.1)
        assert current <= previous + 1e-12
        previous = current
    assert estimate_flops(10, 2, 4, 8, 3) == (6 * 10 + 12 * 2 * 4 * 8) * 3


def test_one_cpu_step_logs_and_updates_weights(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _tiny()
    before = model.embed.weight.detach().clone()
    settings = _settings()
    trainer = Trainer(
        model,
        settings,
        device=torch.device("cpu"),
        log_dir=tmp_path / "tb",
        checkpoint_dir=tmp_path / "ckpt",
    )
    result = trainer.fit(iter([_batch(1), _batch(2)]), max_steps=1, eval_batches=iter([_batch(3)]))
    assert result.steps == 1
    assert math.isfinite(result.last_loss)
    assert result.tokens == 2 * 2 * 7
    assert not torch.equal(model.embed.weight, before)
    assert any((tmp_path / "tb").glob("events.out.tfevents*"))
    decay_ids = {id(parameter) for group in trainer.optimizer.param_groups if group["weight_decay"] for parameter in group["params"]}
    plain_ids = {id(parameter) for group in trainer.optimizer.param_groups if not group["weight_decay"] for parameter in group["params"]}
    assert id(model.embed.weight) in decay_ids
    assert id(model.norm.weight) in plain_ids
    assert decay_ids.isdisjoint(plain_ids)
    counted = sum(parameter.numel() for group in parameter_groups(model, 0.1) for parameter in group["params"])
    assert counted == model.num_parameters()


def test_checkpoint_resume_restores_the_step_and_weights(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _tiny()
    trainer = Trainer(
        model,
        _settings(eval_every_steps=0),
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
    )
    trainer.fit(iter([_batch(1), _batch(2)]), max_steps=1)
    path = tmp_path / "step_00000001.pt"
    trainer.save_checkpoint(path)
    saved = model.embed.weight.detach().clone()
    trainer.training_step([_batch(4), _batch(5)])
    assert not torch.equal(model.embed.weight, saved)
    restored = _tiny()
    resumed = Trainer(restored, _settings(), device=torch.device("cpu"))
    resumed.load_checkpoint(path)
    assert resumed.completed_steps == 1
    assert resumed.tokens_seen == 2 * 2 * 7
    torch.testing.assert_close(restored.embed.weight, saved)
    trainer.save_checkpoint(tmp_path / "step_00000002.pt")
    assert not path.exists()
    assert (tmp_path / "step_00000002.pt").is_file()


def test_failed_steps_are_not_applied(tmp_path: Path) -> None:
    nan_model = _ScalarModel(math.nan)
    trainer = Trainer(nan_model, _settings(gradient_accumulation_steps=1), device=torch.device("cpu"))
    before = nan_model.weight.detach().clone()
    try:
        trainer.training_step([_batch(1)])
    except TrainingError as exc:
        assert "non-finite loss" in str(exc)
    else:
        raise AssertionError("expected a non-finite loss to stop the step")
    torch.testing.assert_close(nan_model.weight, before)

    inf_model = _InfGradModel()
    trainer = Trainer(inf_model, _settings(gradient_accumulation_steps=1), device=torch.device("cpu"))
    before = inf_model.weight.detach().clone()
    try:
        trainer.training_step([_batch(1)])
    except TrainingError as exc:
        assert "non-finite gradient" in str(exc)
    else:
        raise AssertionError("expected a non-finite gradient to stop the step")
    torch.testing.assert_close(inf_model.weight, before)

    oom_model = _OOMModel()
    trainer = Trainer(oom_model, _settings(gradient_accumulation_steps=1), device=torch.device("cpu"))
    try:
        trainer.training_step([_batch(1)])
    except TrainingError as exc:
        assert "out of memory" in str(exc)
    else:
        raise AssertionError("expected an out-of-memory failure to stop the step")

    trainer = Trainer(_tiny(), _settings(gradient_accumulation_steps=1, eval_every_steps=0), device=torch.device("cpu"))
    try:
        trainer.fit(iter([_batch(1)]), max_steps=2)
    except TrainingError as exc:
        assert "exhausted" in str(exc)
    else:
        raise AssertionError("expected the data iterator to end loudly")

    (tmp_path / "broken.pt").write_bytes(b"not a checkpoint")
    try:
        trainer.load_checkpoint(tmp_path / "broken.pt")
    except TrainingError as exc:
        assert "could not read checkpoint" in str(exc)
    else:
        raise AssertionError("expected a corrupt checkpoint to fail")


def test_interrupt_writes_a_checkpoint_and_the_launch_command_refuses(tmp_path: Path) -> None:
    trainer = Trainer(
        _tiny(),
        _settings(eval_every_steps=0),
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path,
    )

    def interrupted() -> None:
        raise KeyboardInterrupt
        yield _batch(1)  # pragma: no cover

    try:
        trainer.fit(interrupted(), max_steps=1)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected the interrupt to propagate")
    assert (tmp_path / "interrupt.pt").is_file()
    try:
        Trainer(_tiny(), _settings(early_stopping=True), device=torch.device("cpu"))
    except TrainingError as exc:
        assert "patience" in str(exc)
    else:
        raise AssertionError("expected early stopping without patience to be rejected")
    status = main(["--config", str(ROOT / "configs/training/smoke_test.yaml")])
    assert status == 2


def _settings(**overrides: object) -> TrainSettings:
    values: dict[str, object] = dict(
        learning_rate=1e-3,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        grad_clip_norm=1.0,
        warmup_steps=4,
        min_lr_ratio=0.1,
        total_steps=8,
        gradient_accumulation_steps=2,
        precision="fp32",
        seed=7,
        gradient_checkpointing=False,
        log_every_steps=1,
        eval_every_steps=1,
        keep_last=1,
        early_stopping=False,
    )
    values.update(overrides)
    return TrainSettings(**values)  # type: ignore[arg-type]


def _batch(seed: int) -> Batch:
    generator = torch.Generator().manual_seed(seed)
    token_ids = torch.randint(1, 32, (2, 8), generator=generator)
    labels = token_ids.clone()
    labels[:, -1] = -100
    return Batch(input_ids=token_ids, labels=labels)


def _tiny() -> CodeModel:
    return CodeModel(
        ModelArch(
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
    )


class _ScalarModel(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.value = value

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **_: object) -> ModelOutput:
        del labels
        loss = self.weight.sum() * self.value
        return ModelOutput(logits=self.weight.reshape(1, 1, 4).expand(input_ids.shape[0], input_ids.shape[1], 4), loss=loss)


class _InfBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, weight: torch.Tensor) -> torch.Tensor:
        del ctx
        return weight.sum()

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        del ctx, grad_output
        return torch.tensor([[math.inf, 0.0], [0.0, 0.0]])


class _InfGradModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **_: object) -> ModelOutput:
        del labels
        return ModelOutput(
            logits=self.weight.reshape(1, 1, 4).expand(input_ids.shape[0], input_ids.shape[1], 4),
            loss=_InfBackward.apply(self.weight),
        )


class _OOMModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **_: object) -> ModelOutput:
        del input_ids, labels
        raise torch.cuda.OutOfMemoryError("simulated")
