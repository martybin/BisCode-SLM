"""AdamW training loop for the code model.

One call applies one optimizer update. A non-finite loss or gradient raises
``TrainingError`` and does not call ``optimizer.step``. The loop also stops
when the batch iterator ends early instead of repeating or skipping a step.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from codeslm.training.errors import TrainingError
from codeslm.training.schedule import estimate_flops, learning_rate

CHECKPOINT_FORMAT = "codeslm-checkpoint-v1"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class TrainSettings:
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    grad_clip_norm: float
    warmup_steps: int
    min_lr_ratio: float
    total_steps: int
    gradient_accumulation_steps: int = 1
    precision: str = "fp32"
    seed: int = 0
    gradient_checkpointing: bool = False
    log_every_steps: int = 1
    eval_every_steps: int = 0
    keep_last: int = 1
    early_stopping: bool = False


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    block_ids: torch.Tensor | None = None

    def to(self, device: torch.device) -> Batch:
        def move(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.to(device)

        return Batch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            attention_mask=move(self.attention_mask),
            position_ids=move(self.position_ids),
            block_ids=move(self.block_ids),
        )


@dataclass(frozen=True)
class StepResult:
    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    tokens: int
    tokens_per_second: float
    flops: int


@dataclass(frozen=True)
class FitResult:
    steps: int
    tokens: int
    last_loss: float


class Trainer:
    """Owns the optimizer, the scaler, and the step counter for one model."""

    def __init__(
        self,
        model: nn.Module,
        settings: TrainSettings,
        *,
        device: torch.device | None = None,
        log_dir: Path | None = None,
        checkpoint_dir: Path | None = None,
    ) -> None:
        if settings.gradient_accumulation_steps < 1:
            raise TrainingError("gradient_accumulation_steps must be positive")
        if settings.early_stopping:
            raise TrainingError("early stopping is enabled, but the config has no patience value")
        self.model = model
        self.settings = settings
        self.device = device or torch.device("cpu")
        self.checkpoint_dir = checkpoint_dir
        self.completed_steps = 0
        self.tokens_seen = 0
        self.autocast_dtype = _autocast_dtype(settings.precision, self.device)
        set_seed(settings.seed, self.device)
        if hasattr(model, "gradient_checkpointing"):
            model.gradient_checkpointing = settings.gradient_checkpointing
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            parameter_groups(self.model, settings.weight_decay),
            lr=settings.learning_rate,
            betas=settings.betas,
            eps=settings.eps,
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.autocast_dtype is torch.float16)
        self._writer = SummaryWriter(log_dir=str(log_dir)) if log_dir is not None else None
        if self._writer is not None:
            parameters = getattr(model, "num_parameters", None)
            if callable(parameters):
                self._writer.add_scalar("model/parameters", parameters(), 0)

    def fit(
        self,
        batches: Iterator[Batch],
        *,
        max_steps: int,
        eval_batches: Iterator[Batch] | None = None,
    ) -> FitResult:
        """Run optimizer updates until ``max_steps`` or the token budget's caller stops."""
        if max_steps < 1:
            raise TrainingError("max_steps must be positive")
        last_loss = 0.0
        try:
            for _ in range(max_steps):
                micros = self._next_batches(batches, self.settings.gradient_accumulation_steps, "training")
                result = self.training_step(micros)
                last_loss = result.loss
                if (
                    self.settings.eval_every_steps > 0
                    and self.completed_steps % self.settings.eval_every_steps == 0
                ):
                    if eval_batches is None:
                        raise TrainingError("evaluation is due but no eval batches were provided")
                    observed = self.evaluate(
                        self._next_batches(eval_batches, 1, "evaluation")
                    )
                    if self._writer is not None:
                        self._writer.add_scalar("eval/loss", observed, self.completed_steps)
        except KeyboardInterrupt:
            if self.checkpoint_dir is not None:
                self.save_checkpoint(self.checkpoint_dir / "interrupt.pt")
            raise
        finally:
            if self._writer is not None:
                self._writer.flush()
                self._writer.close()
                self._writer = None
        return FitResult(steps=self.completed_steps, tokens=self.tokens_seen, last_loss=last_loss)

    def training_step(self, batches: Sequence[Batch]) -> StepResult:
        expected = self.settings.gradient_accumulation_steps
        if len(batches) != expected:
            raise TrainingError(f"expected {expected} micro-batches, received {len(batches)}")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        step = self.completed_steps
        current_lr = learning_rate(
            step,
            total_steps=self.settings.total_steps,
            warmup_steps=self.settings.warmup_steps,
            peak=self.settings.learning_rate,
            min_ratio=self.settings.min_lr_ratio,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = current_lr
        losses: list[float] = []
        tokens = 0
        started = time.perf_counter()
        try:
            for batch in batches:
                batch = batch.to(self.device)
                batch_tokens = _token_count(batch)
                if batch_tokens < 1:
                    raise TrainingError("batch has no training tokens")
                tokens += batch_tokens
                with self._autocast():
                    output = self.model(
                        batch.input_ids,
                        labels=batch.labels,
                        attention_mask=batch.attention_mask,
                        position_ids=batch.position_ids,
                        block_ids=batch.block_ids,
                    )
                    loss = output.loss
                if loss is None or not torch.isfinite(loss):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise TrainingError("non-finite loss; the optimizer step was not applied")
                (loss / expected).backward()
                losses.append(float(loss.detach().cpu()))
        except torch.cuda.OutOfMemoryError as exc:
            self.optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise TrainingError("CUDA out of memory; the optimizer step was not applied") from exc
        grad_norm = self._apply_gradients()
        elapsed = max(time.perf_counter() - started, 1e-9)
        self.completed_steps += 1
        self.tokens_seen += tokens
        flops = _model_flops(self.model, batches[0].input_ids.shape[1], tokens)
        result = StepResult(
            step=self.completed_steps,
            loss=sum(losses) / len(losses),
            learning_rate=current_lr,
            grad_norm=grad_norm,
            tokens=tokens,
            tokens_per_second=tokens / elapsed,
            flops=flops,
        )
        self._log(result)
        return result

    @torch.no_grad()
    def evaluate(self, batches: Sequence[Batch]) -> float:
        if not batches:
            raise TrainingError("evaluation received no batches")
        was_training = self.model.training
        self.model.eval()
        losses: list[float] = []
        try:
            for batch in batches:
                batch = batch.to(self.device)
                with self._autocast():
                    output = self.model(
                        batch.input_ids,
                        labels=batch.labels,
                        attention_mask=batch.attention_mask,
                        position_ids=batch.position_ids,
                        block_ids=batch.block_ids,
                    )
                loss = output.loss
                if loss is None or not torch.isfinite(loss):
                    raise TrainingError("non-finite evaluation loss")
                losses.append(float(loss.detach().cpu()))
        finally:
            self.model.train(was_training)
        return sum(losses) / len(losses)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CHECKPOINT_FORMAT,
            "step": self.completed_steps,
            "tokens": self.tokens_seen,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "settings": _settings_dict(self.settings),
        }
        torch.save(payload, path)
        if self.checkpoint_dir is not None and self.settings.keep_last > 0:
            _retain_latest(self.checkpoint_dir, self.settings.keep_last)

    def load_checkpoint(self, path: Path) -> None:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise TrainingError(f"could not read checkpoint {path}") from exc
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise TrainingError(f"checkpoint format is not {CHECKPOINT_FORMAT}")
        try:
            self.model.load_state_dict(payload["model"])
            self.optimizer.load_state_dict(payload["optimizer"])
            self.scaler.load_state_dict(payload["scaler"])
            torch.set_rng_state(payload["torch_rng"])
            random.setstate(payload["python_rng"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise TrainingError(f"checkpoint {path} is incomplete or does not match the model") from exc
        self.completed_steps = int(payload["step"])
        self.tokens_seen = int(payload["tokens"])

    def _apply_gradients(self) -> float:
        parameters = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
        grad_norm = clip_grad_norm_(parameters, self.settings.grad_clip_norm)
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise TrainingError("non-finite gradient; the optimizer step was not applied")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm)

    def _log(self, result: StepResult) -> None:
        if self._writer is None or result.step % self.settings.log_every_steps != 0:
            return
        self._writer.add_scalar("train/loss", result.loss, result.step)
        self._writer.add_scalar("train/learning_rate", result.learning_rate, result.step)
        self._writer.add_scalar("train/grad_norm", result.grad_norm, result.step)
        self._writer.add_scalar("train/tokens", self.tokens_seen, result.step)
        self._writer.add_scalar("train/tokens_per_second", result.tokens_per_second, result.step)
        self._writer.add_scalar("train/flops", result.flops, result.step)
        if self.device.type == "cuda":
            self._writer.add_scalar("train/gpu_memory_bytes", torch.cuda.max_memory_allocated(self.device), result.step)

    @contextmanager
    def _autocast(self) -> Any:
        if self.autocast_dtype is None:
            with nullcontext():
                yield
            return
        with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
            yield

    def _next_batches(self, batches: Iterator[Batch], count: int, split: str) -> list[Batch]:
        collected: list[Batch] = []
        for _ in range(count):
            try:
                collected.append(next(batches))
            except StopIteration as exc:
                raise TrainingError(f"{split} data exhausted at step {self.completed_steps}") from exc
        return collected


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Decay matrices. Do not decay norms, biases, or a second view of a tied weight."""
    decay: list[nn.Parameter] = []
    plain: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.ndim >= 2:
            decay.append(parameter)
        else:
            plain.append(parameter)
    if not decay and not plain:
        raise TrainingError("model has no trainable parameters")
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if plain:
        groups.append({"params": plain, "weight_decay": 0.0})
    return groups


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    if precision == "fp32" or device.type == "cpu":
        if precision in {"bf16", "fp16"} and device.type == "cpu":
            raise TrainingError(f"{precision} autocast is not used on CPU; choose fp32 or auto")
        return None
    if precision == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return None
    if precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise TrainingError("bf16 was requested but this device does not support it")
        return torch.bfloat16
    if precision == "fp16":
        if device.type != "cuda":
            raise TrainingError("fp16 autocast requires CUDA")
        return torch.float16
    raise TrainingError(f"unknown precision {precision!r}")


def _token_count(batch: Batch) -> int:
    return int(batch.labels.ne(IGNORE_INDEX).sum().item())


def _model_flops(model: nn.Module, seq_len: int, tokens: int) -> int:
    arch = getattr(model, "arch", None)
    counter = getattr(model, "num_parameters", None)
    if arch is None or not callable(counter):
        return 0
    return estimate_flops(counter(), arch.n_layers, arch.d_model, seq_len, tokens)


def _settings_dict(settings: TrainSettings) -> dict[str, Any]:
    return {
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "betas": list(settings.betas),
        "eps": settings.eps,
        "grad_clip_norm": settings.grad_clip_norm,
        "warmup_steps": settings.warmup_steps,
        "min_lr_ratio": settings.min_lr_ratio,
        "total_steps": settings.total_steps,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "precision": settings.precision,
        "seed": settings.seed,
        "gradient_checkpointing": settings.gradient_checkpointing,
    }


def _retain_latest(directory: Path, keep_last: int) -> None:
    checkpoints = sorted(directory.glob("step_*.pt"))
    for stale in checkpoints[:-keep_last]:
        stale.unlink()
