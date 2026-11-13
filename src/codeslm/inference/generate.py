"""Local code generation.

Generation needs a model config, a checkpoint written by the trainer, and a
tokenizer directory. It does not download weights or data.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from codeslm.config.schema import ModelConfig, load_model_config
from codeslm.inference.errors import InferenceError
from codeslm.inference.sampling import sample_token
from codeslm.models.model import CodeModel, ModelOutput
from codeslm.tokenizer.bpe import CodeTokenizer
from codeslm.training.trainer import CHECKPOINT_FORMAT

COMPLETION = "completion"


@dataclass(frozen=True)
class Generation:
    text: str
    token_ids: tuple[int, ...]
    stopped_on_eos: bool


def format_prompt(prompt: str, *, mode: str = COMPLETION) -> str:
    """Return the raw completion prefix.

    Instruction chat templates belong to a later fine-tuning stage. The base
    model sees the prompt text as a code prefix.
    """
    if mode != COMPLETION:
        raise InferenceError(f"prompt mode {mode!r} is not available on the base model")
    if prompt == "":
        raise InferenceError("prompt is empty")
    return prompt


def load_model(model_config: ModelConfig | Path | str, checkpoint: Path | str, device: torch.device) -> CodeModel:
    """Build the architecture from YAML and load the checkpoint weights."""
    config = model_config if isinstance(model_config, ModelConfig) else load_model_config(model_config)
    path = Path(checkpoint)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise InferenceError(f"could not read checkpoint {path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT or "model" not in payload:
        raise InferenceError(f"checkpoint format is not {CHECKPOINT_FORMAT}")
    model = CodeModel.from_config(config)
    try:
        model.load_state_dict(payload["model"])
    except (RuntimeError, ValueError) as exc:
        raise InferenceError(f"checkpoint does not match {config.name}") from exc
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
) -> Generation:
    """Generate a completion and stop on EOS or the token limit."""
    new_ids = list(
        _new_token_ids(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generator=generator,
        )
    )
    eos_id = tokenizer.token_id("eos")
    stopped = bool(new_ids) and new_ids[-1] == eos_id
    visible = new_ids[:-1] if stopped else new_ids
    text = tokenizer.decode(visible, skip_special_tokens=True) if visible else ""
    return Generation(text=text, token_ids=tuple(new_ids), stopped_on_eos=stopped)


def generate_stream(
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
) -> Iterator[str]:
    """Yield newly decoded text as tokens arrive. EOS is not included."""
    eos_id = tokenizer.token_id("eos")
    produced: list[int] = []
    shown = ""
    for token_id in _new_token_ids(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        generator=generator,
    ):
        if token_id == eos_id:
            break
        produced.append(token_id)
        decoded = tokenizer.decode(produced, skip_special_tokens=True)
        if decoded.startswith(shown):
            delta = decoded[len(shown) :]
            shown = decoded
            if delta:
                yield delta


def _new_token_ids(
    model: torch.nn.Module,
    tokenizer: CodeTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    generator: torch.Generator | None,
) -> Iterator[int]:
    if max_new_tokens < 1:
        raise InferenceError("max_new_tokens must be positive")
    _check_vocab(model, tokenizer)
    formatted = format_prompt(prompt)
    ids = tokenizer.encode(formatted)
    if not ids:
        raise InferenceError("prompt encoded to zero tokens")
    eos_id = tokenizer.token_id("eos")
    device = _module_device(model)
    context = _max_seq_len(model)
    model.eval()
    for _ in range(max_new_tokens):
        window = ids[-context:]
        start = len(ids) - len(window)
        position_ids = torch.arange(start, start + len(window), device=device).unsqueeze(0)
        input_ids = torch.tensor([window], dtype=torch.long, device=device)
        output: ModelOutput = model(input_ids, position_ids=position_ids)
        next_id = sample_token(
            output.logits[0, -1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_ids=ids,
            generator=generator,
        )
        ids.append(next_id)
        yield next_id
        if next_id == eos_id:
            return


def _check_vocab(model: torch.nn.Module, tokenizer: CodeTokenizer) -> None:
    vocab = _vocab_size(model)
    if vocab != tokenizer.vocab_size:
        raise InferenceError(f"model vocabulary {vocab} does not match tokenizer vocabulary {tokenizer.vocab_size}")


def _vocab_size(model: torch.nn.Module) -> int:
    arch = getattr(model, "arch", None)
    vocab = getattr(arch, "vocab_size", None)
    if not isinstance(vocab, int):
        raise InferenceError("model has no vocabulary size")
    return vocab


def _max_seq_len(model: torch.nn.Module) -> int:
    arch = getattr(model, "arch", None)
    length = getattr(arch, "max_seq_len", None)
    if not isinstance(length, int) or length < 1:
        raise InferenceError("model has no context length")
    return length


def _module_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
