"""Command-line generation. Defaults to CPU and does not start training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from codeslm.inference.errors import InferenceError
from codeslm.inference.generate import generate_stream, load_model
from codeslm.tokenizer.bpe import CodeTokenizer
from codeslm.tokenizer.errors import TokenizerError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a code completion from a local checkpoint.")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", default=None, help="Completion prefix. Reads stdin when omitted.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    prompt = args.prompt if args.prompt is not None else sys.stdin.read()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available", file=sys.stderr)
        return 1
    try:
        device = torch.device(args.device)
        model = load_model(args.model_config, args.checkpoint, device)
        tokenizer = CodeTokenizer.load(args.tokenizer)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        for piece in generate_stream(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            generator=generator,
        ):
            print(piece, end="", flush=True)
        print()
    except (InferenceError, TokenizerError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0
