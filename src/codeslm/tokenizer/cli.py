"""Train a tokenizer from local text files.

The command reads UTF-8 files on disk. It does not contact Hugging Face and it
does not read the validation split. A vocabulary that would be mostly reserved
padding is not written unless ``--allow-short-corpus`` is set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from codeslm.config.schema import TokenizerSpec, find_project_root, load_model_config
from codeslm.tokenizer.bpe import train_tokenizer
from codeslm.tokenizer.errors import TokenizerError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer from local files.")
    parser.add_argument("--config", type=Path, required=True, help="Model YAML whose special tokens are used.")
    parser.add_argument("--corpus", type=Path, required=True, help="A UTF-8 file, or a directory of UTF-8 files.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for tokenizer.json and metadata.")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override the config vocabulary size.")
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--allow-short-corpus",
        action="store_true",
        help="Write the tokenizer even when more than half of the ids are reserved padding.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = find_project_root(args.config if args.config.is_absolute() else Path.cwd())
        config_path = args.config if args.config.is_absolute() else root / args.config
        model = load_model_config(config_path)
        vocab_size = model.tokenizer.vocab_size if args.vocab_size is None else args.vocab_size
        spec = TokenizerSpec(
            type=model.tokenizer.type,
            vocab_size=vocab_size,
            add_prefix_space=model.tokenizer.add_prefix_space,
            special_tokens=dict(model.tokenizer.special_tokens),
        )
        documents = list(read_corpus(args.corpus))
        tokenizer = train_tokenizer(documents, spec, min_frequency=args.min_frequency, show_progress=False)
    except (TokenizerError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"trained {tokenizer.n_trained:,} tokens, reserved {tokenizer.n_reserved:,}, "
        f"vocab {tokenizer.vocab_size:,}"
    )
    if tokenizer.n_reserved * 2 > tokenizer.vocab_size and not args.allow_short_corpus:
        print(
            "Refusing to save: more than half of the vocabulary would be reserved padding. "
            "Train on a larger local corpus, or pass --allow-short-corpus for a smoke tokenizer.",
            file=sys.stderr,
        )
        return 2
    try:
        tokenizer.save(args.output, overwrite=args.overwrite)
    except TokenizerError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"saved {args.output}")
    return 0


def read_corpus(path: Path) -> list[str]:
    """Load one document per file. A file path is a single document."""
    if not path.exists():
        raise TokenizerError(f"corpus path does not exist: {path}")
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise TokenizerError(f"corpus directory is empty: {path}")
    documents: list[str] = []
    for file in files:
        try:
            documents.append(file.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            raise TokenizerError(f"{file} is not UTF-8 text") from exc
    return documents
