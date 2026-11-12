"""Byte-level BPE for source code.

The vocabulary size includes the eight special tokens from the model config.
Every byte is in the initial alphabet, so text that never appeared in the
training corpus still decodes back to the same Unicode string. No normalizer
is applied: Unicode normalization would change source code.

This module does not download data. Callers pass the training strings.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from codeslm.config.schema import SPECIAL_TOKEN_KEYS, TokenizerSpec
from codeslm.tokenizer.errors import TokenizerError

TOKENIZER_FILE = "tokenizer.json"
META_FILE = "tokenizer_meta.json"
BYTE_ALPHABET_SIZE = 256
META_VERSION = 1


@dataclass(frozen=True)
class TokenStats:
    """Token-length summary for a set of documents."""

    documents: int
    characters: int
    tokens: int
    min_tokens: int
    max_tokens: int
    mean_tokens: float
    median_tokens: float
    characters_per_token: float


class CodeTokenizer:
    """A trained byte-level BPE tokenizer plus the special-token contract."""

    def __init__(self, backend: Tokenizer, spec: TokenizerSpec, n_trained: int) -> None:
        if spec.vocab_size != backend.get_vocab_size():
            raise TokenizerError(
                f"tokenizer has {backend.get_vocab_size()} tokens, spec asks for {spec.vocab_size}"
            )
        if n_trained < 1 or n_trained > spec.vocab_size:
            raise TokenizerError(f"n_trained {n_trained} is outside 1..{spec.vocab_size}")
        self._backend = backend
        self.spec = spec
        self.n_trained = n_trained
        self.n_reserved = spec.vocab_size - n_trained
        self._check_special_ids()

    @property
    def vocab_size(self) -> int:
        return self.spec.vocab_size

    def token_id(self, name_or_token: str) -> int:
        """Resolve a special-token key (``pad``) or a literal token string."""
        token = self.spec.special_tokens.get(name_or_token, name_or_token)
        if not isinstance(token, str):
            raise TokenizerError(f"token name {name_or_token!r} is not a string")
        token_id = self._backend.token_to_id(token)
        if token_id is None:
            raise TokenizerError(f"token {token!r} is not in the vocabulary")
        return token_id

    def token_for_id(self, token_id: int) -> str:
        if token_id < 0 or token_id >= self.vocab_size:
            raise TokenizerError(f"token id {token_id} is outside 0..{self.vocab_size - 1}")
        token = self._backend.id_to_token(token_id)
        if token is None:
            raise TokenizerError(f"token id {token_id} has no string")
        return token

    def vocab_by_id(self) -> list[str]:
        """Vocabulary strings in id order, including reserved padding tokens."""
        table = [""] * self.vocab_size
        for token, token_id in self._backend.get_vocab().items():
            table[token_id] = token
        missing = [index for index, token in enumerate(table) if token == ""]
        if missing:
            raise TokenizerError(f"vocabulary is missing ids {missing[:8]}")
        return table

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode one document. Does not truncate and does not add specials unless asked."""
        if not isinstance(text, str):
            raise TokenizerError(f"text must be str, got {type(text).__name__}")
        ids = list(self._backend.encode(text).ids)
        if add_bos:
            ids.insert(0, self.token_id("bos"))
        if add_eos:
            ids.append(self.token_id("eos"))
        return ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        """Decode ids. Special tokens stay in the string unless skipping is requested.

        The default keeps specials so a string that contained ``<|eos|>`` round-trips.
        Generation display can pass ``skip_special_tokens=True``.
        """
        if isinstance(ids, bool) or not isinstance(ids, Sequence) or isinstance(ids, str):
            raise TokenizerError("ids must be a sequence of integers")
        try:
            values = [int(token_id) for token_id in ids]
        except (TypeError, ValueError) as exc:
            raise TokenizerError("ids must be a sequence of integers") from exc
        for token_id in values:
            if token_id < 0 or token_id >= self.vocab_size:
                raise TokenizerError(f"token id {token_id} is outside 0..{self.vocab_size - 1}")
        return self._backend.decode(values, skip_special_tokens=skip_special_tokens)

    def pad_batch(self, sequences: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[list[int]]]:
        """Pad sequences with ``<|pad|>``. The mask is 1 on real tokens and 0 on pads."""
        if not sequences:
            raise TokenizerError("no sequences to pad")
        width = max(len(sequence) for sequence in sequences)
        pad_id = self.token_id("pad")
        padded: list[list[int]] = []
        mask: list[list[int]] = []
        for sequence in sequences:
            gap = width - len(sequence)
            padded.append(list(sequence) + [pad_id] * gap)
            mask.append([1] * len(sequence) + [0] * gap)
        return padded, mask

    def length_statistics(self, texts: Iterable[str]) -> TokenStats:
        """Token lengths for documents. Does not add BOS or EOS."""
        lengths: list[int] = []
        characters = 0
        for index, text in enumerate(texts):
            if not isinstance(text, str) or text == "":
                raise TokenizerError(f"document {index} must be a non-empty string")
            lengths.append(len(self.encode(text)))
            characters += len(text)
        if not lengths:
            raise TokenizerError("no documents to measure")
        ordered = sorted(lengths)
        midpoint = len(ordered) // 2
        if len(ordered) % 2 == 1:
            median = float(ordered[midpoint])
        else:
            median = (ordered[midpoint - 1] + ordered[midpoint]) / 2
        tokens = sum(lengths)
        return TokenStats(
            documents=len(lengths),
            characters=characters,
            tokens=tokens,
            min_tokens=ordered[0],
            max_tokens=ordered[-1],
            mean_tokens=tokens / len(lengths),
            median_tokens=median,
            characters_per_token=characters / tokens,
        )

    def save(self, directory: Path | str, *, overwrite: bool = False) -> Path:
        """Write ``tokenizer.json`` and ``tokenizer_meta.json``. Returns the directory."""
        path = Path(directory)
        if path.exists() and any(path.iterdir()) and not overwrite:
            raise TokenizerError(f"{path} is not empty. Pass overwrite=True to replace it.")
        path.mkdir(parents=True, exist_ok=True)
        self._backend.save(str(path / TOKENIZER_FILE))
        special_ids = {key: self.token_id(key) for key in SPECIAL_TOKEN_KEYS}
        meta = {
            "meta_version": META_VERSION,
            "type": self.spec.type,
            "vocab_size": self.spec.vocab_size,
            "add_prefix_space": self.spec.add_prefix_space,
            "use_regex": True,
            "special_tokens": {key: self.spec.special_tokens[key] for key in SPECIAL_TOKEN_KEYS},
            "special_ids": special_ids,
            "n_trained": self.n_trained,
            "n_reserved": self.n_reserved,
        }
        (path / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path | str) -> CodeTokenizer:
        path = Path(directory)
        meta_path = path / META_FILE
        model_path = path / TOKENIZER_FILE
        if not meta_path.is_file() or not model_path.is_file():
            raise TokenizerError(f"{path} is missing {TOKENIZER_FILE} or {META_FILE}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TokenizerError(f"{meta_path} is not valid JSON") from exc
        if not isinstance(meta, dict):
            raise TokenizerError(f"{meta_path} must contain an object")
        if meta.get("meta_version") != META_VERSION or meta.get("type") != "byte_level_bpe":
            raise TokenizerError(f"{meta_path} is not a codeslm byte-level BPE file")
        if meta.get("add_prefix_space") is not False:
            raise TokenizerError("saved tokenizer must have add_prefix_space false")
        specials = meta.get("special_tokens")
        if not isinstance(specials, dict):
            raise TokenizerError("saved tokenizer is missing special_tokens")
        spec = TokenizerSpec(
            type="byte_level_bpe",
            vocab_size=int(meta["vocab_size"]),
            add_prefix_space=False,
            special_tokens=specials,
        )
        backend = Tokenizer.from_file(str(model_path))
        loaded = cls(backend, spec, n_trained=int(meta["n_trained"]))
        saved_ids = meta.get("special_ids")
        if not isinstance(saved_ids, dict):
            raise TokenizerError("saved tokenizer is missing special_ids")
        for key in SPECIAL_TOKEN_KEYS:
            if loaded.token_id(key) != int(saved_ids[key]):
                raise TokenizerError(f"special id for {key} does not match the saved file")
        if loaded.n_reserved != int(meta["n_reserved"]):
            raise TokenizerError("saved n_reserved does not match the vocabulary")
        return loaded

    def _check_special_ids(self) -> None:
        expected = special_token_strings(self.spec)
        for index, (key, token) in enumerate(zip(SPECIAL_TOKEN_KEYS, expected, strict=True)):
            found = self._backend.token_to_id(token)
            if found != index:
                raise TokenizerError(
                    f"special token {key} ({token}) has id {found}, expected {index}"
                )


def train_tokenizer(
    texts: Iterable[str],
    spec: TokenizerSpec,
    *,
    min_frequency: int = 2,
    show_progress: bool = False,
) -> CodeTokenizer:
    """Train a byte-level BPE tokenizer and pad the vocabulary out to ``spec.vocab_size``.

    Padding uses ``<|reserved_N|>`` when the corpus does not contain enough
    distinct merges. Those ids keep the embedding matrix the size the model
    config declares. A tiny corpus therefore does not silently shrink the model.
    """
    if spec.type != "byte_level_bpe":
        raise TokenizerError(f"unsupported tokenizer type {spec.type!r}")
    if spec.add_prefix_space:
        raise TokenizerError("add_prefix_space must be false for source code")
    if min_frequency < 1:
        raise TokenizerError("min_frequency must be >= 1")
    specials = special_token_strings(spec)
    minimum = len(specials) + BYTE_ALPHABET_SIZE
    if spec.vocab_size < minimum:
        raise TokenizerError(
            f"vocab_size {spec.vocab_size} is below {minimum}, "
            "which is the special tokens plus the 256-byte alphabet"
        )
    backend = Tokenizer(BPE(unk_token=spec.special_tokens["unk"]))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=spec.vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        special_tokens=specials,
        initial_alphabet=ByteLevel.alphabet(),
    )
    documents = _checked_documents(texts)
    backend.train_from_iterator(documents, trainer=trainer)
    n_trained = backend.get_vocab_size()
    if n_trained > spec.vocab_size:
        raise TokenizerError(f"trainer produced {n_trained} tokens, above vocab_size {spec.vocab_size}")
    reserved = [f"<|reserved_{index}|>" for index in range(spec.vocab_size - n_trained)]
    if reserved:
        added = backend.add_tokens(reserved)
        if added != len(reserved) or backend.get_vocab_size() != spec.vocab_size:
            raise TokenizerError(
                f"failed to pad vocabulary to {spec.vocab_size} (added {added}, size {backend.get_vocab_size()})"
            )
    return CodeTokenizer(backend, spec, n_trained=n_trained)


def special_token_strings(spec: TokenizerSpec) -> list[str]:
    """Special-token strings in the stable id order used by the model config."""
    missing = [key for key in SPECIAL_TOKEN_KEYS if key not in spec.special_tokens]
    if missing:
        raise TokenizerError(f"tokenizer spec is missing special tokens {missing}")
    tokens = [spec.special_tokens[key] for key in SPECIAL_TOKEN_KEYS]
    if any(not isinstance(token, str) or not token for token in tokens):
        raise TokenizerError("special tokens must be non-empty strings")
    if len(set(tokens)) != len(tokens):
        raise TokenizerError("special tokens must be unique")
    return tokens


def _checked_documents(texts: Iterable[str]) -> list[str]:
    """Read the corpus into memory so the trainer can scan it safely.

    The first approved corpora are capped well below the size of this disk.
    A multi-gigabyte streaming trainer can replace this once real data is approved.
    """
    documents: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TokenizerError(f"document {index} is {type(text).__name__}, expected str")
        if text == "":
            raise TokenizerError(f"document {index} is empty")
        documents.append(text)
    if not documents:
        raise TokenizerError("training corpus is empty")
    return documents
