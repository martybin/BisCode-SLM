"""Tokenizer tests on a synthetic corpus. No dataset download."""

from pathlib import Path

import pytest

from codeslm.config.schema import TokenizerSpec, load_model_config
from codeslm.tokenizer import TokenizerError, train_tokenizer
from codeslm.tokenizer.cli import main

ROOT = Path(__file__).resolve().parents[1]

PYTHON = '''def add(left, right):
    """Add two numbers."""
    if left is None:
        return right
    total = left + right
    return total


class Box:
    def __init__(self, value):
        self.value = value
'''

SQL = "SELECT id, name\nFROM users\nWHERE name = 'Ada';\n"
WHITESPACE = "\n\n    \tif value >= 0:\n        return value\n"
UNICODE = "print('café — λ 😀')\n"
OPERATORS = "x = a + b * c // d == e and not f\n"


def test_roundtrip_preserves_code_whitespace_and_unicode() -> None:
    tokenizer = _train(300)
    for text in (PYTHON, SQL, WHITESPACE, UNICODE, OPERATORS):
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_occupy_the_first_ids_and_stay_atomic() -> None:
    tokenizer = _train(300)
    spec = _spec(300)
    vocab = tokenizer.vocab_by_id()
    for index, key in enumerate(("pad", "unk", "bos", "eos", "fim_prefix", "fim_middle", "fim_suffix", "sep")):
        assert tokenizer.token_id(key) == index
        assert vocab[index] == spec.special_tokens[key]
    text = "keep <|eos|> inside\n"
    ids = tokenizer.encode(text)
    assert tokenizer.token_id("eos") in ids
    assert ids.count(tokenizer.token_id("eos")) == 1
    assert tokenizer.decode(ids) == text
    assert "<|eos|>" not in tokenizer.decode(ids, skip_special_tokens=True)


def test_byte_alphabet_covers_text_absent_from_training() -> None:
    tokenizer = _train(280, corpus=[PYTHON])
    unseen = "async def gather(items):\n    return [item for item in items if item]\n"
    ids = tokenizer.encode(unseen)
    assert tokenizer.token_id("unk") not in ids
    assert tokenizer.decode(ids) == unseen


def test_requested_vocab_size_is_padded_when_the_corpus_is_small() -> None:
    tokenizer = train_tokenizer([PYTHON], _spec(800), min_frequency=1)
    assert tokenizer.vocab_size == 800
    assert 264 <= tokenizer.n_trained < 800
    assert tokenizer.n_reserved == 800 - tokenizer.n_trained
    assert tokenizer.token_for_id(799).startswith("<|reserved_")


def test_bos_and_eos_flags_and_padding_mask() -> None:
    tokenizer = _train(300)
    ids = tokenizer.encode("return 1\n", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.token_id("bos")
    assert ids[-1] == tokenizer.token_id("eos")
    assert tokenizer.decode(ids) == "<|bos|>return 1\n<|eos|>"
    short = tokenizer.encode("a\n")
    long = tokenizer.encode(PYTHON)
    padded, mask = tokenizer.pad_batch([short, long])
    assert len(padded[0]) == len(padded[1]) == len(long)
    assert mask[0].count(1) == len(short)
    assert mask[0].count(0) == len(long) - len(short)
    assert padded[0][len(short):] == [tokenizer.token_id("pad")] * (len(long) - len(short))
    assert sum(mask[1]) == len(long)


def test_length_statistics() -> None:
    tokenizer = _train(300)
    stats = tokenizer.length_statistics([PYTHON, SQL, WHITESPACE])
    assert stats.documents == 3
    assert stats.characters == len(PYTHON) + len(SQL) + len(WHITESPACE)
    assert stats.tokens == sum(len(tokenizer.encode(text)) for text in (PYTHON, SQL, WHITESPACE))
    assert stats.min_tokens <= stats.median_tokens <= stats.max_tokens
    assert stats.characters_per_token == stats.characters / stats.tokens


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    tokenizer = _train(300)
    directory = tokenizer.save(tmp_path / "tokenizer")
    loaded = type(tokenizer).load(directory)
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.n_trained == tokenizer.n_trained
    assert loaded.encode(PYTHON) == tokenizer.encode(PYTHON)
    assert loaded.decode(loaded.encode(UNICODE)) == UNICODE
    with pytest.raises(TokenizerError, match="not empty"):
        tokenizer.save(directory)


def test_rejects_empty_corpus_and_a_vocab_below_the_byte_alphabet() -> None:
    with pytest.raises(TokenizerError, match="empty"):
        train_tokenizer([], _spec(300))
    with pytest.raises(TokenizerError, match="empty"):
        train_tokenizer([PYTHON, ""], _spec(300))
    with pytest.raises(TokenizerError, match="vocab_size"):
        train_tokenizer([PYTHON], _spec(32))


def test_cli_writes_a_small_tokenizer_and_refuses_a_mostly_reserved_vocab(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in {"add.py": PYTHON, "query.sql": SQL, "note.py": UNICODE}.items():
        (corpus / name).write_text(text, encoding="utf-8")
    output = tmp_path / "out"
    status = main(
        [
            "--config",
            str(ROOT / "configs/model/smoke.yaml"),
            "--corpus",
            str(corpus),
            "--output",
            str(output),
            "--vocab-size",
            "280",
            "--min-frequency",
            "1",
        ]
    )
    assert status == 0
    assert (output / "tokenizer.json").is_file()
    assert (output / "tokenizer_meta.json").is_file()
    refused = tmp_path / "refused"
    status = main(
        [
            "--config",
            str(ROOT / "configs/model/base.yaml"),
            "--corpus",
            str(corpus),
            "--output",
            str(refused),
            "--vocab-size",
            "32000",
        ]
    )
    assert status == 2
    assert not refused.exists()


def _spec(vocab_size: int) -> TokenizerSpec:
    shipped = load_model_config(ROOT / "configs/model/smoke.yaml").tokenizer
    return TokenizerSpec(
        type=shipped.type,
        vocab_size=vocab_size,
        add_prefix_space=shipped.add_prefix_space,
        special_tokens=dict(shipped.special_tokens),
    )


def _train(vocab_size: int, corpus: list[str] | None = None):
    documents = corpus or [PYTHON, SQL, WHITESPACE, UNICODE, OPERATORS]
    return train_tokenizer(documents, _spec(vocab_size), min_frequency=1)
