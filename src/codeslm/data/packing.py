"""Pack documents into fixed-length causal-language-model rows.

Each document is followed by EOS. Position ids restart at 0 when a new document
starts. A document that crosses a row boundary continues its positions in the
next row. Tokens do not predict across a document boundary, and pad tokens are
ignored by the loss.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from codeslm.data.errors import DataError
from codeslm.tokenizer.bpe import CodeTokenizer

IGNORE_INDEX = -100


@dataclass(frozen=True)
class PackedSequence:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    position_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    block_ids: tuple[int, ...]


def pack_texts(texts: Sequence[str], tokenizer: CodeTokenizer, seq_len: int) -> list[PackedSequence]:
    """Encode texts, append EOS, and pack them."""
    if seq_len < 2:
        raise DataError("seq_len must be at least 2 so a token can have a label")
    rows = [tokenizer.encode(text, add_eos=True) for text in texts]
    if any(not row for row in rows):
        raise DataError("a document encoded to zero tokens")
    return pack_token_rows(rows, seq_len=seq_len, pad_id=tokenizer.token_id("pad"))


def pack_token_rows(rows: Sequence[Sequence[int]], *, seq_len: int, pad_id: int) -> list[PackedSequence]:
    """Pack pre-encoded documents. Each row already includes its end-of-sequence id."""
    if seq_len < 2:
        raise DataError("seq_len must be at least 2")
    packed: list[PackedSequence] = []
    current_ids: list[int] = []
    current_positions: list[int] = []
    current_blocks: list[int] = []
    next_block = 0

    def emit() -> None:
        gap = seq_len - len(current_ids)
        ids = current_ids + [pad_id] * gap
        positions = current_positions + [0] * gap
        blocks = current_blocks + [-1] * gap
        mask = [1] * len(current_ids) + [0] * gap
        labels = [_label_at(ids, blocks, mask, index, seq_len) for index in range(seq_len)]
        packed.append(
            PackedSequence(
                input_ids=tuple(ids),
                labels=tuple(labels),
                position_ids=tuple(positions),
                attention_mask=tuple(mask),
                block_ids=tuple(blocks),
            )
        )
        current_ids.clear()
        current_positions.clear()
        current_blocks.clear()

    for row in rows:
        if not row:
            raise DataError("cannot pack an empty token row")
        offset = 0
        block = next_block
        next_block += 1
        while offset < len(row):
            space = seq_len - len(current_ids)
            if space == 0:
                emit()
                space = seq_len
            take = min(space, len(row) - offset)
            for step in range(take):
                current_ids.append(row[offset + step])
                current_positions.append(offset + step)
                current_blocks.append(block)
            offset += take
            if len(current_ids) == seq_len:
                emit()
    if current_ids:
        emit()
    return packed


def causal_block_mask(block_ids: Sequence[int]) -> list[list[bool]]:
    """True when ``j`` is an earlier token in the same non-pad document as ``i``."""
    size = len(block_ids)
    mask: list[list[bool]] = []
    for row_index, row_block in enumerate(block_ids):
        mask.append(
            [
                row_block >= 0 and row_block == block_ids[column_index] and column_index <= row_index
                for column_index in range(size)
            ]
        )
    return mask


def _label_at(ids: list[int], blocks: list[int], mask: list[int], index: int, seq_len: int) -> int:
    if index + 1 >= seq_len or mask[index] == 0 or mask[index + 1] == 0:
        return IGNORE_INDEX
    if blocks[index] < 0 or blocks[index] != blocks[index + 1]:
        return IGNORE_INDEX
    return ids[index + 1]
