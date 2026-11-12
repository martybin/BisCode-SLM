"""Exact hashing and MinHash near-duplicate detection.

Exact duplicates use the SHA-256 of the UTF-8 bytes, with no whitespace
normalization. Near-duplicates use 5-word shingles, 64 MinHash permutations,
and 16 LSH bands of 4 rows. A later record is dropped only when its estimated
Jaccard similarity with an earlier kept record is at least the configured
threshold. The kept record stays.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

from codeslm.data.records import CodeRecord, Decision

_PRIME = (1 << 61) - 1
_NUM_PERM = 64
_BANDS = 16
_ROWS = 4
_SHINGLE_SIZE = 5


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def mark_exact_duplicates(records: list[CodeRecord], decisions: list[Decision]) -> None:
    """Exclude later copies of an identical kept body. Updates ``decisions`` in place."""
    seen: dict[str, int] = {}
    for index, record in enumerate(records):
        if decisions[index].disposition != "keep":
            continue
        digest = content_hash(record.content)
        earlier = seen.get(digest)
        if earlier is None:
            seen[digest] = index
            continue
        _replace(
            decisions,
            index,
            "exclude",
            "exact_duplicate",
            f"sha256={digest[:16]} kept_index={earlier}",
        )


def mark_near_duplicates(records: list[CodeRecord], decisions: list[Decision], *, threshold: float, seed: int) -> None:
    """Exclude later near-duplicates of a kept record. Updates ``decisions`` in place."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("jaccard threshold must be in (0, 1]")
    candidates = [index for index, decision in enumerate(decisions) if decision.disposition == "keep"]
    if len(candidates) < 2:
        return
    hasher = MinHasher(num_perm=_NUM_PERM, seed=seed)
    signatures = {index: hasher.signature(records[index].content) for index in candidates}
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index in candidates:
        signature = signatures[index]
        keys = list(_band_keys(signature))
        matches: set[int] = set()
        for key in keys:
            matches.update(buckets[key])
        duplicate_of: tuple[int, float] | None = None
        for other in matches:
            estimate = _agreement(signature, signatures[other])
            if estimate >= threshold:
                duplicate_of = (other, estimate)
                break
        if duplicate_of is None:
            for key in keys:
                buckets[key].append(index)
            continue
        other, estimate = duplicate_of
        _replace(
            decisions,
            index,
            "exclude",
            "near_duplicate",
            f"jaccard={estimate:.3f} kept_index={other} kept_path={records[other].path}",
        )


class MinHasher:
    """MinHash signatures over word shingles."""

    def __init__(self, num_perm: int = _NUM_PERM, seed: int = 1) -> None:
        if num_perm < 1:
            raise ValueError("num_perm must be positive")
        rng = random.Random(seed)
        self._a = [rng.randrange(1, _PRIME) for _ in range(num_perm)]
        self._b = [rng.randrange(0, _PRIME) for _ in range(num_perm)]

    def signature(self, text: str) -> tuple[int, ...]:
        shingles = word_shingles(text, _SHINGLE_SIZE)
        hashed = [_shingle_hash(shingle) for shingle in shingles]
        signature: list[int] = []
        for coefficient, offset in zip(self._a, self._b, strict=True):
            signature.append(min((coefficient * value + offset) % _PRIME for value in hashed))
        return tuple(signature)


def word_shingles(text: str, size: int) -> tuple[str, ...]:
    words = text.split()
    if not words:
        return ("",)
    if len(words) <= size:
        return (" ".join(words),)
    return tuple(" ".join(words[start : start + size]) for start in range(len(words) - size + 1))


def estimate_jaccard(text_a: str, text_b: str, *, num_perm: int = _NUM_PERM, seed: int = 1) -> float:
    hasher = MinHasher(num_perm=num_perm, seed=seed)
    return _agreement(hasher.signature(text_a), hasher.signature(text_b))


def _agreement(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("signatures must be non-empty and the same length")
    matches = sum(item_a == item_b for item_a, item_b in zip(left, right, strict=True))
    return matches / len(left)


def _shingle_hash(shingle: str) -> int:
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _band_keys(signature: tuple[int, ...]) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if len(signature) != _BANDS * _ROWS:
        raise ValueError(f"signature length {len(signature)} does not match {_BANDS} bands of {_ROWS}")
    return tuple(
        (band, signature[band * _ROWS : (band + 1) * _ROWS])
        for band in range(_BANDS)
    )


def _replace(decisions: list[Decision], index: int, disposition: str, reason: str, detail: str) -> None:
    current = decisions[index]
    decisions[index] = Decision(
        split=current.split,
        record_index=current.record_index,
        source_id=current.source_id,
        repo_name=current.repo_name,
        path=current.path,
        license=current.license,
        language=current.language,
        disposition=disposition,
        reason=reason,
        detail=detail,
    )
