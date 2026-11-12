"""Repository-level splits.

Files that share ``repo_name`` stay in the same split. A repository that appears
in the validation records is removed from training, and so is a training body
whose exact hash appears in validation.
"""

from __future__ import annotations

import hashlib

from codeslm.data.dedup import content_hash
from codeslm.data.records import CodeRecord, Decision


def assign_repositories(
    records: list[CodeRecord],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[CodeRecord], list[CodeRecord]]:
    """Split records by repository hash. Fraction 0 keeps every record for training."""
    if not 0.0 <= validation_fraction <= 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    train: list[CodeRecord] = []
    validation: list[CodeRecord] = []
    for record in records:
        if _repository_unit(record.repo_name, seed) < validation_fraction:
            validation.append(record)
        else:
            train.append(record)
    return train, validation


def mark_validation_leakage(
    train_records: list[CodeRecord],
    train_decisions: list[Decision],
    validation_records: list[CodeRecord],
    validation_decisions: list[Decision],
) -> None:
    """Drop kept training records that repeat a kept validation repo or body."""
    validation_repos = {
        record.repo_name
        for record, decision in zip(validation_records, validation_decisions, strict=True)
        if decision.disposition == "keep"
    }
    validation_hashes = {
        content_hash(record.content)
        for record, decision in zip(validation_records, validation_decisions, strict=True)
        if decision.disposition == "keep"
    }
    for index, record in enumerate(train_records):
        if train_decisions[index].disposition != "keep":
            continue
        if record.repo_name in validation_repos:
            _replace(train_decisions, index, "exclude", "validation_repo_overlap", record.repo_name)
            continue
        digest = content_hash(record.content)
        if digest in validation_hashes:
            _replace(train_decisions, index, "exclude", "validation_content_overlap", f"sha256={digest[:16]}")


def mark_repository_cap(records: list[CodeRecord], decisions: list[Decision], *, max_files: int) -> None:
    """Keep at most ``max_files`` kept files per repository, chosen by path order."""
    if max_files < 1:
        raise ValueError("max_files must be positive")
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if decisions[index].disposition != "keep":
            continue
        grouped.setdefault(record.repo_name, []).append(index)
    for indexes in grouped.values():
        ordered = sorted(indexes, key=lambda index: (records[index].path, index))
        for extra in ordered[max_files:]:
            _replace(decisions, extra, "exclude", "repo_file_cap", f"max_files={max_files}")


def _repository_unit(repo_name: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{repo_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / 2**64


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
