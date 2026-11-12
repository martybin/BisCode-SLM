"""Run local filtering, dedup, and leakage checks.

``run_pipeline`` never downloads data. Callers pass records they already hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codeslm.config.schema import DataConfig
from codeslm.data.dedup import mark_exact_duplicates, mark_near_duplicates
from codeslm.data.errors import DataError
from codeslm.data.filters import filter_records
from codeslm.data.records import CodeRecord, Decision
from codeslm.data.reports import write_reports
from codeslm.data.split import mark_repository_cap, mark_validation_leakage


@dataclass(frozen=True)
class PipelineResult:
    train: tuple[CodeRecord, ...]
    validation: tuple[CodeRecord, ...]
    train_decisions: tuple[Decision, ...]
    validation_decisions: tuple[Decision, ...]

    @property
    def decisions(self) -> tuple[Decision, ...]:
        return self.train_decisions + self.validation_decisions


def run_pipeline(
    train_records: list[CodeRecord],
    config: DataConfig,
    *,
    validation_records: list[CodeRecord] | None = None,
    output_dir: Path | None = None,
    seed: int = 1234,
) -> PipelineResult:
    """Filter and deduplicate local records. Validation overlap is removed from training."""
    if config.split_strategy != "repository":
        raise DataError(f"unsupported split strategy {config.split_strategy!r}")
    validation = list(validation_records or [])
    train_decisions = _prepare(train_records, config, split="train", seed=seed)
    validation_decisions = _prepare(validation, config, split="validation", seed=seed)
    if config.drop_train_repos_seen_in_validation:
        mark_validation_leakage(train_records, train_decisions, validation, validation_decisions)
    if output_dir is not None:
        write_reports(
            output_dir,
            config=config,
            train_records=train_records,
            train_decisions=train_decisions,
            validation_records=validation,
            validation_decisions=validation_decisions,
        )
    return PipelineResult(
        train=_kept(train_records, train_decisions),
        validation=_kept(validation, validation_decisions),
        train_decisions=tuple(train_decisions),
        validation_decisions=tuple(validation_decisions),
    )


def _prepare(records: list[CodeRecord], config: DataConfig, *, split: str, seed: int) -> list[Decision]:
    decisions = filter_records(records, config, split=split)
    if config.exact_dedup:
        mark_exact_duplicates(records, decisions)
    if config.minhash:
        mark_near_duplicates(records, decisions, threshold=config.jaccard_threshold, seed=seed)
    mark_repository_cap(records, decisions, max_files=config.max_files_per_repo)
    return decisions


def _kept(records: list[CodeRecord], decisions: list[Decision]) -> tuple[CodeRecord, ...]:
    return tuple(record for record, decision in zip(records, decisions, strict=True) if decision.disposition == "keep")
