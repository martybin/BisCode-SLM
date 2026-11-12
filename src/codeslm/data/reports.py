"""Provenance reports for a local pipeline run.

The writers record counts, licenses, and rejection reasons. They do not write
file contents.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from codeslm.config.schema import DataConfig
from codeslm.data.records import CodeRecord, Decision


def build_statistics(
    train_records: list[CodeRecord],
    train_decisions: list[Decision],
    validation_records: list[CodeRecord],
    validation_decisions: list[Decision],
) -> dict[str, object]:
    kept_train = _kept(train_records, train_decisions)
    kept_validation = _kept(validation_records, validation_decisions)
    reasons = Counter(decision.reason for decision in (*train_decisions, *validation_decisions))
    dispositions = Counter(decision.disposition for decision in (*train_decisions, *validation_decisions))
    languages = Counter(record.language for record in kept_train)
    return {
        "documents_in": len(train_records) + len(validation_records),
        "train_in": len(train_records),
        "validation_in": len(validation_records),
        "train_kept": len(kept_train),
        "validation_kept": len(kept_validation),
        "kept": dispositions["keep"],
        "excluded": dispositions["exclude"],
        "quarantined": dispositions["quarantine"],
        "reasons": dict(sorted(reasons.items())),
        "languages_train_kept": dict(sorted(languages.items())),
        "repositories_train_kept": len({record.repo_name for record in kept_train}),
        "characters_train_kept": sum(len(record.content) for record in kept_train),
        "exact_duplicates": reasons["exact_duplicate"],
        "near_duplicates": reasons["near_duplicate"],
    }


def build_license_report(
    records: list[CodeRecord],
    decisions: list[Decision],
    config: DataConfig,
) -> dict[str, object]:
    by_license: dict[str, Counter[str]] = {}
    unknown_examples: list[dict[str, str]] = []
    for record, decision in zip(records, decisions, strict=True):
        label = (record.license or "").strip().lower() or "empty"
        by_license.setdefault(label, Counter())
        by_license[label][decision.disposition] += 1
        if decision.reason == "license_unknown" and len(unknown_examples) < 20:
            unknown_examples.append(
                {
                    "split": decision.split,
                    "repo_name": record.repo_name,
                    "path": record.path,
                    "license": label,
                }
            )
    return {
        "allow": list(config.license_allow),
        "exclude": list(config.license_exclude),
        "quarantine_unknown": config.quarantine_unknown,
        "counts_by_license": {label: dict(sorted(counts.items())) for label, counts in sorted(by_license.items())},
        "unknown_examples": unknown_examples,
    }


def build_manifest(config: DataConfig, statistics: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "data_config": config.name,
        "download_enabled": config.download_enabled,
        "network_fetch": False,
        "split_strategy": config.split_strategy,
        "train_repo": config.train_repo,
        "validation_repo": config.validation_repo,
        "license_allow": list(config.license_allow),
        "statistics": statistics,
    }


def write_reports(
    directory: Path,
    *,
    config: DataConfig,
    train_records: list[CodeRecord],
    train_decisions: list[Decision],
    validation_records: list[CodeRecord],
    validation_decisions: list[Decision],
) -> dict[str, object]:
    """Write the manifest, license report, statistics, and a decision log."""
    directory.mkdir(parents=True, exist_ok=True)
    statistics = build_statistics(train_records, train_decisions, validation_records, validation_decisions)
    license_report = build_license_report(
        [*train_records, *validation_records],
        [*train_decisions, *validation_decisions],
        config,
    )
    manifest = build_manifest(config, statistics)
    (directory / "dataset_statistics.json").write_text(json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
    (directory / "license_report.json").write_text(json.dumps(license_report, indent=2) + "\n", encoding="utf-8")
    (directory / "dataset_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with (directory / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for decision in (*train_decisions, *validation_decisions):
            handle.write(
                json.dumps(
                    {
                        "split": decision.split,
                        "record_index": decision.record_index,
                        "source_id": decision.source_id,
                        "repo_name": decision.repo_name,
                        "path": decision.path,
                        "license": decision.license,
                        "language": decision.language,
                        "disposition": decision.disposition,
                        "reason": decision.reason,
                        "detail": decision.detail,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    return {"statistics": statistics, "license_report": license_report, "manifest": manifest}


def _kept(records: list[CodeRecord], decisions: list[Decision]) -> list[CodeRecord]:
    return [record for record, decision in zip(records, decisions, strict=True) if decision.disposition == "keep"]
