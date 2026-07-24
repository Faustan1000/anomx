"""Contracts shared by algorithm orchestrators."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class JobSpec:
    connector: str
    model: str
    detector: str
    scorer: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> JobSpec:
        return cls(
            connector=str(payload.get("connector") or "").strip(),
            model=str(payload.get("model") or "").strip(),
            detector=str(payload.get("detector") or "").strip(),
            scorer=str(payload.get("scorer") or "").strip(),
            config=dict(payload.get("config") or {}),
        )


@dataclass(frozen=True)
class JobSummary:
    rows_processed: int
    anomaly_count: int
    feature_columns: list[str] = field(default_factory=list)
    score_column: str = "zscore"
    duration_ms: int = 0


@dataclass(frozen=True)
class JobResult:
    job_id: str
    status: str
    connector: str
    model: str
    scorer: str
    detector: str
    summary: JobSummary
    records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
