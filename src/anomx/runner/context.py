"""JSON-encodable execution context passed from node to node.

A job's working state travels between nodes as one JSON-encodable dict — the
synchronous runner passes it in memory, the platform passes the same structure
between tasks via redis. Conventional keys:

```
data           input observations as records
predictions    model output records
scores         scorer output records
decisions      detector output records
classes        classifier output records
flags          booleans steering if/else gates (e.g. should_train)
model_ref      reference to the trained model in the executor registry
meta           free-form metadata (job id, step number, ...)
```
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from anomx._shared import ensure_dataframe

FRAME_KEYS = ("data", "predictions", "scores", "decisions", "classes")


def frame_to_records(frame: Any) -> list[dict[str, Any]]:
    """Serialize a DataFrame into JSON-friendly, index-free records.

    Records must round-trip loss-free between nodes, so the index is dropped
    and datetime columns are stringified.
    """
    normalized = ensure_dataframe(frame).reset_index(drop=True)
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].astype(str)
    return [
        {str(key): (None if pd.isna(value) else value) if not isinstance(value, (list, dict)) else value for key, value in record.items()}
        for record in normalized.to_dict(orient="records")
    ]


def records_to_frame(records: Any) -> pd.DataFrame:
    """Rebuild a DataFrame from context records."""
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, list):
        return pd.DataFrame(records)
    return ensure_dataframe(records)


def read_context_frame(context: dict[str, Any], key: str) -> pd.DataFrame | None:
    """Read one conventional frame key from the context."""
    value = context.get(key)
    if value is None or (isinstance(value, list) and not value):
        return None
    return records_to_frame(value)


def write_context_frame(context: dict[str, Any], key: str, frame: Any) -> None:
    """Write a frame into the context in its JSON-encodable form."""
    context[key] = frame_to_records(frame)


def read_context_flag(context: dict[str, Any], key: str) -> bool:
    """Read a boolean flag from `flags` with a top-level fallback."""
    flags = context.get("flags")
    if isinstance(flags, dict) and key in flags:
        return bool(flags.get(key))
    return bool(context.get(key))


def write_context_flag(context: dict[str, Any], key: str, value: bool) -> None:
    flags = context.get("flags")
    if not isinstance(flags, dict):
        flags = {}
        context["flags"] = flags
    flags[key] = bool(value)


def normalize_context(context: Any, data: Any = None) -> dict[str, Any]:
    """Build a valid context dict, optionally seeding the input data."""
    normalized_context: dict[str, Any] = dict(context) if isinstance(context, dict) else {}
    if data is not None:
        write_context_frame(normalized_context, "data", data)
    return normalized_context
