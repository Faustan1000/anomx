"""Datasets connected to an Anomx platform.

An :class:`AnomxDataset` bridges local work and the platform's data module:

```python
dataset = AnomxDataset.from_anomx("data_dataset:42f1…")   # load by identifier
observation_set = dataset.to_observation_set()

dataset = AnomxDataset(name="lab-run-7", frame=frame)
dataset.sync_with_anomx()                                  # create it in anomx
```

Credentials come from the `~/.anomx` home the CLI agent maintains; without a
configured platform connection an :class:`AnomxConnectionError` is raised.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from anomx._shared import ensure_dataframe
from anomx.data.base.observation_set import ObservationSet

REQUEST_TIMEOUT_SECONDS = 30


class AnomxConnectionError(RuntimeError):
    """Raised when no platform connection is configured or a call fails."""


def read_platform_connection(home: Any | None = None) -> dict[str, str]:
    """Read the platform URL and token from the `~/.anomx` home."""
    from anomx.agent.store import AnomxHome

    anomx_home = home if home is not None else AnomxHome()
    connection = anomx_home.platform_connection()
    if connection is None:
        raise AnomxConnectionError(
            "No Anomx platform connection is configured in `~/.anomx`. "
            "Run `anomx connect` first."
        )
    return connection


def _platform_request(
    connection: dict[str, str],
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> tuple[int, bytes, str]:
    url = f"{connection['url'].rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(  # noqa: S310 - URL comes from the user's own configuration
        url,
        data=body,
        method=method,
        headers={
            "Accept": "application/json, text/csv, application/octet-stream",
            "Authorization": f"Bearer {connection['token']}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            return response.status, response.read(), str(response.headers.get("Content-Type") or "")
    except HTTPError as error:
        detail = ""
        try:
            detail = json.loads(error.read().decode("utf-8", errors="replace")).get("detail", "")
        except Exception:
            pass
        raise AnomxConnectionError(
            f"Anomx platform call `{method} {path}` failed with HTTP {error.code}."
            + (f" {detail}" if detail else "")
        ) from error
    except URLError as error:
        raise AnomxConnectionError(f"Could not reach the Anomx platform at `{connection['url']}`: {error.reason}") from error


class AnomxDataset:
    """A dataset that can be loaded from or created on an Anomx platform."""

    def __init__(
        self,
        *,
        identifier: str | None = None,
        name: str = "",
        description: str = "",
        frame: pd.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.identifier = str(identifier).strip() if identifier else None
        self.name = name
        self.description = description
        self.frame = ensure_dataframe(frame) if frame is not None else pd.DataFrame()
        self.metadata = dict(metadata or {})

    @classmethod
    def from_anomx(cls, identifier: str, *, home: Any | None = None) -> "AnomxDataset":
        """Load a whole dataset (metadata and samples) from the platform."""
        connection = read_platform_connection(home)
        normalized_identifier = str(identifier or "").strip()
        if not normalized_identifier:
            raise AnomxConnectionError("A dataset identifier is required.")

        _, metadata_bytes, _ = _platform_request(connection, f"/datasets/{normalized_identifier}")
        metadata = json.loads(metadata_bytes.decode("utf-8"))

        status, export_bytes, content_type = _platform_request(
            connection,
            f"/datasets/{normalized_identifier}/export",
            method="POST",
            payload={"format": "csv"},
        )
        if status == 202 or "json" in content_type:
            raise AnomxConnectionError(
                "The platform queued the dataset export for delivery instead of returning it directly; "
                "the dataset is too large for synchronous loading."
            )
        frame = pd.read_csv(io.StringIO(export_bytes.decode("utf-8"))) if export_bytes.strip() else pd.DataFrame()

        return cls(
            identifier=normalized_identifier,
            name=str(metadata.get("name") or ""),
            description=str(metadata.get("description") or ""),
            frame=frame,
            metadata=metadata,
        )

    def sync_with_anomx(self, *, home: Any | None = None) -> str:
        """Create this locally composed dataset on the platform.

        Returns the created dataset identifier and stores it on the instance.
        Sample upload is not part of this first pass — the platform object is
        created with the dataset's metadata.
        """
        connection = read_platform_connection(home)
        if self.identifier:
            raise AnomxConnectionError(f"This dataset is already synced as `{self.identifier}`.")
        if not self.name.strip():
            raise AnomxConnectionError("A dataset name is required before syncing with Anomx.")

        _, response_bytes, _ = _platform_request(
            connection,
            "/datasets",
            method="POST",
            payload={
                "name": self.name.strip(),
                "description": self.description.strip(),
            },
        )
        payload = json.loads(response_bytes.decode("utf-8"))
        anomx_meta = payload.get("_anomx") if isinstance(payload.get("_anomx"), dict) else {}
        self.identifier = str(anomx_meta.get("object_reference") or payload.get("id") or "").strip() or None
        if not self.identifier:
            raise AnomxConnectionError("The platform did not return an identifier for the created dataset.")
        self.metadata = payload
        return self.identifier

    def to_observation_set(self, **kwargs: Any) -> ObservationSet:
        """Materialize the loaded samples as the canonical observation set."""
        return ObservationSet(observations=self.frame.copy(), metadata={"anomx_identifier": self.identifier or ""}, **kwargs)

    def to_csv(self) -> str:
        """Serialize the samples as CSV (mirrors the platform export format)."""
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(self.frame.columns.tolist())
        for row in self.frame.itertuples(index=False):
            writer.writerow(list(row))
        return buffer.getvalue()
