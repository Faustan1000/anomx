"""Explicit HTTP access to Anomx data, without importing Django or the agent."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class PlatformError(RuntimeError):
    """A platform request failed or returned an invalid response."""


@dataclass(frozen=True)
class PlatformClient:
    """HTTP adapter configured explicitly or with ANOMX_URL and ANOMX_TOKEN.

    ``url`` includes the API prefix (for example ``https://host/api``).
    Credentials are never loaded from the agent or embedded in dataset metadata.
    """

    url: str
    token: str = field(repr=False)
    timeout: float = 30.0
    channels_path: str = "/jobs/channel-data"

    def __post_init__(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Anomx URL must be an absolute HTTP(S) API URL.")
        if not self.token or self.timeout <= 0:
            raise ValueError("Anomx token and a positive request timeout are required.")

    @classmethod
    def from_env(cls) -> PlatformClient:
        """Read an explicit standalone-script connection from the environment."""
        url = os.environ.get("ANOMX_URL", "")
        token = os.environ.get("ANOMX_TOKEN", "")
        if not url or not token:
            raise PlatformError(
                "Set ANOMX_URL and ANOMX_TOKEN, or provide PlatformClient explicitly."
            )
        return cls(url=url, token=token)

    def request(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        """POST a JSON request and return its decoded JSON response."""
        if not path.startswith("/") or path.startswith("//") or ".." in path.split("/"):
            raise ValueError("API paths must be absolute paths within the configured API.")
        request = Request(
            self.url.rstrip("/") + path,
            data=json.dumps(payload, allow_nan=False).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
                if not isinstance(result, (dict, list)):
                    raise ValueError("Expected a JSON object or list.")
                return result
        except HTTPError as exc:
            raise PlatformError(f"Anomx API returned HTTP {exc.code} for {path}.") from exc
        except URLError as exc:
            raise PlatformError(f"Could not connect to the Anomx API: {exc.reason}") from exc
        except (ValueError, UnicodeError) as exc:
            raise PlatformError("Anomx API did not return valid JSON.") from exc

    def load_channels(
        self,
        channels: Sequence[str],
        *,
        start: str | None = None,
        end: str | None = None,
        frequency: str | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Read wide timestamped records from the authenticated channel endpoint."""
        result = self.request(
            self.channels_path,
            {
                "channels": list(channels),
                "start": start,
                "end": end,
                "frequency": frequency,
            },
        )
        return result.get("records", result) if isinstance(result, dict) else result


__all__ = ["PlatformClient", "PlatformError"]
