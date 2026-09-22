"""HTTP helpers for connecting the CLI agent to an Anomx Platform."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from anomx import __version__
from anomx.agent.store import AnomxHome
from anomx.agent.skills import sync_platform_skills

DEFAULT_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class PlatformLoginResult:
    """Result returned after a successful CLI agent platform login."""

    url: str
    token: str
    user_email: str
    organization_url: str
    hostname: str


class PlatformClientError(RuntimeError):
    """Raised when the CLI cannot talk to the configured platform."""


class PlatformHttpError(PlatformClientError):
    """Raised when the platform returns a non-success HTTP status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def normalize_platform_url(value: str) -> str:
    """Return a normalized platform API origin from user input."""

    candidate = value.strip().rstrip("/")
    if not candidate:
        raise PlatformClientError("Platform URL is required.")
    if "://" not in candidate:
        scheme = "http" if _looks_local(candidate) else "https"
        candidate = f"{scheme}://{candidate}"

    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PlatformClientError("Platform URL must include a valid http or https origin.")

    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))


def platform_domain(url: str) -> str:
    """Return a compact domain label for UI helper text."""

    parsed = urlparse(url)
    return parsed.netloc or url


def local_hostname() -> str:
    """Return the hostname reported to the platform for this CLI agent."""

    return socket.gethostname() or "unknown-host"


def connect_platform(url: str, email: str, password: str) -> PlatformLoginResult:
    """Log into the platform as a CLI agent and return the issued token."""

    normalized_url = resolve_platform_api_url(url)
    hostname = local_hostname()
    payload = _request_json(
        normalized_url,
        "/auth/login",
        {
            "email": email,
            "password": password,
            "client": "cli_agent",
            "client_hostname": hostname,
            "client_version": __version__,
        },
    )
    token = str(payload.get("token") or "").strip()
    if not token:
        raise PlatformClientError("Platform login did not return a bearer token.")
    _verify_cli_agent_token(normalized_url, token)

    user = payload.get("user") if isinstance(payload.get("user"), dict) else {}
    organization = user.get("organization") if isinstance(user.get("organization"), dict) else {}
    return PlatformLoginResult(
        url=normalized_url,
        token=token,
        user_email=str(user.get("email") or email),
        organization_url=str(organization.get("url") or ""),
        hostname=hostname,
    )


def heartbeat_platform_connection(home: AnomxHome) -> bool:
    """Refresh the current CLI agent token's last-used timestamp if configured."""

    connection = home.platform_connection()
    if connection is None:
        return False
    base_url = _canonical_platform_api_url(connection["url"])
    if base_url != connection["url"]:
        _store_platform_url(home, connection, base_url)
        connection = {
            **connection,
            "url": base_url,
        }
    token = connection["token"]
    usage = home.load_cli_usage()
    try:
        payload = _request_agent_heartbeat(base_url, token, usage=usage)
        _store_platform_context(home, payload)
    except PlatformHttpError as exc:
        if exc.status_code != 404:
            return False
        fallback_base_url = _api_fallback_url(base_url)
        if fallback_base_url is not None:
            fallback_payload = _try_platform_heartbeat(fallback_base_url, token, usage=usage)
            if fallback_payload is not None:
                _store_platform_url(home, connection, fallback_base_url)
                _store_platform_context(home, fallback_payload)
                return True
        return False
    except PlatformClientError:
        return False
    return True


def _verify_cli_agent_token(base_url: str, token: str) -> None:
    try:
        _request_agent_heartbeat(base_url, token)
    except PlatformHttpError as exc:
        if exc.status_code == 404:
            raise PlatformClientError(
                "This platform does not expose CLI agent registration. Update the platform backend."
            ) from exc
        if exc.status_code == 403:
            raise PlatformClientError(
                "The platform returned a normal session token instead of a CLI agent token. Update the platform backend."
            ) from exc
        raise


def _request_agent_heartbeat(
    base_url: str,
    token: str,
    *,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    return _request_json(
        base_url,
        "/auth/me/agent/heartbeat",
        {
            "client_hostname": local_hostname(),
            "client_version": __version__,
            "usage": usage if usage is not None else AnomxHome().load_cli_usage(),
        },
        token=token,
    )


def _try_platform_heartbeat(
    base_url: str,
    token: str,
    *,
    usage: dict[str, int] | None = None,
) -> dict[str, Any] | None:
    try:
        return _request_agent_heartbeat(base_url, token, usage=usage)
    except PlatformClientError:
        return None


def _store_platform_context(home: AnomxHome, payload: dict[str, Any]) -> None:
    config = home.load_config()
    config["platform_instructions"] = str(payload.get("platform_instructions") or "").strip()
    config["custom_instructions"] = str(payload.get("custom_instructions") or "").strip()
    home.save_config(config)
    sync_platform_skills(home.skills_dir, payload.get("skills"))


def _store_platform_url(home: AnomxHome, connection: dict[str, str], url: str) -> None:
    if connection["url"] == url:
        return
    home.set_platform_connection(
        url=url,
        token=connection["token"],
        user_email=connection.get("user_email", ""),
        organization_url=connection.get("organization_url", ""),
        hostname=connection.get("hostname", ""),
    )


def resolve_platform_api_url(value: str) -> str:
    """Return the platform API base URL, falling back to `/api` deployments."""

    normalized_url = _canonical_platform_api_url(normalize_platform_url(value))
    try:
        _request_json(normalized_url, "/auth/registration", {}, method="GET")
        return normalized_url
    except PlatformHttpError as exc:
        if exc.status_code != 404:
            return normalized_url

    fallback_url = _api_fallback_url(normalized_url)
    if fallback_url is None:
        return normalized_url
    try:
        _request_json(fallback_url, "/auth/registration", {}, method="GET")
    except PlatformHttpError as exc:
        if exc.status_code != 404:
            return fallback_url
        return normalized_url
    return fallback_url


def _request_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    method: str = "POST",
    token: str | None = None,
) -> dict[str, Any]:
    normalized_path = path if path.startswith("/") else f"/{path}"
    url = f"{base_url}{normalized_path}"
    body = None if method == "GET" else json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "User-Agent": f"anomx-cli/{__version__}",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise PlatformHttpError(exc.code, _format_http_error(exc)) from exc
    except URLError as exc:
        raise PlatformClientError(f"Could not reach the platform: {exc.reason}") from exc
    except TimeoutError as exc:
        raise PlatformClientError("Timed out while connecting to the platform.") from exc

    if not response_body.strip():
        return {}
    try:
        data = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise PlatformClientError("Platform returned a non-JSON response.") from exc
    if not isinstance(data, dict):
        raise PlatformClientError("Platform returned an unexpected response.")
    return data


def _format_http_error(exc: HTTPError) -> str:
    raw_body = (exc.read() if exc.fp is not None else b"").decode(
        "utf-8",
        errors="replace",
    ).strip()
    if raw_body:
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if detail:
                return str(detail)
    return f"Platform request failed with HTTP {exc.code}."


def _looks_local(value: str) -> bool:
    host = value.split("/", 1)[0].split(":", 1)[0].lower()
    return host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local")


def _api_fallback_url(base_url: str) -> str | None:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1"):
        return urlunparse((parsed.scheme, parsed.netloc, path[: -len("/v1")], "", "", ""))
    if path == "/api" or path.endswith("/api"):
        return None
    fallback_path = f"{path}/api" if path else "/api"
    return urlunparse((parsed.scheme, parsed.netloc, fallback_path, "", "", ""))


def _canonical_platform_api_url(value: str) -> str:
    parsed = urlparse(value)
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1"):
        path = path[: -len("/v1")]
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
