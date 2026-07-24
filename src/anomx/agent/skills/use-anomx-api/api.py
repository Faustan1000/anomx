"""Small stdlib helper for calling a connected Anomx Platform API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT_ONLY_PATHS = frozenset({"/docs", "/openapi.json"})


def call_api(
    method: str,
    path: str,
    *,
    query: dict[str, Any] | None = None,
    body: Any | None = None,
    output_name: str = "",
    timeout: float = 30,
) -> dict[str, Any]:
    base_url = _api_base_url(_required_env("ANOMX_PLATFORM_API_URL", "ANOMX_PLATFORM_URL"))
    token = _required_env("ANOMX_PLATFORM_API_KEY", "ANOMX_PLATFORM_TOKEN", "ANOMX_API_KEY")
    responses_dir = Path(
        os.environ.get("ANOMX_RESPONSES_DIR")
        or Path.home() / ".anomx" / "responses"
    )
    normalized_method = method.strip().upper()
    url = _build_url(base_url, path, query)
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "anomx-agent-skill",
    }
    data = None
    if body is not None and normalized_method != "GET":
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=normalized_method)
    status_code = 0
    content_type = ""
    raw = b""
    error = ""
    try:
        with urlopen(request, timeout=timeout) as response:
            status_code = int(getattr(response, "status", 200))
            content_type = str(response.headers.get("content-type", ""))
            raw = response.read()
    except HTTPError as exc:
        status_code = int(exc.code)
        content_type = str(exc.headers.get("content-type", "")) if exc.headers else ""
        raw = exc.read()
        error = str(exc)
    except (OSError, TimeoutError, URLError) as exc:
        error = str(exc)

    payload, parsed_as_json = _decode_payload(raw, content_type)
    response_path = _write_response(responses_dir, output_name=output_name, payload=payload)
    result = {
        "ok": 200 <= status_code < 300,
        "status_code": status_code,
        "method": normalized_method,
        "url": url,
        "content_type": content_type,
        "length_bytes": len(raw),
        "parsed_as_json": parsed_as_json,
        "result_count": _result_count(payload),
        "response_path": str(response_path),
    }
    if error:
        result["error"] = error
    return result


def _required_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    joined = ", ".join(names)
    raise RuntimeError(f"Missing platform API environment variable. Expected one of: {joined}")


def _build_url(base_url: str, path: str, query: dict[str, Any] | None) -> str:
    root_path = path if path.startswith("/") else f"/{path}"
    if path.startswith(("http://", "https://")):
        url = path
    elif root_path in ROOT_ONLY_PATHS:
        parsed = urlparse(base_url)
        origin = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
        url = urljoin(f"{origin.rstrip('/')}/", root_path.lstrip("/"))
    else:
        url = urljoin(f"{base_url.rstrip('/')}/", path.lstrip("/"))
    if query:
        items = []
        for key, value in query.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                items.extend((key, item) for item in value if item is not None)
            else:
                items.append((key, value))
        if items:
            url = f"{url}{'&' if '?' in url else '?'}{urlencode(items, doseq=True)}"
    return url


def _api_base_url(value: str) -> str:
    parsed = urlparse(value.strip().rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1"):
        normalized_path = path[: -len("/v1")]
    elif path.endswith("/api"):
        normalized_path = path
    else:
        normalized_path = f"{path}/api" if path else "/api"
    return urlunparse((parsed.scheme, parsed.netloc, normalized_path, "", "", ""))


def _decode_payload(raw: bytes, content_type: str) -> tuple[Any, bool]:
    text = raw.decode("utf-8", errors="replace")
    if "json" in content_type.lower() or text.strip().startswith(("{", "[")):
        try:
            return json.loads(text), True
        except json.JSONDecodeError:
            pass
    return {"content": text}, False


def _write_response(responses_dir: Path, *, output_name: str, payload: Any) -> Path:
    responses_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_output_name(output_name) or f"anomx-api-{uuid4().hex[:12]}"
    path = responses_dir / f"{safe_name}.json"
    counter = 2
    while path.exists():
        path = responses_dir / f"{safe_name}-{counter}.json"
        counter += 1
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _safe_output_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in value.strip().lower()
    ).strip("-_")
    return cleaned[:80]


def _result_count(payload: Any) -> int | None:
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, dict):
        return None
    for key in ("results", "items", "objects", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    count = payload.get("count")
    return count if isinstance(count, int) else None


def _json_argument(value: str) -> Any:
    if not value:
        return None
    return json.loads(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Call the connected Anomx Platform API.")
    parser.add_argument("method", choices=("GET", "POST", "PUT", "PATCH", "DELETE"))
    parser.add_argument("path", help="API path, for example /objects or /data/channels")
    parser.add_argument("--query", default="", help="JSON object of query parameters")
    parser.add_argument("--body", default="", help="JSON request body")
    parser.add_argument("--output-name", default="", help="Optional response filename stem")
    args = parser.parse_args(argv)
    result = call_api(
        args.method,
        args.path,
        query=_json_argument(args.query),
        body=_json_argument(args.body),
        output_name=args.output_name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
