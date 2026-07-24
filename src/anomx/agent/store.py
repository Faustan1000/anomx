"""Persistent storage for the Anomx CLI agent.

The CLI intentionally keeps its state in a small, inspectable home directory,
similar in spirit to Codex's local CLI state:

```
~/.anomx/
  config.toml
  auth.json
  brain/YYYYMMDD_<id>.anomx
  responses/*.json
  skills/<command>/README.md
  session_index.jsonl
  sessions/YYYY/MM/DD/rollout-<timestamp>-<id>.jsonl
```
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tomllib
import traceback
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from anomx.agent.base.agents import AgentKind
from anomx.agent.helpers.debug import SessionDebugLogger
from anomx.agent.helpers.mode import AgentMode
from anomx.agent.helpers.utils import parse_agent_kind, utc_now_iso

DEFAULT_HOME_NAME = ".anomx"
ANOMX_HOME_ENV = "ANOMX_HOME"


@dataclass(frozen=True)
class ProviderOption:
    """AI backend option exposed by the CLI onboarding flow."""

    key: str
    label: str
    models: tuple[str, ...]
    allow_custom_model: bool = False
    connect_hint: str = "Connect to this AI backend"
    requires_api_key: bool = True


@dataclass(frozen=True)
class ModelMetadata:
    """Model information used for selection and context tracking."""

    model: str
    display_name: str
    context_window: int | None
    max_output_tokens: int | None


@dataclass(frozen=True)
class ThinkingIntensityOption:
    """Provider/model-specific thinking intensity selection."""

    value: str
    label: str
    detail: str


@dataclass(frozen=True)
class SessionRecord:
    """A stored CLI session transcript."""

    session_id: str
    path: Path
    created_at: str
    updated_at: str
    cwd: str
    provider: str
    model: str
    title: str
    message_count: int = 0
    unread: bool = False
    last_user_at: str = ""
    mode: AgentMode = AgentMode.CONFIRM
    agent_kind: AgentKind = AgentKind.STANDARD


@dataclass(frozen=True)
class ProjectRecord:
    """A locally known Anomx project folder."""

    path: Path
    name: str
    created_at: str = ""
    updated_at: str = ""
    sandbox_hash: str = ""


AI_PROVIDERS: tuple[ProviderOption, ...] = (
    ProviderOption(
        "desy",
        "DESY Assistant",
        ("desy-assistant", "reasoning", "coding"),
        allow_custom_model=True,
        connect_hint="Connect to LLMs hosted on the Maxwell cluster",
    ),
    ProviderOption(
        "blablador",
        "JSC Blablador",
        ("alias-code", "alias-fast", "alias-large", "alias-huge"),
        allow_custom_model=True,
        connect_hint="Connect to LLMs hosted on the Jülich Supercomputing Centre",
    ),
    ProviderOption(
        "anthropic",
        "Anthropic",
        (
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ),
        allow_custom_model=True,
        connect_hint="Connect to Claude models hosted by Anthropic",
    ),
    ProviderOption(
        "openai",
        "OpenAI",
        ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini"),
        allow_custom_model=True,
        connect_hint="Connect to GPT models hosted by OpenAI",
    ),
    ProviderOption(
        "ollama",
        "Ollama",
        ("qwen3.6", "qwen3-coder:30b", "qwen2.5-coder:32b"),
        allow_custom_model=True,
        connect_hint="Connect to LLMs that you run on your machine via Ollama",
        requires_api_key=False,
    ),
)

AI_PROVIDER_KEYS = tuple(provider.key for provider in AI_PROVIDERS)

THINKING_INTENSITY_AUTO = "auto"
THINKING_INTENSITY_OPTIONS: dict[str, ThinkingIntensityOption] = {
    THINKING_INTENSITY_AUTO: ThinkingIntensityOption(
        THINKING_INTENSITY_AUTO,
        "Provider default",
        "Use the backend default for this model",
    ),
    "minimal": ThinkingIntensityOption(
        "minimal",
        "Minimal",
        "Fastest OpenAI reasoning mode",
    ),
    "low": ThinkingIntensityOption(
        "low",
        "Low",
        "Prioritize speed and lower token use",
    ),
    "medium": ThinkingIntensityOption(
        "medium",
        "Medium",
        "Balance reasoning depth and latency",
    ),
    "high": ThinkingIntensityOption(
        "high",
        "High",
        "Use deeper reasoning for harder tasks",
    ),
    "xhigh": ThinkingIntensityOption(
        "xhigh",
        "Extra high",
        "Claude long-horizon agentic work",
    ),
    "max": ThinkingIntensityOption(
        "max",
        "Max",
        "Claude maximum capability mode",
    ),
}

MODEL_METADATA: dict[str, ModelMetadata] = {
    "gpt-5.5": ModelMetadata("gpt-5.5", "GPT-5.5", 1_000_000, 128_000),
    "gpt-5.4": ModelMetadata("gpt-5.4", "GPT-5.4", 1_000_000, 128_000),
    "gpt-5.4-mini": ModelMetadata("gpt-5.4-mini", "GPT-5.4 mini", 400_000, 128_000),
    "claude-opus-4-8": ModelMetadata(
        "claude-opus-4-8",
        "Claude Opus 4.8",
        1_000_000,
        128_000,
    ),
    "claude-sonnet-4-6": ModelMetadata(
        "claude-sonnet-4-6",
        "Claude Sonnet 4.6",
        1_000_000,
        64_000,
    ),
    "claude-haiku-4-5-20251001": ModelMetadata(
        "claude-haiku-4-5-20251001",
        "Claude Haiku 4.5",
        200_000,
        64_000,
    ),
    "qwen3.6": ModelMetadata("qwen3.6", "Qwen 3.6", None, None),
    "qwen3-coder:30b": ModelMetadata("qwen3-coder:30b", "Qwen3 Coder 30B", None, None),
    "qwen2.5-coder:32b": ModelMetadata(
        "qwen2.5-coder:32b",
        "Qwen2.5 Coder 32B",
        None,
        None,
    ),
}

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "onboarding_complete": False,
    "provider": "openai",
    "model": "gpt-5.5",
    "user_name": "",
    "thinking_intensity": THINKING_INTENSITY_AUTO,
    "max_output_tokens": 8192,
    "agent_mode": AgentMode.CONFIRM.value,
    "agent_kind": AgentKind.STANDARD.value,
    "require_trusted_repo": True,
    "history_persistence": "save_all",
    "debug_mode": False,
    "debug_full_session_logs": False,
    "debug_full_session_logs_path": None,
    "last_session_id": None,
    "platform_url": None,
    "platform_last_url": None,
    "platform_last_email": None,
    "skipped_version": "",
    "sandbox_enabled": False,
    "sandbox_system": "docker",
    "sandbox_method": "mount",
    "sandbox_cpu_limit": "2",
    "sandbox_ram_limit": "4g",
    "sandbox_hd_limit": "10g",
    "sandbox_strategy": "stop",
    "projects": {},
    "global_allowed_commands": [],
    "global_rejected_commands": [],
    "connected_backends": [],
}

CONFIG_SCALAR_FIELDS = (
    "schema_version",
    "onboarding_complete",
    "provider",
    "model",
    "user_name",
    "thinking_intensity",
    "max_output_tokens",
    "agent_mode",
    "agent_kind",
    "history_persistence",
    "debug_mode",
    "debug_full_session_logs",
    "debug_full_session_logs_path",
    "last_session_id",
    "platform_url",
    "platform_last_url",
    "platform_last_email",
    "skipped_version",
    "sandbox_enabled",
    "sandbox_system",
    "sandbox_method",
    "sandbox_cpu_limit",
    "sandbox_ram_limit",
    "sandbox_hd_limit",
    "sandbox_strategy",
)


def default_config() -> dict[str, Any]:
    """Return a mutable default config without sharing nested values."""

    config = DEFAULT_CONFIG.copy()
    config["projects"] = {}
    return config


def resolve_anomx_home(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the Anomx home directory.

    `ANOMX_HOME` mirrors Codex-style home overrides and is useful for tests,
    isolated development runs, and future packaged desktop environments.
    """

    source = os.environ if env is None else env
    configured_home = source.get(ANOMX_HOME_ENV)
    if configured_home:
        return Path(configured_home).expanduser()
    return Path.home() / DEFAULT_HOME_NAME


def provider_by_key(provider_key: str) -> ProviderOption | None:
    """Return a configured provider option by key."""

    return next((provider for provider in AI_PROVIDERS if provider.key == provider_key), None)


def model_metadata(model: str) -> ModelMetadata | None:
    """Return metadata for a configured model."""

    return MODEL_METADATA.get(model)


def model_context_window(model: str) -> int | None:
    """Return the model context window in tokens when known."""

    metadata = model_metadata(model)
    return None if metadata is None else metadata.context_window


def model_detail(model: str) -> str:
    """Return a compact model detail string for menus."""

    metadata = model_metadata(model)
    if metadata is None:
        return "custom context"
    parts: list[str] = []
    if metadata.context_window is not None:
        parts.append(f"{_format_token_count(metadata.context_window)} context")
    if metadata.max_output_tokens is not None:
        parts.append(f"{_format_token_count(metadata.max_output_tokens)} max output")
    return " · ".join(parts) if parts else "custom context"


def normalize_thinking_intensity(value: object) -> str:
    """Return a known thinking intensity value, falling back to provider defaults."""

    intensity = str(value or THINKING_INTENSITY_AUTO).strip().lower()
    return intensity if intensity in THINKING_INTENSITY_OPTIONS else THINKING_INTENSITY_AUTO


def thinking_intensity_options(
    provider_key: str,
    model: str,
) -> tuple[ThinkingIntensityOption, ...]:
    """Return supported thinking intensity options for a provider/model pair."""

    if provider_key == "openai" and model.startswith("gpt-5"):
        return tuple(
            THINKING_INTENSITY_OPTIONS[value]
            for value in (THINKING_INTENSITY_AUTO, "minimal", "low", "medium", "high")
        )
    if provider_key == "anthropic":
        if model == "claude-opus-4-8":
            return tuple(
                THINKING_INTENSITY_OPTIONS[value]
                for value in (THINKING_INTENSITY_AUTO, "low", "medium", "high", "xhigh", "max")
            )
        if model in {"claude-opus-4-6", "claude-sonnet-4-6"}:
            return tuple(
                THINKING_INTENSITY_OPTIONS[value]
                for value in (THINKING_INTENSITY_AUTO, "low", "medium", "high", "max")
            )
    return ()


def thinking_intensity_supported(provider_key: str, model: str) -> bool:
    """Return whether the selected provider/model exposes thinking intensity."""

    return bool(thinking_intensity_options(provider_key, model))


def _format_token_count(tokens: int) -> str:
    if tokens >= 1_000_000 and tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}M"
    if tokens >= 1_000 and tokens % 1_000 == 0:
        return f"{tokens // 1_000}K"
    return f"{tokens:,}"


class AnomxHome:
    """Read and write the persistent Anomx CLI home directory."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = resolve_anomx_home() if root is None else root.expanduser()

    @property
    def config_path(self) -> Path:
        """Return the global CLI config path."""

        return self.root / "config.toml"

    @property
    def auth_path(self) -> Path:
        """Return the auth and API-key metadata path."""

        return self.root / "auth.json"

    @property
    def trusted_repos_path(self) -> Path:
        """Return the legacy trusted repository registry path."""

        return self.root / "trusted_repos.json"

    @property
    def session_index_path(self) -> Path:
        """Return the append-only session index path."""

        return self.root / "session_index.jsonl"

    @property
    def usage_path(self) -> Path:
        """Return the compact local CLI usage counter path."""

        return self.root / "usage.json"

    @property
    def sessions_dir(self) -> Path:
        """Return the session transcript root."""

        return self.root / "sessions"

    @property
    def skills_dir(self) -> Path:
        """Return the global user-created skills directory."""

        return self.root / "skills"

    @property
    def responses_dir(self) -> Path:
        """Return the directory where large tool responses are written."""

        return self.root / "responses"

    @property
    def brain_dir(self) -> Path:
        """Return the persistent memory directory."""

        return self.root / "brain"

    @property
    def instructions_dir(self) -> Path:
        """Return the custom instructions directory."""

        return self.root / "instructions"

    @property
    def search_dir(self) -> Path:
        """Return the persistent search index root."""

        return self.root / "search"

    @property
    def debug_logger(self) -> SessionDebugLogger:
        """Return a cached :class:`SessionDebugLogger` for this home."""
        if not hasattr(self, "_debug_logger_cache"):
            config = self.load_config()
            location = self.debug_location(config)
            self._debug_logger_cache = SessionDebugLogger(location)
        return self._debug_logger_cache

    def debug_location(self, config: Mapping[str, Any] | None = None) -> Path:
        """Return the root directory for debug output."""
        cfg = self.load_config() if config is None else config
        configured = self._normalized_debug_logs_path_config(
            cfg.get("debug_full_session_logs_path")
        )
        return Path(configured).expanduser() if configured else self.root

    def ensure(self) -> None:
        """Create required home directories."""

        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.brain_dir.mkdir(parents=True, exist_ok=True)
        self.instructions_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.search_dir.mkdir(parents=True, exist_ok=True)

    def has_config(self) -> bool:
        """Return whether the home directory already has a persisted config."""

        return self.config_path.exists()

    def load_config(self) -> dict[str, Any]:
        """Load the CLI config, applying defaults for missing fields."""

        config = default_config()
        config.update(self._read_toml_object(self.config_path))
        config["agent_mode"] = AgentMode.parse(config.get("agent_mode")).value
        config["agent_kind"] = parse_agent_kind(config.get("agent_kind")).value
        config["thinking_intensity"] = normalize_thinking_intensity(
            config.get("thinking_intensity")
        )
        config["history_persistence"] = "save_all"
        config["require_trusted_repo"] = True
        config["debug_mode"] = bool(config.get("debug_mode"))
        config["debug_full_session_logs"] = bool(config.get("debug_full_session_logs"))
        config["debug_full_session_logs_path"] = self._normalized_debug_logs_path_config(
            config.get("debug_full_session_logs_path")
        )
        projects = config.get("projects")
        if not isinstance(projects, dict):
            config["projects"] = {}
        return config

    def save_config(self, config: Mapping[str, Any]) -> None:
        """Save the CLI config."""

        self.ensure()
        merged = default_config()
        merged.update(dict(config))
        merged["agent_mode"] = AgentMode.parse(merged.get("agent_mode")).value
        merged["agent_kind"] = parse_agent_kind(merged.get("agent_kind")).value
        merged["thinking_intensity"] = normalize_thinking_intensity(
            merged.get("thinking_intensity")
        )
        merged["history_persistence"] = "save_all"
        merged["require_trusted_repo"] = True
        merged["debug_mode"] = bool(merged.get("debug_mode"))
        merged["debug_full_session_logs"] = bool(merged.get("debug_full_session_logs"))
        merged["debug_full_session_logs_path"] = self._normalized_debug_logs_path_config(
            merged.get("debug_full_session_logs_path")
        )
        self._write_config_toml(self.config_path, merged)

    def debug_mode_enabled(self, config: Mapping[str, Any] | None = None) -> bool:
        """Return whether debug logging is active."""

        active_config = self.load_config() if config is None else config
        return bool(active_config.get("debug_mode"))

    def full_session_logs_enabled(self, config: Mapping[str, Any] | None = None) -> bool:
        """Return whether full backend request logging is active."""

        active_config = self.load_config() if config is None else config
        return bool(active_config.get("debug_mode")) and bool(
            active_config.get("debug_full_session_logs")
        )

    def full_session_logs_dir(self, config: Mapping[str, Any] | None = None) -> Path:
        """Return the directory for full backend request logs."""

        active_config = self.load_config() if config is None else config
        configured = self._normalized_debug_logs_path_config(
            active_config.get("debug_full_session_logs_path")
        )
        if configured is None:
            return self.root
        return Path(configured).expanduser()

    @property
    def crash_logs_dir(self) -> Path:
        """Return the debug crash-log directory."""
        return self.root / "crash_logs"

    def write_crash_log(
        self,
        error: BaseException,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Write a crash log when debug mode is active."""
        config = self.load_config()
        if not self.debug_mode_enabled(config):
            return None
        self.crash_logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = self._debug_timestamp()
        path = self.crash_logs_dir / f"crash-{timestamp}-{uuid4().hex[:8]}.txt"
        lines = [
            "Anomx crash log",
            f"timestamp: {utc_now_iso()}",
            f"error: {type(error).__name__}: {error}",
            "",
        ]
        if context:
            lines.extend(
                [
                    "Context:",
                    json.dumps(dict(context), indent=2, sort_keys=True, default=str),
                    "",
                ]
            )
        lines.extend(
            [
                "Traceback:",
                "".join(
                    traceback.format_exception(
                        type(error),
                        error,
                        error.__traceback__,
                    )
                ).rstrip(),
                "",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _normalized_debug_logs_path_config(self, value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _debug_timestamp() -> str:
        return utc_now_iso().replace(":", "").replace("-", "")

    def load_auth(self) -> dict[str, Any]:
        """Load provider credentials metadata."""

        default_auth: dict[str, Any] = {"schema_version": 1, "api_keys": {}, "platform": {}}
        auth = default_auth.copy()
        auth.update(self._read_json_object(self.auth_path, default={}))
        api_keys = auth.get("api_keys")
        if not isinstance(api_keys, dict):
            auth["api_keys"] = {}
        platform = auth.get("platform")
        if not isinstance(platform, dict):
            auth["platform"] = {}
        return auth

    def save_auth(self, auth: Mapping[str, Any]) -> None:
        """Save provider credentials metadata with owner-only file permissions."""

        self.ensure()
        payload = {"schema_version": 1, "api_keys": {}, "platform": {}}
        payload.update(dict(auth))
        self._write_json_object(self.auth_path, payload, mode=0o600)

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Store an API key for a provider."""

        auth = self.load_auth()
        api_keys = cast(dict[str, str], auth["api_keys"])
        api_keys[provider] = api_key
        self.save_auth(auth)

    def has_api_key(self, provider: str) -> bool:
        """Return whether an API key is configured for a provider."""

        auth = self.load_auth()
        api_keys = cast(dict[str, str], auth["api_keys"])
        return bool(api_keys.get(provider))

    def remove_api_key(self, provider: str) -> None:
        """Remove a stored API key for a provider."""

        auth = self.load_auth()
        api_keys = cast(dict[str, str], auth["api_keys"])
        if api_keys.pop(provider, None) is not None:
            self.save_auth(auth)

    def _explicit_connected_backends(self) -> list[str]:
        """Return the provider keys the user explicitly connected."""

        config = self.load_config()
        raw = config.get("connected_backends", [])
        if not isinstance(raw, list):
            return []
        return [str(entry).strip() for entry in raw if str(entry).strip()]

    def is_backend_connected(self, provider: str) -> bool:
        """Return whether a backend is connected and usable.

        Credential-based backends are connected once an API key is stored.
        Keyless backends (such as Ollama) are connected via an explicit toggle.
        """

        option = provider_by_key(provider)
        if option is not None and not option.requires_api_key:
            if provider in self._explicit_connected_backends():
                return True
            # Migrate a legacy single-provider selection for keyless backends.
            config = self.load_config()
            return bool(config.get("onboarding_complete")) and (
                str(config.get("provider", "")) == provider
            )
        return self.has_api_key(provider)

    def connected_backend_keys(self) -> list[str]:
        """Return the provider keys for every connected backend, in catalog order."""

        return [
            provider.key
            for provider in AI_PROVIDERS
            if self.is_backend_connected(provider.key)
        ]

    def set_backend_connected(self, provider: str, connected: bool) -> None:
        """Connect or disconnect a keyless backend (e.g. Ollama).

        Credential-based backends are connected by storing an API key and
        disconnected by removing it; this only tracks the explicit toggle.
        """

        config = self.load_config()
        current = config.get("connected_backends", [])
        keys = [str(entry).strip() for entry in current if str(entry).strip()] if isinstance(current, list) else []
        if connected and provider not in keys:
            keys.append(provider)
        elif not connected and provider in keys:
            keys = [key for key in keys if key != provider]
        config["connected_backends"] = keys
        self.save_config(config)

    def set_platform_connection(
        self,
        *,
        url: str,
        token: str,
        user_email: str = "",
        organization_url: str = "",
        hostname: str = "",
    ) -> None:
        """Store the connected Anomx Platform endpoint and bearer token."""

        config = self.load_config()
        config["platform_url"] = url
        self.save_config(config)

        auth = self.load_auth()
        auth["platform"] = {
            "token": token,
            "user_email": user_email,
            "organization_url": organization_url,
            "hostname": hostname,
        }
        self.save_auth(auth)

    def set_platform_form_defaults(self, *, url: str, email: str) -> None:
        """Persist the non-secret platform connection form defaults."""

        config = self.load_config()
        config["platform_last_url"] = url
        config["platform_last_email"] = email
        self.save_config(config)

    def clear_platform_connection(self) -> None:
        """Remove the connected Anomx Platform endpoint and bearer token."""

        config = self.load_config()
        config["platform_url"] = None
        self.save_config(config)

        auth = self.load_auth()
        auth["platform"] = {}
        self.save_auth(auth)

    def platform_connection(self) -> dict[str, str] | None:
        """Return the configured platform connection when a URL and token exist."""

        config = self.load_config()
        url = str(config.get("platform_url") or "").strip()
        auth = self.load_auth()
        platform = auth.get("platform")
        if not isinstance(platform, dict):
            return None
        token = str(platform.get("token") or "").strip()
        if not url or not token:
            return None
        return {
            "url": url,
            "token": token,
            "user_email": str(platform.get("user_email") or ""),
            "organization_url": str(platform.get("organization_url") or ""),
            "hostname": str(platform.get("hostname") or ""),
        }

    def has_platform_connection(self) -> bool:
        """Return whether the CLI has a platform URL and bearer token."""

        return self.platform_connection() is not None

    def load_global_allowed_commands(self) -> list[str]:
        """Return the list of globally allowed command allowance keys."""
        config = self.load_config()
        raw = config.get("global_allowed_commands", [])
        return list(raw) if isinstance(raw, list) else []

    def load_global_rejected_commands(self) -> list[str]:
        """Return the list of globally rejected command allowance keys."""
        config = self.load_config()
        raw = config.get("global_rejected_commands", [])
        return list(raw) if isinstance(raw, list) else []

    def add_global_allowed_command(self, allowance_key: str) -> None:
        """Persist a globally allowed command allowance key."""
        config = self.load_config()
        allowed = self.load_global_allowed_commands()
        if allowance_key not in allowed:
            allowed.append(allowance_key)
        config["global_allowed_commands"] = allowed
        self.save_config(config)

    def add_global_rejected_command(self, allowance_key: str) -> None:
        """Persist a globally rejected command allowance key."""
        config = self.load_config()
        rejected = self.load_global_rejected_commands()
        if allowance_key not in rejected:
            rejected.append(allowance_key)
        config["global_rejected_commands"] = rejected
        self.save_config(config)

    def remove_global_allowed_command(self, allowance_key: str) -> None:
        """Remove a globally allowed command allowance key."""
        config = self.load_config()
        allowed = self.load_global_allowed_commands()
        if allowance_key in allowed:
            allowed.remove(allowance_key)
        config["global_allowed_commands"] = allowed
        self.save_config(config)

    def remove_global_rejected_command(self, allowance_key: str) -> None:
        """Remove a globally rejected command allowance key."""
        config = self.load_config()
        rejected = self.load_global_rejected_commands()
        if allowance_key in rejected:
            rejected.remove(allowance_key)
        config["global_rejected_commands"] = rejected
        self.save_config(config)

    def clear_sessions(self, keep_session_path: Path | None = None) -> None:
        """Delete stored session history and rebuild session metadata."""

        keep_path = (
            keep_session_path.expanduser().resolve() if keep_session_path is not None else None
        )
        records = self.list_sessions(limit=None)
        remaining_records: list[SessionRecord] = []
        for record in records:
            record_path = record.path.expanduser().resolve()
            if keep_path is not None and record_path == keep_path:
                remaining_records.append(record)
                continue
            record.path.unlink(missing_ok=True)
            self._delete_worker_sessions_for_record(record)

        if self.sessions_dir.exists():
            for directory in sorted(self.sessions_dir.rglob("*"), reverse=True):
                if directory.is_dir():
                    try:
                        directory.rmdir()
                    except OSError:
                        continue
            if keep_path is None:
                shutil.rmtree(self.sessions_dir, ignore_errors=True)
        self.session_index_path.unlink(missing_ok=True)
        for record in reversed(remaining_records):
            self._append_session_index(record)

        config = self.load_config()
        config["last_session_id"] = remaining_records[0].session_id if remaining_records else None
        self.save_config(config)

    def delete_session(self, session_path: Path) -> bool:
        """Delete a single stored session and rebuild session metadata."""

        delete_path = session_path.expanduser().resolve()
        records = self.list_sessions(limit=None)
        remaining_records: list[SessionRecord] = []
        deleted = False
        for record in records:
            record_path = record.path.expanduser().resolve()
            if record_path == delete_path:
                deleted = True
                record.path.unlink(missing_ok=True)
                self._delete_worker_sessions_for_record(record)
                continue
            remaining_records.append(record)

        if not deleted:
            return False

        if self.sessions_dir.exists():
            for directory in sorted(self.sessions_dir.rglob("*"), reverse=True):
                if directory.is_dir():
                    try:
                        directory.rmdir()
                    except OSError:
                        continue
            if not remaining_records:
                shutil.rmtree(self.sessions_dir, ignore_errors=True)

        self.session_index_path.unlink(missing_ok=True)
        for record in reversed(remaining_records):
            self._append_session_index(record)

        remaining_session_ids = {record.session_id for record in remaining_records}
        config = self.load_config()
        if config.get("last_session_id") not in remaining_session_ids:
            config["last_session_id"] = (
                remaining_records[0].session_id if remaining_records else None
            )
        self.save_config(config)
        return True

    def is_repo_trusted(self, repo_path: Path) -> bool:
        """Return whether the repository path has already passed access check."""

        config = self.load_config()
        projects = cast(dict[str, Any], config["projects"])
        repo_key = self._repo_key(repo_path)
        repo_entry = projects.get(repo_key)
        if not isinstance(repo_entry, dict):
            return self._is_repo_trusted_in_legacy_file(repo_key)
        return repo_entry.get("trust_level") == "trusted"

    def trust_repo(self, repo_path: Path) -> None:
        """Mark a repository path as trusted."""

        self.ensure()
        config = self.load_config()
        projects = cast(dict[str, Any], config["projects"])
        repo_key = self._repo_key(repo_path)
        existing = projects.get(repo_key)
        repo_entry = existing.copy() if isinstance(existing, dict) else {}
        repo_entry["trust_level"] = "trusted"
        repo_entry["trusted_at"] = utc_now_iso()
        projects[repo_key] = repo_entry
        self.save_config(config)

    def project_for_path(self, project_path: Path) -> ProjectRecord | None:
        """Return stored project metadata for a folder, if present."""

        config = self.load_config()
        projects = cast(dict[str, Any], config["projects"])
        project_key = self._repo_key(project_path)
        project_entry = projects.get(project_key)
        if not isinstance(project_entry, dict):
            return None
        name = str(project_entry.get("name") or "").strip()
        if not name:
            return None
        return ProjectRecord(
            path=Path(project_key),
            name=name,
            created_at=str(project_entry.get("created_at") or ""),
            updated_at=str(project_entry.get("updated_at") or ""),
            sandbox_hash=str(project_entry.get("sandbox_hash") or ""),
        )

    def save_project(self, project_path: Path, name: str) -> ProjectRecord:
        """Persist local project metadata without changing trust settings."""

        self.ensure()
        config = self.load_config()
        projects = cast(dict[str, Any], config["projects"])
        project_key = self._repo_key(project_path)
        existing = projects.get(project_key)
        project_entry = existing.copy() if isinstance(existing, dict) else {}
        now = utc_now_iso()
        project_entry.setdefault("created_at", now)
        project_entry["updated_at"] = now
        project_entry["name"] = name.strip() or Path(project_key).name or "Anomx Project"
        project_entry["path"] = project_key
        if not project_entry.get("sandbox_hash"):
            project_entry["sandbox_hash"] = self._sandbox_hash(project_path)
        projects[project_key] = project_entry
        self.save_config(config)
        return ProjectRecord(
            path=Path(project_key),
            name=str(project_entry["name"]),
            created_at=str(project_entry.get("created_at") or ""),
            updated_at=str(project_entry.get("updated_at") or ""),
            sandbox_hash=str(project_entry.get("sandbox_hash") or ""),
        )

    @staticmethod
    def _sandbox_hash(project_path: Path) -> str:
        import hashlib
        raw = str(project_path.resolve()).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:6]

    def create_session(
        self,
        cwd: Path,
        provider: str,
        model: str,
        mode: AgentMode | str = AgentMode.CONFIRM,
        agent_kind: AgentKind | str = AgentKind.STANDARD,
    ) -> SessionRecord:
        """Create an empty session transcript and index entry."""

        self.ensure()
        now = utc_now_iso()
        agent_mode = AgentMode.parse(mode)
        parsed_agent_kind = parse_agent_kind(agent_kind)
        session_id = uuid4().hex
        date_parts = datetime.now(tz=UTC).strftime("%Y/%m/%d")
        session_dir = self.sessions_dir / date_parts
        session_dir.mkdir(parents=True, exist_ok=True)
        filename_timestamp = now.removesuffix("Z").replace(":", "-")
        session_path = session_dir / f"rollout-{filename_timestamp}-{session_id}.jsonl"
        record = SessionRecord(
            session_id=session_id,
            path=session_path,
            created_at=now,
            updated_at=now,
            cwd=str(cwd.expanduser().resolve()),
            provider=provider,
            model=model,
            title="New session",
            message_count=0,
            unread=False,
            last_user_at=now,
            mode=agent_mode,
            agent_kind=parsed_agent_kind,
        )
        metadata = {
            "id": record.session_id,
            "session_id": record.session_id,
            "timestamp": record.created_at,
            "created_at": record.created_at,
            "cwd": record.cwd,
            "originator": "anomx_cli",
            "source": "cli",
            "model_provider": record.provider,
            "provider": record.provider,
            "model": record.model,
            "title": record.title,
            "unread": record.unread,
            "agent_mode": record.mode.value,
            "agent_kind": record.agent_kind.value,
        }
        self._append_jsonl(
            session_path,
            {"timestamp": now, "type": "session_meta", "payload": metadata},
        )
        self._append_session_index(record)
        config = self.load_config()
        config["last_session_id"] = record.session_id
        self.save_config(config)
        return record

    def append_worker_session_prompt(
        self,
        *,
        parent_session_path: Path | None,
        worker_name: str,
        worker_id: str,
        prompt: str,
    ) -> Path:
        """Append a prompt to a worker sub-session and return its transcript path."""

        self.ensure()
        session_path = self.worker_session_path(
            parent_session_path=parent_session_path,
            worker_name=worker_name,
            worker_id=worker_id,
        )
        if not session_path.exists():
            parent_record = (
                self._read_session_record(parent_session_path)
                if parent_session_path is not None
                else None
            )
            now = utc_now_iso()
            self._append_jsonl(
                session_path,
                {
                    "timestamp": now,
                    "type": "worker_session_meta",
                    "payload": {
                        "created_at": now,
                        "originator": "anomx_cli_worker",
                        "parent_session_id": (
                            parent_record.session_id if parent_record is not None else ""
                        ),
                        "parent_session_path": (
                            str(parent_session_path) if parent_session_path is not None else ""
                        ),
                        "worker_id": worker_id,
                        "worker_name": worker_name,
                    },
                },
            )
        self.append_session_event(session_path, "user_message", {"message": prompt})
        return session_path

    def append_subagent_session_prompt(
        self,
        *,
        parent_session_path: Path | None,
        subagent_name: str,
        subagent_id: str,
        subagent_kind: str,
        prompt: str,
    ) -> Path:
        """Append a prompt to a subagent sub-session and return its transcript path."""

        self.ensure()
        session_path = self.subagent_session_path(
            parent_session_path=parent_session_path,
            subagent_name=subagent_name,
            subagent_id=subagent_id,
        )
        if not session_path.exists():
            parent_record = (
                self._read_session_record(parent_session_path)
                if parent_session_path is not None
                else None
            )
            now = utc_now_iso()
            session_path.parent.mkdir(parents=True, exist_ok=True)
            self._append_jsonl(
                session_path,
                {
                    "timestamp": now,
                    "type": "subagent_session_meta",
                    "payload": {
                        "created_at": now,
                        "originator": "anomx_cli_subagent",
                        "parent_session_id": (
                            parent_record.session_id if parent_record is not None else ""
                        ),
                        "parent_session_path": (
                            str(parent_session_path) if parent_session_path is not None else ""
                        ),
                        "subagent_id": subagent_id,
                        "subagent_name": subagent_name,
                        "subagent_kind": subagent_kind,
                    },
                },
            )
        self.append_session_event(session_path, "user_message", {"message": prompt})
        return session_path

    def subagent_session_path(
        self,
        *,
        parent_session_path: Path | None,
        subagent_name: str,
        subagent_id: str,
    ) -> Path:
        """Return the subagent sub-session transcript path."""

        parent_dir = self._worker_session_parent_dir(parent_session_path)
        subagent_dir = parent_dir / self._subagent_session_folder_name(
            subagent_name,
            subagent_id,
        )
        return subagent_dir / "session.jsonl"

    def worker_session_path(
        self,
        *,
        parent_session_path: Path | None,
        worker_name: str,
        worker_id: str,
    ) -> Path:
        """Return the worker sub-session transcript path."""

        parent_dir = self._worker_session_parent_dir(parent_session_path)
        worker_dir = parent_dir / self._worker_session_folder_name(worker_name, worker_id)
        return worker_dir / "session.jsonl"

    def append_session_event(
        self,
        session_path: Path,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Append an event to a stored session transcript."""

        event_payload = {"type": event_type}
        event_payload.update(dict(payload))
        self._append_jsonl(
            session_path,
            {"timestamp": utc_now_iso(), "type": "event_msg", "payload": event_payload},
        )
        with suppress(Exception):
            self._record_cli_usage_event(session_path, event_type, payload)

    def load_cli_usage(self) -> dict[str, int]:
        """Return local daily CLI user-message counts keyed as ``YYYYMMDD``."""

        if self.usage_path.exists():
            raw_usage = self._read_json_object(self.usage_path, default={})
            return self._normalize_cli_usage(raw_usage)

        usage = self._build_cli_usage_from_sessions()
        if usage:
            self._write_json_object(self.usage_path, usage)
        return usage

    def _record_cli_usage_event(
        self,
        session_path: Path,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        if event_type not in {"user_message", "skill_invocation"}:
            return
        if payload.get("intermediate"):
            return
        if not str(payload.get("message") or "").strip():
            return
        if not self._is_root_cli_session(session_path):
            return

        had_usage_file = self.usage_path.exists()
        usage = self.load_cli_usage()
        if not had_usage_file:
            return
        day_key = datetime.now(tz=UTC).strftime("%Y%m%d")
        usage[day_key] = usage.get(day_key, 0) + 1
        self._write_json_object(self.usage_path, self._trim_cli_usage(usage))

    def _is_root_cli_session(self, session_path: Path) -> bool:
        if not session_path.exists():
            return False
        with session_path.open(encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        if not first_line:
            return False
        first_event = json.loads(first_line)
        return first_event.get("type") == "session_meta"

    def _normalize_cli_usage(self, value: Mapping[str, Any]) -> dict[str, int]:
        normalized: dict[str, int] = {}
        for raw_key, raw_count in value.items():
            key = str(raw_key).strip().replace("-", "")
            if len(key) != 8 or not key.isdigit():
                continue
            try:
                datetime.strptime(key, "%Y%m%d")
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count > 0:
                normalized[key] = count
        return self._trim_cli_usage(normalized)

    def _build_cli_usage_from_sessions(self) -> dict[str, int]:
        if not self.sessions_dir.exists():
            return {}

        usage: dict[str, int] = {}
        for session_path in self.sessions_dir.rglob("*.jsonl"):
            events = self.read_session_events(session_path)
            if not events or events[0].get("type") != "session_meta":
                continue
            for event in events:
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue
                event_type = (
                    payload.get("type")
                    if event.get("type") == "event_msg"
                    else event.get("type")
                )
                if event_type not in {"user_message", "skill_invocation"}:
                    continue
                if payload.get("intermediate"):
                    continue
                if not str(payload.get("message", "")).strip():
                    continue
                day_key = self._usage_day_key(event.get("timestamp"))
                if day_key:
                    usage[day_key] = usage.get(day_key, 0) + 1
        return self._trim_cli_usage(usage)

    @staticmethod
    def _usage_day_key(value: object) -> str:
        timestamp = str(value or "").strip()
        if not timestamp:
            return ""
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return ""
        return parsed.astimezone(UTC).strftime("%Y%m%d")

    @staticmethod
    def _trim_cli_usage(usage: Mapping[str, int], *, max_days: int = 371) -> dict[str, int]:
        return {
            key: int(usage[key])
            for key in sorted(usage)[-max_days:]
            if int(usage[key]) > 0
        }

    def update_session_title(self, session_path: Path, title: str) -> None:
        """Update the title stored in the session metadata event."""

        events = self.read_session_events(session_path)
        if not events:
            return

        first_event = events[0]
        payload = first_event.get("payload")
        if first_event.get("type") != "session_meta" or not isinstance(payload, dict):
            return

        payload["title"] = title
        tmp_path = session_path.with_suffix(f"{session_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for event in events:
                json.dump(event, handle, sort_keys=True)
                handle.write("\n")
        tmp_path.replace(session_path)

    def set_session_unread(self, session_path: Path, unread: bool) -> None:
        """Update the unread-answer flag stored in session metadata."""

        events = self.read_session_events(session_path)
        if not events:
            return

        first_event = events[0]
        payload = first_event.get("payload")
        if first_event.get("type") != "session_meta" or not isinstance(payload, dict):
            return

        if bool(payload.get("unread", False)) == unread:
            return
        payload["unread"] = unread
        tmp_path = session_path.with_suffix(f"{session_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for event in events:
                json.dump(event, handle, sort_keys=True)
                handle.write("\n")
        tmp_path.replace(session_path)

    def update_session_mode(self, session_path: Path, mode: AgentMode | str) -> None:
        """Update the execution mode stored in the session metadata event."""

        events = self.read_session_events(session_path)
        if not events:
            return

        first_event = events[0]
        payload = first_event.get("payload")
        if first_event.get("type") != "session_meta" or not isinstance(payload, dict):
            return

        agent_mode = AgentMode.parse(mode)
        if str(payload.get("agent_mode", "")) == agent_mode.value:
            return
        payload["agent_mode"] = agent_mode.value
        tmp_path = session_path.with_suffix(f"{session_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for event in events:
                json.dump(event, handle, sort_keys=True)
                handle.write("\n")
        tmp_path.replace(session_path)

    def update_session_model(self, session_path: Path, provider: str, model: str) -> None:
        """Persist the last used provider/model in the session metadata event."""

        provider = str(provider or "").strip()
        model = str(model or "").strip()
        if not provider or not model:
            return

        events = self.read_session_events(session_path)
        if not events:
            return

        first_event = events[0]
        payload = first_event.get("payload")
        if first_event.get("type") != "session_meta" or not isinstance(payload, dict):
            return

        if str(payload.get("provider", "")) == provider and str(payload.get("model", "")) == model:
            return
        payload["model_provider"] = provider
        payload["provider"] = provider
        payload["model"] = model
        tmp_path = session_path.with_suffix(f"{session_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for event in events:
                json.dump(event, handle, sort_keys=True)
                handle.write("\n")
        tmp_path.replace(session_path)

    def update_session_agent(
        self,
        session_path: Path,
        agent_kind: AgentKind | str,
        mode: AgentMode | str | None = None,
    ) -> None:
        """Update the agent kind and optional approval mode in session metadata."""

        events = self.read_session_events(session_path)
        if not events:
            return

        first_event = events[0]
        payload = first_event.get("payload")
        if first_event.get("type") != "session_meta" or not isinstance(payload, dict):
            return

        parsed_agent_kind = parse_agent_kind(agent_kind)
        changed = str(payload.get("agent_kind", "")) != parsed_agent_kind.value
        payload["agent_kind"] = parsed_agent_kind.value
        if mode is not None:
            agent_mode = AgentMode.parse(mode)
            if str(payload.get("agent_mode", "")) != agent_mode.value:
                payload["agent_mode"] = agent_mode.value
                changed = True
        if not changed:
            return

        tmp_path = session_path.with_suffix(f"{session_path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for event in events:
                json.dump(event, handle, sort_keys=True)
                handle.write("\n")
        tmp_path.replace(session_path)

    def list_sessions(self, limit: int | None = 20) -> list[SessionRecord]:
        """List recently created sessions, newest first."""

        if not self.sessions_dir.exists():
            return []

        records = [
            record
            for path in self.sessions_dir.rglob("*.jsonl")
            if (record := self._read_session_record(path)) is not None
        ]
        records.sort(key=self._session_sort_key, reverse=True)
        return records if limit is None else records[:limit]

    def _session_sort_key(self, record: SessionRecord) -> tuple[str, int]:
        try:
            modified_at = record.path.stat().st_mtime_ns
        except OSError:
            modified_at = 0
        return record.updated_at, modified_at

    def read_session_events(self, session_path: Path) -> list[dict[str, Any]]:
        """Read all JSONL events for a session transcript."""

        if not session_path.exists():
            return []

        events: list[dict[str, Any]] = []
        with session_path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(cast(dict[str, Any], event))
        return events

    def _append_session_index(self, record: SessionRecord) -> None:
        payload = {
            "timestamp": record.created_at,
            "type": "session_index",
            "payload": {
                "session_id": record.session_id,
                "path": str(record.path),
                "cwd": record.cwd,
                "provider": record.provider,
                "model": record.model,
                "title": record.title,
                "unread": record.unread,
                "last_user_at": record.last_user_at,
            "agent_mode": record.mode.value,
            "agent_kind": record.agent_kind.value,
            },
        }
        self._append_jsonl(self.session_index_path, payload)

    def _read_session_record(self, path: Path) -> SessionRecord | None:
        events = self.read_session_events(path)
        if not events:
            return None

        first_event = events[0]
        if first_event.get("type") != "session_meta":
            return None

        metadata = cast(dict[str, Any], first_event.get("payload", {}))
        last_event = events[-1]
        title = self._session_title(metadata, events)
        return SessionRecord(
            session_id=str(metadata.get("id", metadata.get("session_id", path.stem))),
            path=path,
            created_at=str(metadata.get("created_at", first_event.get("timestamp", ""))),
            updated_at=str(last_event.get("timestamp", metadata.get("created_at", ""))),
            cwd=str(metadata.get("cwd", "")),
            provider=str(metadata.get("model_provider", metadata.get("provider", ""))),
            model=str(metadata.get("model", "")),
            title=title,
            message_count=self._session_message_count(events),
            unread=bool(metadata.get("unread", False)),
            last_user_at=self._last_user_message_timestamp(events)
            or str(metadata.get("created_at", first_event.get("timestamp", ""))),
            mode=AgentMode.parse(metadata.get("agent_mode")),
            agent_kind=parse_agent_kind(metadata.get("agent_kind")),
        )

    def _last_user_message_timestamp(self, events: list[dict[str, Any]]) -> str:
        for event in reversed(events):
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = (
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            if event_type in {"user_message", "skill_invocation"}:
                return str(event.get("timestamp", ""))
        return ""

    def _session_title(self, metadata: Mapping[str, Any], events: list[dict[str, Any]]) -> str:
        configured_title = metadata.get("title")
        if isinstance(configured_title, str) and configured_title != "New session":
            return configured_title

        for event in events:
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = (
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            if event_type not in {"user_message", "skill_invocation"}:
                continue
            message = str(payload.get("message", "")).strip()
            if message:
                return message[:60]
        return str(configured_title or "New session")

    def _session_message_count(self, events: list[dict[str, Any]]) -> int:
        count = 0
        for event in events:
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            event_type = (
                payload.get("type") if event.get("type") == "event_msg" else event.get("type")
            )
            message = str(payload.get("message", "")).strip()
            if event_type in {"user_message", "skill_invocation", "agent_message"} and message:
                count += 1
            elif event_type == "system_message" and message:
                role = str(payload.get("role", "system"))
                if role not in {"worker", "question"}:
                    count += 1
        return count

    def _worker_session_parent_dir(self, parent_session_path: Path | None) -> Path:
        if parent_session_path is not None:
            parent = parent_session_path.expanduser()
            with suppress(ValueError):
                parent.relative_to(self.sessions_dir)
                return parent.parent
        date_parts = datetime.now(tz=UTC).strftime("%Y/%m/%d")
        return self.sessions_dir / date_parts

    def _worker_session_folder_name(self, worker_name: str, worker_id: str) -> str:
        name = self._safe_worker_session_component(worker_name or "Worker")
        identifier = self._safe_worker_session_component(worker_id or uuid4().hex[:8])
        return f"worker-{name}-{identifier}"

    def _subagent_session_folder_name(self, subagent_name: str, subagent_id: str) -> str:
        name = self._safe_worker_session_component(subagent_name or "Subagent")
        identifier = self._safe_worker_session_component(subagent_id or uuid4().hex[:8])
        return f"subagent-{name}-{identifier}"

    def _safe_worker_session_component(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
        return cleaned.strip(".-_") or "unknown"

    def _delete_worker_sessions_for_record(self, record: SessionRecord) -> None:
        for directory in (
            *record.path.parent.glob("worker-*"),
            *record.path.parent.glob("subagent-*"),
        ):
            if not directory.is_dir():
                continue
            session_path = directory / "session.jsonl"
            if self._worker_session_belongs_to_record(
                session_path,
                record,
            ) or self._subagent_session_belongs_to_record(session_path, record):
                shutil.rmtree(directory, ignore_errors=True)

    def _worker_session_belongs_to_record(
        self,
        worker_session_path: Path,
        record: SessionRecord,
    ) -> bool:
        events = self.read_session_events(worker_session_path)
        if not events:
            return False
        first_event = events[0]
        if first_event.get("type") != "worker_session_meta":
            return False
        payload = first_event.get("payload")
        if not isinstance(payload, dict):
            return False
        parent_session_id = str(payload.get("parent_session_id") or "")
        if parent_session_id and parent_session_id == record.session_id:
            return True
        parent_session_path = str(payload.get("parent_session_path") or "")
        return bool(parent_session_path) and parent_session_path == str(record.path)

    def _subagent_session_belongs_to_record(
        self,
        subagent_session_path: Path,
        record: SessionRecord,
    ) -> bool:
        events = self.read_session_events(subagent_session_path)
        if not events:
            return False
        first_event = events[0]
        if first_event.get("type") != "subagent_session_meta":
            return False
        payload = first_event.get("payload")
        if not isinstance(payload, dict):
            return False
        parent_session_id = str(payload.get("parent_session_id") or "")
        if parent_session_id and parent_session_id == record.session_id:
            return True
        parent_session_path = str(payload.get("parent_session_path") or "")
        return bool(parent_session_path) and parent_session_path == str(record.path)

    def _is_repo_trusted_in_legacy_file(self, repo_key: str) -> bool:
        trusted = self._read_json_object(self.trusted_repos_path, default={"repos": {}})
        repos = trusted.get("repos")
        if not isinstance(repos, dict):
            return False
        repo_entry = repos.get(repo_key)
        if not isinstance(repo_entry, dict):
            return False
        return bool(repo_entry.get("trusted"))

    def _read_toml_object(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("rb") as handle:
            return tomllib.load(handle)

    def _write_config_toml(self, path: Path, config: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for field in CONFIG_SCALAR_FIELDS:
            value = config.get(field)
            if value is None:
                continue
            lines.append(f"{field} = {self._format_toml_value(value)}")

        global_allowed = config.get("global_allowed_commands")
        if isinstance(global_allowed, list) and global_allowed:
            lines.append(
                f"global_allowed_commands = {json.dumps(global_allowed)}"
            )

        global_rejected = config.get("global_rejected_commands")
        if isinstance(global_rejected, list) and global_rejected:
            lines.append(
                f"global_rejected_commands = {json.dumps(global_rejected)}"
            )

        projects = config.get("projects")
        if isinstance(projects, dict):
            for project_path, project in sorted(projects.items()):
                if not isinstance(project, dict):
                    continue
                lines.append("")
                lines.append(f"[projects.{json.dumps(str(project_path))}]")
                for key, value in sorted(project.items()):
                    lines.append(f"{key} = {self._format_toml_value(value)}")

        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp_path.replace(path)

    def _format_toml_value(self, value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(str(value))

    def _read_json_object(self, path: Path, default: Mapping[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return dict(default)
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return dict(default)
        return cast(dict[str, Any], data)

    def _write_json_object(
        self,
        path: Path,
        payload: Mapping[str, Any],
        mode: int | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)
        if mode is not None:
            path.chmod(mode)

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")

    def _repo_key(self, repo_path: Path) -> str:
        return str(repo_path.expanduser().resolve())
