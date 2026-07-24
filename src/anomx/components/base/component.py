"""Component metadata surface and discovery.

Every anomx building block — normality models, scorers, detectors,
classifiers, algorithms — inherits from :class:`BaseComponent`. The base class
provides everything a platform needs to render the component without importing
domain code: name, description, docs, icon, configuration schema, current
configuration, capability keys, and data signature.
"""

from __future__ import annotations

import inspect
import pkgutil
import re
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar

from anomx._shared import normalise_component_key, normalize_text
from anomx.components.base.capabilities import collect_capability_keys
from anomx.components.base.signature import ModelSignature

PROJECT_ROOT = Path(__file__).resolve().parents[4]
COMPONENT_PACKAGE_NAMES = (
    "anomx.components.algorithms",
    "anomx.components.detection",
    "anomx.components.models",
)


def humanize_component_name(value: str | None) -> str:
    normalized_value = str(value or "").strip()
    if not normalized_value:
        return ""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", normalized_value).strip()


def read_component_source_path(component_class: type[object]) -> str:
    source_path = inspect.getsourcefile(component_class) or inspect.getfile(component_class)
    resolved_path = Path(source_path).resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved_path)


class BaseComponent:
    """Common metadata surface for installable pipeline components."""

    component_type: ClassVar[str | None] = None
    component_key: ClassVar[str] = ""
    component_name: ClassVar[str] = ""
    component_description: ClassVar[str] = ""
    component_docs: ClassVar[str] = ""
    component_icon: ClassVar[str] = ""
    component_image_path: ClassVar[str] = ""
    component_status: ClassVar[str] = "active"
    component_default_config: ClassVar[dict[str, Any]] = {}
    component_config_schema: ClassVar[dict[str, Any]] = {}
    component_code_version: ClassVar[str] = ""
    signature: ClassVar[ModelSignature | None] = None

    @classmethod
    def is_component_abstract(cls) -> bool:
        return inspect.isabstract(cls) or not cls.get_component_type()

    @classmethod
    def get_component_type(cls) -> str:
        return normalize_text(getattr(cls, "component_type", "")) or ""

    @classmethod
    def get_component_key(cls) -> str:
        value = getattr(cls, "component_key", "") or cls.get_component_name() or cls.__name__
        return normalise_component_key(value)

    @classmethod
    def get_component_name(cls) -> str:
        explicit_name = normalize_text(getattr(cls, "component_name", ""))
        return explicit_name or humanize_component_name(cls.__name__)

    @classmethod
    def get_component_docs(cls) -> str:
        explicit_docs = normalize_text(getattr(cls, "component_docs", ""))
        if explicit_docs:
            return explicit_docs
        return normalize_text(inspect.getdoc(cls))

    @classmethod
    def get_component_description(cls) -> str:
        explicit_description = normalize_text(getattr(cls, "component_description", ""))
        if explicit_description:
            return explicit_description
        docs = cls.get_component_docs()
        if not docs:
            return ""
        return normalize_text(docs.split("\n\n", 1)[0])

    @classmethod
    def get_component_icon(cls) -> str:
        return normalize_text(getattr(cls, "component_icon", ""))

    @classmethod
    def get_component_image_path(cls) -> str:
        return normalize_text(getattr(cls, "component_image_path", ""))

    @classmethod
    def get_component_status(cls) -> str:
        return normalize_text(getattr(cls, "component_status", "")) or "active"

    @classmethod
    def get_component_default_config(cls) -> dict[str, Any]:
        value = getattr(cls, "component_default_config", {})
        return dict(value) if isinstance(value, dict) else {}

    @classmethod
    def get_component_config_schema(cls) -> dict[str, Any]:
        value = getattr(cls, "component_config_schema", {})
        return dict(value) if isinstance(value, dict) else {}

    @classmethod
    def get_component_code_version(cls) -> str:
        return normalize_text(getattr(cls, "component_code_version", ""))

    @classmethod
    def get_component_source_path(cls) -> str:
        return read_component_source_path(cls)

    @classmethod
    def get_component_import_path(cls) -> str:
        return f"{cls.__module__}.{cls.__name__}"

    @classmethod
    def get_component_capabilities(cls) -> list[str]:
        return collect_capability_keys(cls)

    @classmethod
    def get_component_signature(cls) -> ModelSignature | None:
        signature = getattr(cls, "signature", None)
        return signature if isinstance(signature, ModelSignature) else None

    @classmethod
    def get_component_parameters(cls) -> list[dict[str, Any]]:
        """Describe the configurable parameters of this component.

        The schema and defaults are the source of truth; the `__init__`
        docstring may add per-parameter descriptions later.
        """
        schema = cls.get_component_config_schema()
        defaults = cls.get_component_default_config()
        parameter_names = list(dict.fromkeys([*schema.keys(), *defaults.keys()]))
        return [
            {
                "name": parameter_name,
                "type": normalize_text((schema.get(parameter_name) or {}).get("type")) or "string",
                "default": defaults.get(parameter_name),
            }
            for parameter_name in parameter_names
        ]

    def get_config(self) -> dict[str, Any]:
        """Return the effective configuration of this component instance."""
        config = getattr(self, "config", None)
        merged_config = {**self.get_component_default_config()}
        if isinstance(config, dict):
            merged_config.update(config)
        return merged_config

    @classmethod
    def get_component_definition_payload(cls) -> dict[str, Any]:
        signature = cls.get_component_signature()
        return {
            "capabilities": cls.get_component_capabilities(),
            "code_version": cls.get_component_code_version(),
            "component_type": cls.get_component_type(),
            "config_schema": cls.get_component_config_schema(),
            "default_config": cls.get_component_default_config(),
            "description": cls.get_component_description(),
            "docs": cls.get_component_docs(),
            "icon": cls.get_component_icon(),
            "image_path": cls.get_component_image_path(),
            "import_path": cls.get_component_import_path(),
            "key": cls.get_component_key(),
            "name": cls.get_component_name(),
            "parameters": cls.get_component_parameters(),
            "python_class": cls.__name__,
            "python_module": cls.__module__,
            "signature": signature.to_dict() if signature is not None else {},
            "source_path": cls.get_component_source_path(),
            "status": cls.get_component_status(),
        }


def import_component_modules(package_names: tuple[str, ...] = COMPONENT_PACKAGE_NAMES) -> None:
    """Import every module below the configured component packages."""
    for package_name in package_names:
        package = import_module(package_name)
        package_paths = getattr(package, "__path__", None)
        if package_paths is None:
            continue
        for module_info in pkgutil.walk_packages(package_paths, f"{package.__name__}."):
            import_module(module_info.name)


def iter_component_classes(package_names: tuple[str, ...] = COMPONENT_PACKAGE_NAMES):
    """Yield every non-abstract component class in the package set."""
    import_component_modules(package_names=package_names)
    seen_classes: set[type[object]] = set()
    pending_classes: list[type[object]] = [BaseComponent]

    while pending_classes:
        current_class = pending_classes.pop()
        for subclass in current_class.__subclasses__():
            if subclass in seen_classes:
                continue
            seen_classes.add(subclass)
            pending_classes.append(subclass)
            if subclass.is_component_abstract():
                continue
            yield subclass


def discover_component_payloads(package_names: tuple[str, ...] = COMPONENT_PACKAGE_NAMES) -> list[dict[str, Any]]:
    """Discover all concrete components and return stable metadata payloads."""
    payloads_by_key: dict[str, dict[str, Any]] = {}
    for component_class in iter_component_classes(package_names=package_names):
        payload = component_class.get_component_definition_payload()
        payload_key = payload["key"]
        if payload_key:
            payloads_by_key[payload_key] = payload
    return [payloads_by_key[key] for key in sorted(payloads_by_key)]
