"""Schema-driven configuration surface for desktop memory providers.

Memory providers already declare their configurable fields via
``MemoryProvider.get_config_schema()`` (the same declaration ``hermes memory
setup`` walks). This module is the *adapter* that normalizes those raw
declarations into a stable, JSON-serializable shape the desktop config panel
renders generically — no per-provider UI, no hand-maintained registry.

Combined with ``discover_memory_providers()`` driving the dropdown, adding or
porting a provider is pure declaration: implement ``get_config_schema()`` (and
optionally ``read_current_config()`` / ``save_config()``) and the provider
shows up, configured, in the desktop UI with zero bespoke code.

This module is intentionally pure transformation: it imports nothing from the
config/env layer and does no I/O. ``web_server`` owns loading the live
provider, reading current values, writing via the provider's ``save_config()``,
and persisting secrets to the env store. Keeping the mapping here (and the I/O
there) is what lets the same normalized schema drive both the HTTP payload and
its validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Field kinds understood by the generic renderer.
KIND_TEXT = "text"
KIND_SELECT = "select"
KIND_SECRET = "secret"
KIND_BOOLEAN = "boolean"

# Native value types, derived from a field's declared default. Submitted form
# values arrive as strings; ``coerce_value`` casts them back to these so a
# provider's config file keeps booleans/numbers rather than stringified ones.
VALUE_STR = "str"
VALUE_BOOL = "bool"
VALUE_INT = "int"
VALUE_FLOAT = "float"


@dataclass(frozen=True)
class ProviderFieldOption:
    """A single choice for a ``select`` field."""

    value: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class ProviderField:
    """One configurable field, normalized from a provider's schema entry.

    Storage is decided by ``kind``:

    * ``text`` / ``select`` / ``boolean`` — persisted by the provider's own
      ``save_config()`` to its native location, keyed by ``key``.
    * ``secret`` — persisted to the env store under ``env_key`` and never read
      back over the API (only an ``is_set`` flag is surfaced).

    ``when`` carries a provider's conditional-visibility clause (e.g. Hindsight
    only shows ``api_url`` ``when`` ``mode == cloud``); the renderer hides
    fields whose clause doesn't match the current values, and the same clause
    gates server-side validation so hidden fields aren't required.
    """

    key: str
    label: str
    kind: str = KIND_TEXT
    value_type: str = VALUE_STR
    default: str = ""
    description: str = ""
    placeholder: str = ""
    required: bool = False
    url: str = ""
    options: Tuple[ProviderFieldOption, ...] = ()
    env_key: Optional[str] = None
    when: Tuple[Tuple[str, str], ...] = ()

    @property
    def is_secret(self) -> bool:
        return self.kind == KIND_SECRET

    def allowed_values(self) -> set[str]:
        return {opt.value for opt in self.options}

    def when_matches(self, values: Dict[str, Any]) -> bool:
        """Whether this field is active given the current field values."""

        return all(str(values.get(k, "")) == v for k, v in self.when)


@dataclass(frozen=True)
class MemoryProvider:
    """A declared memory provider and its normalized configurable fields."""

    name: str
    label: str
    fields: Tuple[ProviderField, ...] = ()

    def field(self, key: str) -> Optional[ProviderField]:
        for f in self.fields:
            if f.key == key:
                return f
        return None


def _label_from_key(raw: str) -> str:
    """Humanize a snake/kebab key into a field/provider label."""

    cleaned = raw.replace("_", " ").replace("-", " ").strip()
    if not cleaned:
        return raw
    # Title-case but leave acronym-ish all-caps tokens intact.
    return " ".join(w if w.isupper() else w.capitalize() for w in cleaned.split())


def _value_type_of(default: Any) -> str:
    # bool first — bool is a subclass of int.
    if isinstance(default, bool):
        return VALUE_BOOL
    if isinstance(default, int):
        return VALUE_INT
    if isinstance(default, float):
        return VALUE_FLOAT
    return VALUE_STR


def _default_to_str(default: Any) -> str:
    if default is None:
        return ""
    if isinstance(default, bool):
        return "true" if default else "false"
    return str(default)


def _field_from_schema(raw: Dict[str, Any]) -> Optional[ProviderField]:
    """Normalize one ``get_config_schema()`` entry into a ``ProviderField``."""

    key = str(raw.get("key") or "").strip()
    if not key:
        return None

    default = raw.get("default")
    secret = bool(raw.get("secret"))
    choices = raw.get("choices")
    env_var = raw.get("env_var")

    when_raw = raw.get("when")
    when: Tuple[Tuple[str, str], ...] = ()
    if isinstance(when_raw, dict):
        when = tuple((str(k), str(v)) for k, v in when_raw.items())

    if secret:
        kind, value_type, options = KIND_SECRET, VALUE_STR, ()
    elif choices:
        kind = KIND_SELECT
        value_type = _value_type_of(default)
        options = tuple(
            ProviderFieldOption(value=str(c), label=str(c)) for c in choices
        )
    elif isinstance(default, bool):
        kind, value_type, options = KIND_BOOLEAN, VALUE_BOOL, ()
    else:
        kind, value_type, options = KIND_TEXT, _value_type_of(default), ()

    return ProviderField(
        key=key,
        label=_label_from_key(key),
        kind=kind,
        value_type=value_type,
        default=_default_to_str(default),
        description=str(raw.get("description") or ""),
        placeholder=str(raw.get("placeholder") or ""),
        required=bool(raw.get("required")),
        url=str(raw.get("url") or ""),
        options=options,
        env_key=str(env_var) if env_var else None,
        when=when,
    )


def describe_provider(
    name: str,
    schema: List[Dict[str, Any]],
    label: Optional[str] = None,
) -> MemoryProvider:
    """Adapt a provider's raw ``get_config_schema()`` into a normalized descriptor.

    ``schema`` is the list returned by the live provider; ``label`` overrides
    the humanized name (e.g. a provider's display name from ``plugin.yaml``).
    Unparseable entries are skipped rather than raising, so one malformed field
    can't blank out a provider's whole config surface.
    """

    fields: List[ProviderField] = []
    for raw in schema or []:
        if not isinstance(raw, dict):
            continue
        field = _field_from_schema(raw)
        if field is not None:
            fields.append(field)

    return MemoryProvider(
        name=name,
        label=label or _label_from_key(name),
        fields=tuple(fields),
    )


def coerce_value(field: ProviderField, raw: str) -> Any:
    """Validate + cast a submitted string back to the field's native type.

    Raises ``ValueError`` for a select value outside its options. Empty input
    falls back to the field's declared default. Booleans accept the usual
    truthy/falsy spellings; numbers parse leniently and fall back to the
    default on a bad parse rather than corrupting the config.
    """

    value = (raw or "").strip()

    if field.kind == KIND_SELECT:
        if not value:
            value = field.default
        if field.options and value not in field.allowed_values():
            raise ValueError(f"Invalid value for '{field.key}'")
        return value

    if field.value_type == VALUE_BOOL:
        if value == "":
            value = field.default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    if field.value_type in (VALUE_INT, VALUE_FLOAT):
        if value == "":
            value = field.default
        try:
            return int(value) if field.value_type == VALUE_INT else float(value)
        except (TypeError, ValueError):
            try:
                return (
                    int(field.default)
                    if field.value_type == VALUE_INT
                    else float(field.default)
                )
            except (TypeError, ValueError):
                return 0 if field.value_type == VALUE_INT else 0.0

    return value or field.default
