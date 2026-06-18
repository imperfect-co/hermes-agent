"""Tests for the schema-driven memory-provider config adapter.

These assert the *mapping contract* (how a provider's declared
``get_config_schema()`` becomes the normalized desktop field shape) and the
write-time coercion — not a snapshot of any particular provider's fields, which
are free to change.
"""

import pytest

from hermes_cli.memory_providers import (
    KIND_BOOLEAN,
    KIND_SECRET,
    KIND_SELECT,
    KIND_TEXT,
    VALUE_BOOL,
    VALUE_INT,
    coerce_value,
    describe_provider,
)


def _by_key(provider):
    return {f.key: f for f in provider.fields}


def test_secret_field_maps_to_secret_kind_bound_to_env():
    provider = describe_provider(
        "x",
        [{"key": "api_key", "secret": True, "env_var": "X_API_KEY", "url": "https://x"}],
    )
    field = _by_key(provider)["api_key"]

    assert field.kind == KIND_SECRET
    assert field.is_secret is True
    assert field.env_key == "X_API_KEY"
    assert field.url == "https://x"


def test_choices_map_to_select_options():
    provider = describe_provider(
        "x", [{"key": "mode", "default": "cloud", "choices": ["cloud", "local"]}]
    )
    field = _by_key(provider)["mode"]

    assert field.kind == KIND_SELECT
    assert field.allowed_values() == {"cloud", "local"}


def test_bool_default_maps_to_boolean_kind():
    provider = describe_provider("x", [{"key": "auto", "default": True}])
    field = _by_key(provider)["auto"]

    assert field.kind == KIND_BOOLEAN
    assert field.value_type == VALUE_BOOL
    assert field.default == "true"


def test_int_default_maps_to_text_with_int_value_type():
    provider = describe_provider("x", [{"key": "tokens", "default": 4096}])
    field = _by_key(provider)["tokens"]

    assert field.kind == KIND_TEXT
    assert field.value_type == VALUE_INT
    assert field.default == "4096"


def test_when_clause_is_carried_through():
    provider = describe_provider(
        "x",
        [
            {"key": "mode", "default": "cloud", "choices": ["cloud", "local"]},
            {"key": "api_url", "default": "u", "when": {"mode": "cloud"}},
        ],
    )
    api_url = _by_key(provider)["api_url"]

    assert api_url.when == (("mode", "cloud"),)
    assert api_url.when_matches({"mode": "cloud"}) is True
    assert api_url.when_matches({"mode": "local"}) is False


def test_every_field_has_a_known_kind():
    # Invariant: the adapter never emits a field the renderer can't handle.
    provider = describe_provider(
        "x",
        [
            {"key": "a", "secret": True, "env_var": "A"},
            {"key": "b", "choices": ["1", "2"]},
            {"key": "c", "default": True},
            {"key": "d", "default": "text"},
        ],
    )
    known = {KIND_TEXT, KIND_SELECT, KIND_SECRET, KIND_BOOLEAN}

    assert provider.fields  # non-empty
    assert all(f.kind in known for f in provider.fields)


def test_malformed_entries_are_skipped_not_fatal():
    provider = describe_provider("x", ["nope", {}, {"key": ""}, {"key": "ok"}])

    assert _by_key(provider).keys() == {"ok"}


def test_coerce_rejects_value_outside_select_options():
    provider = describe_provider("x", [{"key": "mode", "choices": ["cloud", "local"]}])
    field = _by_key(provider)["mode"]

    with pytest.raises(ValueError):
        coerce_value(field, "bogus")


def test_coerce_casts_bool_and_int_to_native_types():
    provider = describe_provider(
        "x", [{"key": "auto", "default": True}, {"key": "tokens", "default": 4096}]
    )
    fields = _by_key(provider)

    assert coerce_value(fields["auto"], "true") is True
    assert coerce_value(fields["auto"], "off") is False
    assert coerce_value(fields["tokens"], "8000") == 8000


def test_coerce_empty_falls_back_to_default():
    provider = describe_provider("x", [{"key": "name", "default": "hermes"}])
    field = _by_key(provider)["name"]

    assert coerce_value(field, "") == "hermes"
