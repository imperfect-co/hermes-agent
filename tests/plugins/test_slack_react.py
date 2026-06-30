"""Tests for the slack_react plugin (Slack + WhatsApp emoji reactions).

These prove the [[react:EMOJI]] path end-to-end against fakes:
  * react-only ([[react:+1]] NO_REPLY) fires exactly one reaction and returns
    the NO_REPLY silence token,
  * react+reply fires the reaction and returns the cleaned text,
  * reactions are resolved from the per-turn HERMES_SESSION_* context vars
    (NOT SessionEntry.origin),
  * WhatsApp shortcodes are mapped to literal unicode and group reactions
    carry the sender JID as participant,
  * coroutines are dispatched onto the gateway event loop (the behaviour
    WhatsApp's loop-bound aiohttp session depends on).
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest

import plugins.slack_react as sr


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSlackAdapter:
    def __init__(self):
        self.calls = []

    async def _add_reaction(self, channel, timestamp, emoji):
        self.calls.append((channel, timestamp, emoji))
        return True


class FakeWhatsAppAdapter:
    def __init__(self):
        self.calls = []

    async def send_reaction(self, chat_id, message_id, emoji, *, from_me=False, participant=None):
        self.calls.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "emoji": emoji,
                "from_me": from_me,
                "participant": participant,
            }
        )
        return SimpleNamespace(success=True)


def _patch_target(monkeypatch, adapter, *, chat_id="C1", message_id="111.222", participant=None):
    """Force _resolve_target to a controlled tuple (no live gateway needed).

    runner._gateway_loop=None routes _dispatch through the _run_async fallback,
    which runs the adapter coroutine on a private loop.
    """
    runner = SimpleNamespace(_gateway_loop=None)
    monkeypatch.setattr(
        sr,
        "_resolve_target",
        lambda platform: (runner, adapter, chat_id, message_id, participant),
    )
    return runner


# ---------------------------------------------------------------------------
# pre_llm_call — policy injection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("platform,expected", [("slack", "Slack"), ("whatsapp", "WhatsApp")])
def test_pre_llm_call_injects_policy(platform, expected):
    out = sr._pre_llm_call(platform=platform)
    assert isinstance(out, dict)
    assert "[[react:EMOJI]]" in out["context"]
    assert expected in out["context"]
    assert "NO_REPLY" in out["context"]


@pytest.mark.parametrize("platform", ["discord", "telegram", "", None])
def test_pre_llm_call_skips_other_platforms(platform):
    assert sr._pre_llm_call(platform=platform) is None


# ---------------------------------------------------------------------------
# transform_llm_output — Slack
# ---------------------------------------------------------------------------

def test_slack_react_only_returns_no_reply(monkeypatch):
    adapter = FakeSlackAdapter()
    _patch_target(monkeypatch, adapter, chat_id="C9", message_id="ts9")
    out = sr._transform_llm_output(platform="slack", response_text="[[react:+1]] NO_REPLY")
    assert out == "NO_REPLY"
    assert adapter.calls == [("C9", "ts9", "+1")]


def test_slack_react_plus_reply_returns_clean_text(monkeypatch):
    adapter = FakeSlackAdapter()
    _patch_target(monkeypatch, adapter)
    out = sr._transform_llm_output(
        platform="slack", response_text="[[react:eyes]] looking into it now"
    )
    assert out == "looking into it now"
    assert adapter.calls == [("C1", "111.222", "eyes")]


def test_slack_bare_directive_is_silent(monkeypatch):
    adapter = FakeSlackAdapter()
    _patch_target(monkeypatch, adapter)
    out = sr._transform_llm_output(platform="slack", response_text="[[react:tada]]")
    assert out == "NO_REPLY"
    assert adapter.calls == [("C1", "111.222", "tada")]


def test_slack_multiple_directives_dedup(monkeypatch):
    adapter = FakeSlackAdapter()
    _patch_target(monkeypatch, adapter)
    out = sr._transform_llm_output(
        platform="slack", response_text="[[react:+1]][[react:tada]][[react:+1]] thanks all"
    )
    assert out == "thanks all"
    assert adapter.calls == [("C1", "111.222", "+1"), ("C1", "111.222", "tada")]


def test_slack_reaction_count_capped(monkeypatch):
    adapter = FakeSlackAdapter()
    _patch_target(monkeypatch, adapter)
    directives = "".join(f"[[react:r{i}]]" for i in range(sr._MAX_REACTIONS_PER_RESPONSE + 3))
    sr._transform_llm_output(platform="slack", response_text=directives + " done")
    assert len(adapter.calls) == sr._MAX_REACTIONS_PER_RESPONSE


# ---------------------------------------------------------------------------
# transform_llm_output — WhatsApp (shortcode -> unicode, participant)
# ---------------------------------------------------------------------------

def test_whatsapp_maps_shortcode_to_unicode(monkeypatch):
    adapter = FakeWhatsAppAdapter()
    _patch_target(monkeypatch, adapter, chat_id="123@s.whatsapp.net", message_id="M1")
    out = sr._transform_llm_output(platform="whatsapp", response_text="[[react:+1]] NO_REPLY")
    assert out == "NO_REPLY"
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["emoji"] == "\U0001F44D"  # 👍
    assert adapter.calls[0]["from_me"] is False
    assert adapter.calls[0]["participant"] is None


def test_whatsapp_skips_unmapped_shortcode(monkeypatch):
    adapter = FakeWhatsAppAdapter()
    _patch_target(monkeypatch, adapter)
    out = sr._transform_llm_output(
        platform="whatsapp", response_text="[[react:zzz_unknown]] hello"
    )
    # Directive stripped from text, but nothing delivered (no unicode mapping).
    assert out == "hello"
    assert adapter.calls == []


def test_whatsapp_unmapped_directives_do_not_consume_cap(monkeypatch):
    # Several unmapped shortcodes (no unicode) must not exhaust the fired-reaction
    # budget and block a later valid one.
    adapter = FakeWhatsAppAdapter()
    _patch_target(monkeypatch, adapter, chat_id="123@s.whatsapp.net", message_id="M1")
    unmapped = "".join(f"[[react:nope{i}]]" for i in range(sr._MAX_REACTIONS_PER_RESPONSE + 2))
    out = sr._transform_llm_output(platform="whatsapp", response_text=unmapped + "[[react:+1]] NO_REPLY")
    assert out == "NO_REPLY"
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["emoji"] == "\U0001F44D"


def test_whatsapp_group_passes_participant(monkeypatch):
    adapter = FakeWhatsAppAdapter()
    _patch_target(
        monkeypatch,
        adapter,
        chat_id="g-1@g.us",
        message_id="M2",
        participant="789@s.whatsapp.net",
    )
    sr._transform_llm_output(platform="whatsapp", response_text="[[react:tada]] NO_REPLY")
    assert adapter.calls[0]["participant"] == "789@s.whatsapp.net"


# ---------------------------------------------------------------------------
# transform_llm_output — gating / no-ops
# ---------------------------------------------------------------------------

def test_unsupported_platform_is_noop(monkeypatch):
    adapter = FakeSlackAdapter()
    _patch_target(monkeypatch, adapter)
    assert sr._transform_llm_output(platform="discord", response_text="[[react:+1]] NO_REPLY") is None
    assert adapter.calls == []


def test_no_directive_returns_none(monkeypatch):
    adapter = FakeSlackAdapter()
    _patch_target(monkeypatch, adapter)
    assert sr._transform_llm_output(platform="slack", response_text="just a normal reply") is None
    assert adapter.calls == []


# ---------------------------------------------------------------------------
# _resolve_target — reads context vars, not origin
# ---------------------------------------------------------------------------

def _install_fake_gateway(monkeypatch, env, adapters):
    fake_run = types.ModuleType("gateway.run")
    runner = SimpleNamespace(adapters=adapters, _gateway_loop=None)
    fake_run._gateway_runner_ref = lambda: runner
    monkeypatch.setitem(sys.modules, "gateway.run", fake_run)
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda name, default="": env.get(name, default),
    )
    return runner


def test_resolve_target_reads_context_vars(monkeypatch):
    from gateway.config import Platform

    slack_adapter = FakeSlackAdapter()
    env = {"HERMES_SESSION_CHAT_ID": "C5", "HERMES_SESSION_MESSAGE_ID": "ts5"}
    runner = _install_fake_gateway(monkeypatch, env, {Platform.SLACK: slack_adapter})

    result = sr._resolve_target("slack")
    assert result is not None
    got_runner, adapter, chat_id, message_id, participant = result
    assert got_runner is runner
    assert adapter is slack_adapter
    assert (chat_id, message_id) == ("C5", "ts5")
    assert participant is None


def test_resolve_target_none_when_message_id_missing(monkeypatch):
    from gateway.config import Platform

    env = {"HERMES_SESSION_CHAT_ID": "C5", "HERMES_SESSION_MESSAGE_ID": ""}
    _install_fake_gateway(monkeypatch, env, {Platform.SLACK: FakeSlackAdapter()})
    assert sr._resolve_target("slack") is None


def test_resolve_target_group_participant(monkeypatch):
    from gateway.config import Platform

    wa = FakeWhatsAppAdapter()
    env = {
        "HERMES_SESSION_CHAT_ID": "grp@g.us",
        "HERMES_SESSION_MESSAGE_ID": "M7",
        "HERMES_SESSION_USER_ID": "555@s.whatsapp.net",
    }
    _install_fake_gateway(monkeypatch, env, {Platform.WHATSAPP: wa})
    result = sr._resolve_target("whatsapp")
    assert result is not None
    _, adapter, chat_id, message_id, participant = result
    assert adapter is wa
    assert participant == "555@s.whatsapp.net"


def test_resolve_target_group_missing_participant_returns_none(monkeypatch):
    from gateway.config import Platform

    wa = FakeWhatsAppAdapter()
    env = {
        "HERMES_SESSION_CHAT_ID": "grp@g.us",
        "HERMES_SESSION_MESSAGE_ID": "M7",
        "HERMES_SESSION_USER_ID": "",  # sender JID unresolved → cannot build group key
    }
    _install_fake_gateway(monkeypatch, env, {Platform.WHATSAPP: wa})
    assert sr._resolve_target("whatsapp") is None


def test_resolve_target_none_when_adapter_absent(monkeypatch):
    from gateway.config import Platform

    env = {"HERMES_SESSION_CHAT_ID": "C5", "HERMES_SESSION_MESSAGE_ID": "ts5"}
    _install_fake_gateway(monkeypatch, env, {})  # no slack adapter registered
    assert sr._resolve_target("slack") is None


# ---------------------------------------------------------------------------
# _dispatch — runs coroutines on the gateway loop (WhatsApp-critical)
# ---------------------------------------------------------------------------

def test_dispatch_runs_on_gateway_loop():
    loop = asyncio.new_event_loop()
    th = threading.Thread(target=loop.run_forever, daemon=True)
    th.start()
    try:
        seen = {}

        async def coro():
            seen["loop"] = asyncio.get_running_loop()
            return "ok"

        runner = SimpleNamespace(_gateway_loop=loop)
        # _dispatch returns the coroutine's own result.
        assert sr._dispatch(runner, lambda: coro()) == "ok"
        assert seen["loop"] is loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        th.join(timeout=2)
        loop.close()


def test_dispatch_times_out_and_returns_false():
    loop = asyncio.new_event_loop()
    th = threading.Thread(target=loop.run_forever, daemon=True)
    th.start()
    try:
        async def slow():
            await asyncio.sleep(5)

        runner = SimpleNamespace(_gateway_loop=loop)
        start = time.monotonic()
        ok = sr._dispatch(runner, lambda: slow(), timeout=0.3)
        elapsed = time.monotonic() - start
        assert ok is False
        assert elapsed < 2.0  # returned at the timeout, not after the full 5s
    finally:
        loop.call_soon_threadsafe(loop.stop)
        th.join(timeout=2)
        loop.close()


def test_dispatch_fallback_without_gateway_loop():
    ran = {"n": 0}

    async def coro():
        ran["n"] += 1
        return True

    runner = SimpleNamespace(_gateway_loop=None)
    assert sr._dispatch(runner, lambda: coro()) is True
    assert ran["n"] == 1


def test_dispatch_skips_non_running_loop():
    # A loop that exists but is not running must NOT be scheduled onto (it would
    # hang to timeout); fall back to the private-loop path instead.
    loop = asyncio.new_event_loop()  # created, never started → not running
    try:
        ran = {"n": 0}

        async def coro():
            ran["n"] += 1
            return True

        runner = SimpleNamespace(_gateway_loop=loop)
        assert sr._dispatch(runner, lambda: coro()) is True
        assert ran["n"] == 1
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# _reaction_ok / success-signal accounting
# ---------------------------------------------------------------------------

class FakeSlackAdapterFail:
    def __init__(self):
        self.calls = []

    async def _add_reaction(self, channel, timestamp, emoji):
        self.calls.append((channel, timestamp, emoji))
        return False  # e.g. missing scope / already reacted


class FakeWhatsAppAdapterFail:
    def __init__(self):
        self.calls = []

    async def send_reaction(self, chat_id, message_id, emoji, *, from_me=False, participant=None):
        self.calls.append(emoji)
        return SimpleNamespace(success=False)


def test_reaction_ok_interprets_results():
    assert sr._reaction_ok(True) is True
    assert sr._reaction_ok(False) is False
    assert sr._reaction_ok(None) is True  # no explicit signal → ran ok
    assert sr._reaction_ok(SimpleNamespace(success=True)) is True
    assert sr._reaction_ok(SimpleNamespace(success=False)) is False


def test_slack_reaction_failure_not_counted(monkeypatch):
    adapter = FakeSlackAdapterFail()
    _patch_target(monkeypatch, adapter)
    # Attempted, but the adapter reported failure → not counted.
    assert sr._add_reactions("slack", ["+1"]) == 0
    assert adapter.calls == [("C1", "111.222", "+1")]


def test_whatsapp_reaction_failure_not_counted(monkeypatch):
    adapter = FakeWhatsAppAdapterFail()
    _patch_target(monkeypatch, adapter, chat_id="123@s.whatsapp.net", message_id="M1")
    assert sr._add_reactions("whatsapp", ["+1"]) == 0
    assert adapter.calls == ["\U0001F44D"]
