"""Tests for the WhatsApp adapter's send_reaction (bridge POST /react contract).

Pins the Python->bridge envelope: chatId (normalised to a JID), messageId,
emoji (literal unicode), fromMe, and the group-only participant field; plus the
connection / message_id guards. Also confirms build_source now plumbs the
triggering message_id through to SessionSource.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


# ---------------------------------------------------------------------------
# Fake aiohttp session
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, status=200, json_data=None, text_data=""):
        self.status = status
        self._json = json_data or {}
        self._text = text_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._json

    async def text(self):
        return self._text


class _FakeSession:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        return self._resp


def _make_adapter():
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True)
    adapter._running = True
    adapter._bridge_port = 3000
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    return adapter


# ---------------------------------------------------------------------------
# send_reaction
# ---------------------------------------------------------------------------

def test_send_reaction_posts_react_payload():
    adapter = _make_adapter()
    session = _FakeSession(_FakeResp(status=200, json_data={"success": True}))
    adapter._http_session = session

    res = asyncio.run(adapter.send_reaction("123@s.whatsapp.net", "MID", "\U0001F44D"))

    assert res.success is True
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"].endswith("/react")
    payload = call["json"]
    assert payload["chatId"].endswith("@s.whatsapp.net")
    assert payload["messageId"] == "MID"
    assert payload["emoji"] == "\U0001F44D"
    assert payload["fromMe"] is False
    assert "participant" not in payload


def test_send_reaction_includes_participant_for_group():
    adapter = _make_adapter()
    session = _FakeSession(_FakeResp(status=200, json_data={"success": True}))
    adapter._http_session = session

    asyncio.run(
        adapter.send_reaction(
            "120363000000000000@g.us",
            "MID",
            "\U0001F389",
            participant="789@s.whatsapp.net",
        )
    )

    payload = session.calls[0]["json"]
    assert payload["participant"].endswith("@s.whatsapp.net")


def test_send_reaction_propagates_bridge_error():
    adapter = _make_adapter()
    session = _FakeSession(_FakeResp(status=500, text_data="boom"))
    adapter._http_session = session

    res = asyncio.run(adapter.send_reaction("123@s.whatsapp.net", "MID", "\U0001F44D"))
    assert res.success is False
    assert "boom" in (res.error or "")


def test_send_reaction_requires_connection():
    adapter = _make_adapter()
    adapter._running = False
    adapter._http_session = None
    res = asyncio.run(adapter.send_reaction("123@s.whatsapp.net", "MID", "\U0001F44D"))
    assert res.success is False


def test_send_reaction_requires_message_id():
    adapter = _make_adapter()
    adapter._http_session = _FakeSession(_FakeResp(status=200))
    res = asyncio.run(adapter.send_reaction("123@s.whatsapp.net", "", "\U0001F44D"))
    assert res.success is False


# ---------------------------------------------------------------------------
# build_source carries the triggering message id
# ---------------------------------------------------------------------------

def test_build_source_sets_message_id():
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    source = adapter.build_source(chat_id="123@s.whatsapp.net", message_id="M9")
    assert source.message_id == "M9"
