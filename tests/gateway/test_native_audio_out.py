"""Tests for native audio output (ADR 0024, Phase 1) gateway wiring.

Covers the selection policy (``_should_send_voice_reply``) and the
compose-then-render reply path (``_send_native_voice_note``) including the
fail-loud fallback to plain text.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools.voice_reply import RenderedVoiceNote, VoiceRenderError


def _event(text="hello", message_type=MessageType.TEXT):
    source = SessionSource(
        platform=Platform.WHATSAPP_CLOUD,
        chat_id="chat-1",
        chat_type="dm",
    )
    return MessageEvent(
        text=text,
        message_type=message_type,
        source=source,
        message_id="msg-1",
    )


def _policy_runner(native_enabled, voice_mode="off"):
    return SimpleNamespace(
        _voice_mode={f"{Platform.WHATSAPP_CLOUD.value}:chat-1": voice_mode},
        _voice_key=lambda platform, chat_id: f"{platform.value}:{chat_id}",
        _native_audio_out_enabled=lambda: native_enabled,
    )


class TestSelectionPolicy:
    def test_native_voice_input_selected_even_when_not_already_sent(self):
        # Native trigger must bypass the legacy base-adapter dedup.
        runner = _policy_runner(native_enabled=True)
        event = _event(message_type=MessageType.VOICE)
        assert (
            GatewayRunner._should_send_voice_reply(
                runner, event, "Here is your answer", [], already_sent=False
            )
            is True
        )

    def test_native_explicit_request_in_text_selected(self):
        runner = _policy_runner(native_enabled=True)
        event = _event(text="please send me a voice note", message_type=MessageType.TEXT)
        assert (
            GatewayRunner._should_send_voice_reply(runner, event, "ok", [])
            is True
        )

    def test_native_plain_text_without_request_not_selected(self):
        runner = _policy_runner(native_enabled=True)
        event = _event(text="what's the weather?", message_type=MessageType.TEXT)
        assert (
            GatewayRunner._should_send_voice_reply(runner, event, "Sunny", [])
            is False
        )

    def test_disabled_and_voice_off_not_selected(self):
        runner = _policy_runner(native_enabled=False, voice_mode="off")
        event = _event(message_type=MessageType.VOICE)
        assert (
            GatewayRunner._should_send_voice_reply(
                runner, event, "answer", [], already_sent=True
            )
            is False
        )

    def test_legacy_voice_all_still_works_when_native_disabled(self):
        runner = _policy_runner(native_enabled=False, voice_mode="all")
        event = _event(message_type=MessageType.TEXT)
        assert (
            GatewayRunner._should_send_voice_reply(runner, event, "answer", [])
            is True
        )

    def test_empty_response_not_selected(self):
        runner = _policy_runner(native_enabled=True)
        event = _event(message_type=MessageType.VOICE)
        assert GatewayRunner._should_send_voice_reply(runner, event, "", []) is False

    def test_error_response_not_selected(self):
        runner = _policy_runner(native_enabled=True)
        event = _event(message_type=MessageType.VOICE)
        assert (
            GatewayRunner._should_send_voice_reply(runner, event, "Error: boom", [])
            is False
        )

    def test_agent_already_called_tts_dedups(self):
        runner = _policy_runner(native_enabled=True)
        event = _event(message_type=MessageType.VOICE)
        agent_messages = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "text_to_speech"}}],
            }
        ]
        assert (
            GatewayRunner._should_send_voice_reply(runner, event, "answer", agent_messages)
            is False
        )


def _render_runner():
    return SimpleNamespace(
        _native_audio_out_render_config=lambda: ("Charon", {}),
        _deliver_voice_audio=AsyncMock(),
    )


class TestSendNativeVoiceNote:
    @pytest.mark.asyncio
    async def test_delivers_structured_part(self, tmp_path):
        runner = _render_runner()
        event = _event()
        rendered = RenderedVoiceNote(path=str(tmp_path / "out.ogg"), locale="en-US")

        with patch("tools.voice_reply.render_voice_note", return_value=rendered):
            await GatewayRunner._send_native_voice_note(runner, event, "Hello there")

        runner._deliver_voice_audio.assert_awaited_once()
        args = runner._deliver_voice_audio.call_args.args
        assert args[0] is event
        assert args[1] == rendered.path

    @pytest.mark.asyncio
    async def test_render_failure_falls_back_to_text(self, tmp_path):
        runner = _render_runner()
        event = _event()

        with patch(
            "tools.voice_reply.render_voice_note",
            side_effect=VoiceRenderError("gemini down"),
        ):
            # Must not raise — the plain text reply is delivered by the normal path.
            await GatewayRunner._send_native_voice_note(runner, event, "Hello there")

        runner._deliver_voice_audio.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_text_after_strip_no_render(self):
        runner = _render_runner()
        event = _event()
        with patch("tools.voice_reply.render_voice_note") as render:
            await GatewayRunner._send_native_voice_note(runner, event, "   ")
        render.assert_not_called()
        runner._deliver_voice_audio.assert_not_awaited()


class TestNativeAudioOutEnabled:
    def test_reads_config_flag(self):
        runner = SimpleNamespace()
        with patch(
            "hermes_cli.config.load_config",
            return_value={"voice": {"native_audio_out": True}},
        ):
            assert GatewayRunner._native_audio_out_enabled(runner) is True
        with patch("hermes_cli.config.load_config", return_value={"voice": {}}):
            assert GatewayRunner._native_audio_out_enabled(runner) is False

    def test_render_config_defaults_to_charon(self):
        runner = SimpleNamespace()
        with patch(
            "hermes_cli.config.load_config",
            return_value={"tts": {"gemini": {"model": "m"}}},
        ):
            voice, cfg = GatewayRunner._native_audio_out_render_config(runner)
        assert voice == "Charon"
        assert cfg == {"model": "m"}
