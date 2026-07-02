"""Tests for voice-note replies (voice.tts_reply, ADR 0024) gateway wiring.

Covers the selection policy (``_should_send_voice_reply``), the
compose-then-render reply path (``_send_native_voice_note``) — including
inbound-driven idiomatic locale, URL stripping, and the fail-loud fallback —
the link follow-up (``_send_link_followup``), and the backwards-compatible
``tts_reply``/``native_audio_out`` config read.
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
    async def test_delivers_structured_part_and_reports_delivered(self, tmp_path):
        runner = _render_runner()
        event = _event()
        rendered = RenderedVoiceNote(path=str(tmp_path / "out.ogg"), locale="en-US")

        with patch(
            "tools.voice_reply.render_voice_note", return_value=rendered
        ) as render:
            delivered = await GatewayRunner._send_native_voice_note(
                runner, event, "Hello there"
            )

        # Returns True so the caller suppresses the duplicate text reply.
        assert delivered is True
        runner._deliver_voice_audio.assert_awaited_once()
        args = runner._deliver_voice_audio.call_args.args
        assert args[0] is event
        assert args[1] == rendered.path
        # English inbound -> en-US + English steering direction prepended.
        _, kwargs = render.call_args
        assert kwargs["locale"] == "en-US"
        spoken_arg = render.call_args.args[0]
        assert spoken_arg.startswith(
            "[Voice direction: idiomatic English as spoken in the United States]"
        )

    @pytest.mark.asyncio
    async def test_locale_detected_from_inbound_not_reply(self, tmp_path):
        # Inbound is Spanish; the English reply text must still be voiced in
        # idiomatic Mexican Spanish (detection runs on event.text).
        runner = _render_runner()
        event = _event(text="Hola, ¿me puedes ayudar con el despliegue?")
        rendered = RenderedVoiceNote(path=str(tmp_path / "out.ogg"), locale="es-MX")

        with patch(
            "tools.voice_reply.render_voice_note", return_value=rendered
        ) as render:
            delivered = await GatewayRunner._send_native_voice_note(
                runner, event, "Sure, the deploy is live now."
            )

        assert delivered is True
        _, kwargs = render.call_args
        assert kwargs["locale"] == "es-MX"
        assert render.call_args.args[0].startswith(
            "[Voice direction: idiomatic Spanish as spoken in Mexico]"
        )

    @pytest.mark.asyncio
    async def test_urls_stripped_from_spoken_text(self, tmp_path):
        runner = _render_runner()
        event = _event()
        rendered = RenderedVoiceNote(path=str(tmp_path / "out.ogg"), locale="en-US")

        with patch(
            "tools.voice_reply.render_voice_note", return_value=rendered
        ) as render:
            await GatewayRunner._send_native_voice_note(
                runner, event, "Here is the deploy: https://example.com/deploy/123"
            )

        spoken_arg = render.call_args.args[0]
        assert "https://example.com" not in spoken_arg
        assert "Here is the deploy" in spoken_arg

    @pytest.mark.asyncio
    async def test_render_failure_returns_false(self, tmp_path):
        runner = _render_runner()
        event = _event()

        with patch(
            "tools.voice_reply.render_voice_note",
            side_effect=VoiceRenderError("gemini down"),
        ):
            # Must not raise — the plain text reply is delivered by the normal path.
            delivered = await GatewayRunner._send_native_voice_note(
                runner, event, "Hello there"
            )

        assert delivered is False
        runner._deliver_voice_audio.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_text_after_strip_no_render(self):
        runner = _render_runner()
        event = _event()
        with patch("tools.voice_reply.render_voice_note") as render:
            delivered = await GatewayRunner._send_native_voice_note(runner, event, "   ")
        assert delivered is False
        render.assert_not_called()
        runner._deliver_voice_audio.assert_not_awaited()


def _followup_runner():
    adapter = SimpleNamespace(send=AsyncMock(), name="whatsapp_cloud")
    return SimpleNamespace(
        adapters={Platform.WHATSAPP_CLOUD: adapter},
        _reply_anchor_for_event=lambda event: "anchor-1",
        _thread_metadata_for_source=lambda source, anchor: {"thread_id": "t1"},
    ), adapter


class TestSendLinkFollowup:
    @pytest.mark.asyncio
    async def test_sends_link_when_url_present(self):
        runner, adapter = _followup_runner()
        event = _event()
        await GatewayRunner._send_link_followup(
            runner, event, "The report is at https://example.com/r?id=5 — enjoy."
        )
        adapter.send.assert_awaited_once()
        call = adapter.send.call_args
        assert call.args[0] == "chat-1"
        assert call.args[1] == "https://example.com/r?id=5"
        assert call.kwargs["reply_to"] == "anchor-1"
        assert call.kwargs["metadata"]["notify"] is True

    @pytest.mark.asyncio
    async def test_no_send_when_no_url(self):
        runner, adapter = _followup_runner()
        event = _event()
        await GatewayRunner._send_link_followup(runner, event, "plain reply, no link")
        adapter.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multiple_links_joined(self):
        runner, adapter = _followup_runner()
        event = _event()
        await GatewayRunner._send_link_followup(
            runner, event, "one https://a.com two https://b.com"
        )
        assert adapter.send.call_args.args[1] == "https://a.com\nhttps://b.com"


class TestNativeAudioOutEnabled:
    def test_new_tts_reply_key(self):
        runner = SimpleNamespace()
        with patch(
            "hermes_cli.config.load_config",
            return_value={"voice": {"tts_reply": True}},
        ):
            assert GatewayRunner._native_audio_out_enabled(runner) is True
        with patch(
            "hermes_cli.config.load_config",
            return_value={"voice": {"tts_reply": False}},
        ):
            assert GatewayRunner._native_audio_out_enabled(runner) is False

    def test_legacy_native_audio_out_key_still_read(self):
        # Backwards compatibility: configs written before the rename still work.
        runner = SimpleNamespace()
        with patch(
            "hermes_cli.config.load_config",
            return_value={"voice": {"native_audio_out": True}},
        ):
            assert GatewayRunner._native_audio_out_enabled(runner) is True
        with patch("hermes_cli.config.load_config", return_value={"voice": {}}):
            assert GatewayRunner._native_audio_out_enabled(runner) is False

    def test_new_key_takes_precedence_over_legacy(self):
        runner = SimpleNamespace()
        with patch(
            "hermes_cli.config.load_config",
            return_value={"voice": {"tts_reply": False, "native_audio_out": True}},
        ):
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
