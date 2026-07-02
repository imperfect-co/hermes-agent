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
from gateway.platforms.base import MessageEvent, MessageType, SendResult
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
    @pytest.mark.parametrize(
        "inbound_text, override_inbound, reply_text, expected_locale, expected_prefix",
        [
            (
                "Hello",
                None,
                "Hello there",
                "en-US",
                "[Voice direction: idiomatic English as spoken in the United States]"
            ),
            (
                "Hola, ¿me puedes ayudar con el despliegue?",
                None,
                "Sure, the deploy is live now.",
                "es-MX",
                "[Voice direction: idiomatic Spanish as spoken in Mexico]"
            ),
            (
                "English text",
                "Hola ¿cómo estás?",
                "Sure, the deploy is live now.",
                "es-MX",
                "[Voice direction: idiomatic Spanish as spoken in Mexico]"
            ),
        ]
    )
    @pytest.mark.asyncio
    async def test_delivers_structured_part_and_reports_delivered_parametrized(
        self, tmp_path, inbound_text, override_inbound, reply_text, expected_locale, expected_prefix
    ):
        runner = _render_runner()
        event = _event(text=inbound_text)
        rendered = RenderedVoiceNote(path=str(tmp_path / "out.ogg"), locale=expected_locale)

        with patch(
            "tools.voice_reply.render_voice_note", return_value=rendered
        ) as render:
            delivered = await GatewayRunner._send_native_voice_note(
                runner, event, reply_text, inbound_text=override_inbound
            )

        assert delivered is True
        runner._deliver_voice_audio.assert_awaited_once()
        args = runner._deliver_voice_audio.call_args.args
        assert args[0] is event
        assert args[1] == rendered.path

        _, kwargs = render.call_args
        assert kwargs["locale"] == expected_locale
        spoken_arg = render.call_args.args[0]
        assert spoken_arg.startswith(expected_prefix)

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
    async def test_truncated_long_reply_returns_false_to_keep_text(self, tmp_path):
        # A reply longer than the TTS cap is spoken only up to the cap. The
        # voice note still ships (head-start), but the method must report NOT
        # spoken so the caller keeps the FULL text reply — the user must never
        # silently lose the tail of a long message.
        runner = _render_runner()
        event = _event()
        long_text = "word " * 1200  # 6000 chars > 4000-char cap
        rendered = RenderedVoiceNote(path=str(tmp_path / "out.ogg"), locale="en-US")

        with patch(
            "tools.voice_reply.render_voice_note", return_value=rendered
        ) as render:
            delivered = await GatewayRunner._send_native_voice_note(
                runner, event, long_text
            )

        assert delivered is False
        # The voice note is still rendered and delivered (partial head-start),
        # only the text-suppression signal is withheld.
        render.assert_called_once()
        runner._deliver_voice_audio.assert_awaited_once()
        # Only the first 4000 chars reach TTS.
        spoken_arg = render.call_args.args[0]
        assert len(spoken_arg) <= 4000 + len(
            "[Voice direction: idiomatic English as spoken in the United States]\n"
        )

    @pytest.mark.asyncio
    async def test_reply_at_cap_boundary_still_spoken(self, tmp_path):
        # Exactly at the cap is fully voiced — return True to suppress the text.
        runner = _render_runner()
        event = _event()
        at_cap = "a" * 4000
        rendered = RenderedVoiceNote(path=str(tmp_path / "out.ogg"), locale="en-US")

        with patch("tools.voice_reply.render_voice_note", return_value=rendered):
            delivered = await GatewayRunner._send_native_voice_note(
                runner, event, at_cap
            )

        assert delivered is True

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

    @pytest.mark.asyncio
    async def test_delivery_failure_returns_false(self, tmp_path):
        # Render succeeds but the adapter fails to deliver (e.g. send_voice
        # returns SendResult(success=False)): must report NOT delivered so the
        # caller falls back to the text reply instead of suppressing it.
        runner = SimpleNamespace(
            _native_audio_out_render_config=lambda: ("Charon", {}),
            _deliver_voice_audio=AsyncMock(return_value=False),
        )
        event = _event()
        rendered = RenderedVoiceNote(path=str(tmp_path / "out.ogg"), locale="en-US")

        with patch("tools.voice_reply.render_voice_note", return_value=rendered):
            delivered = await GatewayRunner._send_native_voice_note(
                runner, event, "Hello there"
            )

        assert delivered is False
        runner._deliver_voice_audio.assert_awaited_once()


def _voice_delivery_runner(adapter):
    return SimpleNamespace(
        adapters={Platform.WHATSAPP_CLOUD: adapter},
        _get_guild_id=lambda event: None,
        _reply_anchor_for_event=lambda event: "anchor-1",
        _thread_metadata_for_source=lambda source, anchor: {"thread_id": "t1"},
    )


class TestDeliverVoiceAudio:
    @pytest.mark.asyncio
    async def test_returns_true_when_send_voice_succeeds(self):
        adapter = SimpleNamespace(
            send_voice=AsyncMock(
                return_value=SendResult(success=True, message_id="m1")
            )
        )
        runner = _voice_delivery_runner(adapter)
        ok = await GatewayRunner._deliver_voice_audio(runner, _event(), "/tmp/x.ogg")
        assert ok is True
        adapter.send_voice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_send_voice_reports_failure(self):
        # Adapters report transport failures via SendResult(success=False)
        # WITHOUT raising — delivery must be reported as failed so the caller
        # keeps the text fallback.
        adapter = SimpleNamespace(
            send_voice=AsyncMock(
                return_value=SendResult(success=False, error="upload rejected")
            )
        )
        runner = _voice_delivery_runner(adapter)
        ok = await GatewayRunner._deliver_voice_audio(runner, _event(), "/tmp/x.ogg")
        assert ok is False

    @pytest.mark.asyncio
    async def test_returns_true_when_send_voice_returns_none(self):
        # Older adapters return None on success; don't over-suppress the text.
        adapter = SimpleNamespace(send_voice=AsyncMock(return_value=None))
        runner = _voice_delivery_runner(adapter)
        ok = await GatewayRunner._deliver_voice_audio(runner, _event(), "/tmp/x.ogg")
        assert ok is True

    @pytest.mark.asyncio
    async def test_returns_false_when_no_voice_capability(self):
        adapter = SimpleNamespace()  # neither send_voice nor a voice channel
        runner = _voice_delivery_runner(adapter)
        ok = await GatewayRunner._deliver_voice_audio(runner, _event(), "/tmp/x.ogg")
        assert ok is False


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

    @pytest.mark.asyncio
    async def test_no_text_send_capability_warns_not_silent(self, caplog):
        # Voice-primary stripped the URL from the spoken note and suppressed the
        # text, so a link that can't be sent is lost — it must warn, not drop
        # silently.
        adapter = SimpleNamespace(name="voice_only")  # no `send`
        runner = SimpleNamespace(
            adapters={Platform.WHATSAPP_CLOUD: adapter},
            _reply_anchor_for_event=lambda event: "anchor-1",
            _thread_metadata_for_source=lambda source, anchor: {},
        )
        event = _event()
        with caplog.at_level("WARNING"):
            await GatewayRunner._send_link_followup(
                runner, event, "grab it at https://example.com/x"
            )
        assert "could not be delivered" in caplog.text


class TestNativeAudioOutEnabled:
    def test_new_tts_reply_key(self):
        runner = SimpleNamespace()
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"voice": {"tts_reply": True}},
        ):
            assert GatewayRunner._native_audio_out_enabled(runner) is True
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"voice": {"tts_reply": False}},
        ):
            assert GatewayRunner._native_audio_out_enabled(runner) is False

    def test_legacy_native_audio_out_key_still_read(self):
        # Backwards compatibility: configs written before the rename still work.
        runner = SimpleNamespace()
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"voice": {"native_audio_out": True}},
        ):
            assert GatewayRunner._native_audio_out_enabled(runner) is True
        with patch("hermes_cli.config.load_config_readonly", return_value={"voice": {}}):
            assert GatewayRunner._native_audio_out_enabled(runner) is False

    def test_new_key_takes_precedence_over_legacy(self):
        runner = SimpleNamespace()
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"voice": {"tts_reply": False, "native_audio_out": True}},
        ):
            assert GatewayRunner._native_audio_out_enabled(runner) is False
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"voice": {"tts_reply": True, "native_audio_out": False}},
        ):
            assert GatewayRunner._native_audio_out_enabled(runner) is True

    def test_render_config_defaults_to_charon(self):
        runner = SimpleNamespace()
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"tts": {"gemini": {"model": "m"}}},
        ):
            voice, cfg = GatewayRunner._native_audio_out_render_config(runner)
        assert voice == "Charon"
        assert cfg == {"model": "m"}
