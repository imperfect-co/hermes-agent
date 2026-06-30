"""Gateway-level tests for native-audio routing: buffer + STT bypass.

Drives ``_prepare_inbound_message_text`` directly (no live gateway) the same
way test_native_image_buffer_isolation.py exercises the image path.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _make_runner(audio_mode: str = "native") -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "gemini/gemini-3.5-flash"
    runner._base_url = None
    runner._decide_image_input_mode = lambda: "text"
    # Stub the routing decision so the test doesn't depend on models.dev.
    runner._decide_audio_input_mode = lambda paths: (audio_mode, None)
    # Trap STT so we can assert it was/wasn't invoked.
    runner._enrich_message_with_transcription = AsyncMock(
        return_value=('[transcribed] hello', ["hello"])
    )
    return runner


def _source(chat_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="private",
        user_name=f"user-{chat_id}",
    )


def _voice_event(source: SessionSource, path: str) -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=[path],
        media_types=["audio/ogg"],
    )


@pytest.mark.asyncio
async def test_native_audio_buffers_paths_and_skips_stt():
    runner = _make_runner(audio_mode="native")
    source = _source("chat-a")

    await runner._prepare_inbound_message_text(
        event=_voice_event(source, "/tmp/voice.ogg"),
        source=source,
        history=[],
    )

    # Buffered for inline attachment at the run_conversation call site …
    assert runner._consume_pending_native_audio_paths(build_session_key(source)) == [
        "/tmp/voice.ogg"
    ]
    # … and STT never ran.
    runner._enrich_message_with_transcription.assert_not_called()


@pytest.mark.asyncio
async def test_stt_mode_transcribes_and_does_not_buffer():
    runner = _make_runner(audio_mode="stt")
    source = _source("chat-b")

    await runner._prepare_inbound_message_text(
        event=_voice_event(source, "/tmp/voice.ogg"),
        source=source,
        history=[],
    )

    runner._enrich_message_with_transcription.assert_awaited_once()
    assert runner._consume_pending_native_audio_paths(build_session_key(source)) == []


@pytest.mark.asyncio
async def test_native_audio_buffer_isolated_per_session():
    runner = _make_runner(audio_mode="native")
    source_a = _source("chat-a")
    source_b = _source("chat-b")

    await runner._prepare_inbound_message_text(
        event=_voice_event(source_a, "/tmp/a.ogg"),
        source=source_a,
        history=[],
    )
    await runner._prepare_inbound_message_text(
        event=_voice_event(source_b, "/tmp/b.ogg"),
        source=source_b,
        history=[],
    )

    assert runner._consume_pending_native_audio_paths(build_session_key(source_a)) == [
        "/tmp/a.ogg"
    ]
    assert runner._consume_pending_native_audio_paths(build_session_key(source_b)) == [
        "/tmp/b.ogg"
    ]


@pytest.mark.asyncio
async def test_text_only_turn_buffers_nothing():
    runner = _make_runner(audio_mode="native")
    source = _source("chat-c")

    await runner._prepare_inbound_message_text(
        event=MessageEvent(text="plain text", source=source),
        source=source,
        history=[],
    )

    assert runner._consume_pending_native_audio_paths(build_session_key(source)) == []
    runner._enrich_message_with_transcription.assert_not_called()
