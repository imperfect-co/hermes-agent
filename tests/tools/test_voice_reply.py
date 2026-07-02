"""Tests for tools/voice_reply.py — voice-note replies (tts_reply, ADR 0024)."""

from __future__ import annotations

import base64
import contextlib
import os
from unittest.mock import MagicMock, patch

import pytest

from tools.voice_reply import (
    RenderedVoiceNote,
    VoiceProfile,
    VoiceRenderError,
    detect_voice_language,
    detect_voice_locale,
    detect_voice_profile,
    extract_urls,
    render_voice_note,
    user_requested_spoken_reply,
)


# ---------------------------------------------------------------------------
# Locale detection — idiomatic-territory defaults (es-MX, fa-IR).
# ---------------------------------------------------------------------------
class TestDetectVoiceLocale:
    @pytest.mark.parametrize(
        "text",
        [
            "Claro, ¿cómo estás hoy?",
            "Hola, gracias por el mensaje, todo está muy bien",
            "El perro corre por la calle",
            "Mañana tengo una reunión",
        ],
    )
    def test_spanish_is_idiomatic_mexico(self, text):
        # Idiomatic default: Mexican Spanish, not Castilian es-ES.
        assert detect_voice_locale(text) == "es-MX"

    @pytest.mark.parametrize(
        "text",
        [
            "سلام، حال شما چطور است؟",
            "لطفا وضعیت استقرار را بررسی کن",
            "خیلی ممنون بابت پیام",
        ],
    )
    def test_farsi_is_idiomatic_iran(self, text):
        assert detect_voice_locale(text) == "fa-IR"

    @pytest.mark.parametrize(
        "text",
        [
            "Hey, can you check the deploy status?",
            "no, me too",  # ambiguous English/Spanish overlap stays English
            "lol that was funny out loud",
            "",
            "OK",
        ],
    )
    def test_english_default(self, text):
        assert detect_voice_locale(text) == "en-US"

    def test_custom_default(self):
        assert detect_voice_locale("hello there", default="fr-FR") == "fr-FR"

    @pytest.mark.parametrize(
        "text,language",
        [
            ("Hola, ¿cómo estás?", "es"),
            ("سلام دوست من", "fa"),
            ("hello there friend", "en"),
            ("", "en"),
        ],
    )
    def test_detect_voice_language(self, text, language):
        assert detect_voice_language(text) == language

    @pytest.mark.parametrize(
        "text",
        [
            "﻿hello there",  # leading BOM must not read as Farsi
            "﻿",             # bare BOM
            "plain english﻿",
        ],
    )
    def test_bom_is_not_farsi(self, text):
        # U+FEFF (BOM / zero-width no-break space) sits just past the Arabic
        # Presentation Forms-B letters; it must not trigger Farsi detection.
        assert detect_voice_language(text) == "en"
        assert detect_voice_locale(text) == "en-US"


# ---------------------------------------------------------------------------
# Idiomatic voice profile + non-spoken steering direction
# ---------------------------------------------------------------------------
class TestDetectVoiceProfile:
    def test_spanish_profile(self):
        profile = detect_voice_profile("Hola, ¿cómo estás? Todo está muy bien.")
        assert isinstance(profile, VoiceProfile)
        assert profile.locale == "es-MX"
        assert profile.language == "Spanish"
        assert profile.territory == "Mexico"
        assert profile.direction == (
            "[Voice direction: idiomatic Spanish as spoken in Mexico]"
        )

    def test_farsi_profile(self):
        profile = detect_voice_profile("سلام، حال شما چطور است؟")
        assert profile.locale == "fa-IR"
        assert profile.language == "Farsi"
        assert profile.territory == "Iran"
        assert profile.direction == (
            "[Voice direction: idiomatic Farsi as spoken in Iran]"
        )

    def test_english_profile(self):
        profile = detect_voice_profile("Hey, can you check the deploy status?")
        assert profile.locale == "en-US"
        assert profile.language == "English"
        assert profile.territory == "the United States"
        assert profile.direction == (
            "[Voice direction: idiomatic English as spoken in the United States]"
        )

    def test_empty_defaults_to_english(self):
        assert detect_voice_profile("").locale == "en-US"


# ---------------------------------------------------------------------------
# URL extraction (voice + text combo: links ride a follow-up text)
# ---------------------------------------------------------------------------
class TestExtractUrls:
    def test_extracts_and_dedupes_in_order(self):
        text = "see https://a.com/x and https://b.io then https://a.com/x again"
        assert extract_urls(text) == ["https://a.com/x", "https://b.io"]

    def test_trims_trailing_sentence_punctuation(self):
        assert extract_urls("visit https://example.com/path.") == [
            "https://example.com/path"
        ]
        assert extract_urls("(see http://x.io/a)") == ["http://x.io/a"]

    @pytest.mark.parametrize("text", ["", "no links here", "ftp://not-http.example"])
    def test_none_found(self, text):
        assert extract_urls(text) == []


# ---------------------------------------------------------------------------
# Explicit spoken-reply request detection
# ---------------------------------------------------------------------------
class TestUserRequestedSpokenReply:
    @pytest.mark.parametrize(
        "text",
        [
            "Can you send me a voice note?",
            "reply with audio please",
            "say it out loud",
            "send a voice message",
            "mándame un audio",
            "respóndeme en voz alta",
            "envíame una nota de voz",
            "háblame",
        ],
    )
    def test_positive(self, text):
        assert user_requested_spoken_reply(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "what's the weather today?",
            "the audio quality was bad on that call",
            "I voted in the election",
            "send me the report",
            "",
        ],
    )
    def test_negative(self, text):
        assert user_requested_spoken_reply(text) is False


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)


def _fake_gemini_response(pcm: bytes = b"\x00\x01" * 4800):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/L16;codec=pcm;rate=24000",
                                "data": base64.b64encode(pcm).decode(),
                            }
                        }
                    ]
                }
            }
        ]
    }
    return resp


def _fake_ffmpeg_transcode(cmd, *args, **kwargs):
    """Stand in for the ffmpeg subprocess on runners without ffmpeg.

    GHA bare runners have no system ffmpeg, so the real PCM->opus transcode
    can't run. The render path only needs the call to succeed and leave a
    non-empty opus-in-ogg file at the destination (the last token of the
    ffmpeg command). Write a minimal OggS/OpusHead container so the
    container-shape assertions hold without a real codec.
    """
    out_path = cmd[-1]
    with open(out_path, "wb") as fh:
        fh.write(b"OggS\x00\x02" + b"\x00" * 22 + b"OpusHead\x01\x01")
    result = MagicMock()
    result.returncode = 0
    result.stderr = b""
    return result


@contextlib.contextmanager
def _stub_ffmpeg():
    """Pretend ffmpeg is installed and stub the transcode subprocess.

    These tests validate the Gemini API payload and voice/locale overrides,
    not the audio codec, so they must pass even when ffmpeg is absent (e.g.
    on GHA bare runners). ``shutil.which`` is the module object shared by
    ``tools.voice_reply`` and ``tools.tts_tool``, so one patch covers both
    the pre-flight guard and the transcode lookup.
    """
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"), patch(
        "tools.tts_tool.subprocess.run", side_effect=_fake_ffmpeg_transcode
    ):
        yield


class TestRenderVoiceNote:
    def test_renders_opus_ogg_with_spec_payload(self, tmp_path):
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            return _fake_gemini_response()

        out = str(tmp_path / "reply.mp3")  # wrong ext on purpose; forced to .ogg
        with _stub_ffmpeg(), patch("requests.post", fake_post):
            part = render_voice_note("Hola, ¿cómo estás? Todo está muy bien.", out)

        assert isinstance(part, RenderedVoiceNote)
        assert part.path.endswith(".ogg")
        # Idiomatic default: Mexican Spanish (es-MX), not Castilian es-ES.
        assert part.locale == "es-MX"
        assert part.mime_type == "audio/ogg; codecs=opus"
        assert os.path.getsize(part.path) > 0

        # Real OggS/Opus container produced by ffmpeg libopus
        head = open(part.path, "rb").read(64)
        assert head[:4] == b"OggS"
        assert b"Opus" in head

        gc = captured["json"]["generationConfig"]
        assert gc["responseModalities"] == ["AUDIO"]
        assert (
            gc["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"]
            == "Charon"
        )
        assert gc["speechConfig"]["languageCode"] == "es-MX"
        # Natural 1x speed: no speaking-rate reduction is applied anywhere in
        # the payload (Charon reads at its native pace).
        import json as _json

        _payload = _json.dumps(captured["json"])
        assert "speakingRate" not in _payload
        assert "speaking_rate" not in _payload
        # The default 2.5-flash-preview-tts is a non-thinking model and rejects
        # thinkingConfig (HTTP 400), so it must be omitted for it.
        assert "thinkingConfig" not in gc
        assert "generativelanguage.googleapis.com/v1beta" in captured["url"]
        assert "gemini-2.5-flash-preview-tts" in captured["url"]

    def test_explicit_locale_and_voice_override(self, tmp_path):
        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _fake_gemini_response()

        out = str(tmp_path / "reply.ogg")
        with _stub_ffmpeg(), patch("requests.post", fake_post):
            part = render_voice_note(
                "Buenos días", out, voice="Kore", locale="en-US"
            )
        assert part.locale == "en-US"
        gc = captured["json"]["generationConfig"]
        assert gc["speechConfig"]["languageCode"] == "en-US"
        assert (
            gc["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"]
            == "Kore"
        )

    def test_empty_text_raises(self, tmp_path):
        with pytest.raises(VoiceRenderError):
            render_voice_note("   ", str(tmp_path / "x.ogg"))

    def test_http_error_raises_voice_render_error(self, tmp_path):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "boom"
        resp.json.return_value = {"error": {"message": "boom"}}
        with patch("requests.post", lambda *a, **k: resp):
            with pytest.raises(VoiceRenderError):
                render_voice_note("hello", str(tmp_path / "x.ogg"))

    def test_missing_ffmpeg_raises(self, tmp_path):
        with patch("tools.voice_reply.shutil.which", return_value=None):
            with pytest.raises(VoiceRenderError, match="ffmpeg"):
                render_voice_note("hello", str(tmp_path / "x.ogg"))
