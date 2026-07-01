"""Tests for agent/audio_routing.py — per-turn native-audio-vs-STT routing."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import patch

from agent.audio_routing import (
    DEFAULT_NATIVE_AUDIO_MAX_BYTES,
    DEFAULT_NATIVE_AUDIO_MAX_SECONDS,
    _coerce_mode,
    _guess_audio_format,
    build_native_audio_content_parts,
    decide_audio_input_mode,
    exceeds_native_audio_limits,
    native_audio_max_bytes,
    native_audio_max_seconds,
    strip_audio_placeholder_caption,
)


# ─── _coerce_mode ────────────────────────────────────────────────────────────


class TestCoerceMode:
    def test_valid_modes_pass_through(self):
        assert _coerce_mode("auto") == "auto"
        assert _coerce_mode("always") == "always"
        assert _coerce_mode("never") == "never"

    def test_case_insensitive_and_strip(self):
        assert _coerce_mode("ALWAYS") == "always"
        assert _coerce_mode("  never ") == "never"

    def test_invalid_falls_back_to_auto(self):
        assert _coerce_mode("nonsense") == "auto"
        assert _coerce_mode("") == "auto"
        assert _coerce_mode(None) == "auto"
        assert _coerce_mode(42) == "auto"


# ─── decide_audio_input_mode (decision table) ────────────────────────────────


class TestDecideAudioInputMode:
    def test_never_forces_stt_even_for_audio_model(self):
        with patch("agent.audio_routing._lookup_supports_audio", return_value=True):
            assert (
                decide_audio_input_mode(
                    "gemini", "gemini-3.5-flash", {"stt": {"native_audio": "never"}}
                )
                == "stt"
            )

    def test_always_forces_native_on_supported_provider_even_for_unknown_model(self):
        with patch("agent.audio_routing._lookup_supports_audio", return_value=None):
            assert (
                decide_audio_input_mode(
                    "gemini", "my-experimental-model", {"stt": {"native_audio": "always"}}
                )
                == "native"
            )

    def test_always_still_stt_on_provider_without_audio_adapter(self):
        # An adapter that can't translate input_audio would silently drop the
        # bytes; "always" must not route native there.
        with patch("agent.audio_routing._lookup_supports_audio", return_value=True):
            assert (
                decide_audio_input_mode(
                    "openai", "gpt-4o-audio", {"stt": {"native_audio": "always"}}
                )
                == "stt"
            )

    def test_auto_native_when_model_supports_audio(self):
        with patch("agent.audio_routing._lookup_supports_audio", return_value=True):
            assert (
                decide_audio_input_mode(
                    "gemini", "gemini-3.5-flash", {"stt": {"native_audio": "auto"}}
                )
                == "native"
            )

    def test_auto_stt_when_provider_adapter_lacks_audio_support(self):
        # Even an audio-capable model on a non-Gemini provider routes to STT in
        # auto, because that provider's adapter has no input_audio branch.
        with patch("agent.audio_routing._lookup_supports_audio", return_value=True):
            assert (
                decide_audio_input_mode("openai", "gpt-4o-audio", {"stt": {"native_audio": "auto"}})
                == "stt"
            )

    def test_auto_stt_when_model_lacks_audio(self):
        with patch("agent.audio_routing._lookup_supports_audio", return_value=False):
            assert (
                decide_audio_input_mode("gemini", "gemini-1.0", {"stt": {"native_audio": "auto"}})
                == "stt"
            )

    def test_auto_stt_when_capability_unknown(self):
        with patch("agent.audio_routing._lookup_supports_audio", return_value=None):
            assert (
                decide_audio_input_mode("gemini", "y", {"stt": {"native_audio": "auto"}}) == "stt"
            )

    def test_google_provider_alias_supported(self):
        with patch("agent.audio_routing._lookup_supports_audio", return_value=True):
            assert (
                decide_audio_input_mode(
                    "google", "gemini-3.5-flash", {"stt": {"native_audio": "auto"}}
                )
                == "native"
            )

    def test_none_config_behaves_as_auto(self):
        with patch("agent.audio_routing._lookup_supports_audio", return_value=True):
            assert decide_audio_input_mode("gemini", "gemini-3.5-flash", None) == "native"

    def test_missing_stt_block_behaves_as_auto(self):
        with patch("agent.audio_routing._lookup_supports_audio", return_value=False):
            assert decide_audio_input_mode("gemini", "gemini-1.0", {"agent": {}}) == "stt"


# ─── capability lookup wiring (uses ModelInfo.supports_audio_input) ──────────


class TestLookupSupportsAudio:
    def test_reads_supports_audio_input_from_model_info(self):
        from agent.audio_routing import _lookup_supports_audio

        class _Info:
            def supports_audio_input(self):
                return True

        with patch("agent.models_dev.get_model_info", return_value=_Info()):
            assert _lookup_supports_audio("gemini", "gemini-3.5-flash") is True

    def test_unknown_model_returns_none(self):
        from agent.audio_routing import _lookup_supports_audio

        with patch("agent.models_dev.get_model_info", return_value=None):
            assert _lookup_supports_audio("gemini", "nope") is None

    def test_blank_args_short_circuit(self):
        from agent.audio_routing import _lookup_supports_audio

        assert _lookup_supports_audio("", "model") is None
        assert _lookup_supports_audio("gemini", "") is None


# ─── size/duration guards ────────────────────────────────────────────────────


class TestGuards:
    def test_defaults_when_no_config(self):
        assert native_audio_max_bytes(None) == DEFAULT_NATIVE_AUDIO_MAX_BYTES
        assert native_audio_max_seconds(None) == DEFAULT_NATIVE_AUDIO_MAX_SECONDS

    def test_config_overrides_guards(self):
        cfg = {"stt": {"native_audio_max_bytes": 123, "native_audio_max_seconds": 9}}
        assert native_audio_max_bytes(cfg) == 123
        assert native_audio_max_seconds(cfg) == 9

    def test_bool_is_rejected_as_guard(self):
        # bool is an int subclass; must not be honoured as a byte ceiling.
        cfg = {"stt": {"native_audio_max_bytes": True}}
        assert native_audio_max_bytes(cfg) == DEFAULT_NATIVE_AUDIO_MAX_BYTES

    def test_exceeds_byte_ceiling(self, tmp_path: Path):
        clip = tmp_path / "big.ogg"
        clip.write_bytes(b"x" * 100)
        reason = exceeds_native_audio_limits([str(clip)], {"stt": {"native_audio_max_bytes": 50}})
        assert reason is not None and "native ceiling" in reason

    def test_within_byte_ceiling_returns_none(self, tmp_path: Path):
        clip = tmp_path / "ok.ogg"
        clip.write_bytes(b"x" * 100)
        assert (
            exceeds_native_audio_limits([str(clip)], {"stt": {"native_audio_max_bytes": 200}})
            is None
        )

    def test_duration_ceiling_when_supplied(self, tmp_path: Path):
        clip = tmp_path / "long.ogg"
        clip.write_bytes(b"x" * 10)
        reason = exceeds_native_audio_limits(
            [str(clip)],
            {"stt": {"native_audio_max_seconds": 30}},
            durations={str(clip): 120.0},
        )
        assert reason is not None and "native ceiling" in reason

    def test_missing_file_is_skipped_not_crash(self):
        assert exceeds_native_audio_limits(["/nope/missing.ogg"], None) is None

    def test_duration_ceiling_via_auto_probe_on_wav(self, tmp_path: Path):
        # A real (silent) WAV longer than the configured ceiling must fall back
        # without the caller supplying durations — the helper probes it.
        import wave

        clip = tmp_path / "long.wav"
        rate = 8000
        seconds = 3
        with wave.open(str(clip), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(b"\x00\x00" * rate * seconds)
        reason = exceeds_native_audio_limits([str(clip)], {"stt": {"native_audio_max_seconds": 1}})
        assert reason is not None and "native ceiling" in reason

    def test_unprobeable_duration_passes(self, tmp_path: Path):
        # A format we can't probe (e.g. .mp3 without ffprobe in this helper)
        # is allowed through on duration; the byte ceiling still guards.
        clip = tmp_path / "x.mp3"
        clip.write_bytes(b"\x00" * 10)
        assert exceeds_native_audio_limits([str(clip)], {"stt": {"native_audio_max_seconds": 1}}) is None


class TestProbeAudioDurationSeconds:
    def test_wav_duration(self, tmp_path: Path):
        import wave

        from agent.audio_routing import probe_audio_duration_seconds

        clip = tmp_path / "c.wav"
        with wave.open(str(clip), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(4000)
            wf.writeframes(b"\x00\x00" * 4000 * 2)  # 2 seconds
        assert abs(probe_audio_duration_seconds(str(clip)) - 2.0) < 0.01

    def test_unknown_extension_returns_none(self, tmp_path: Path):
        from agent.audio_routing import probe_audio_duration_seconds

        clip = tmp_path / "c.bin"
        clip.write_bytes(b"abc")
        assert probe_audio_duration_seconds(str(clip)) is None


class TestProviderGate:
    def test_gemini_and_google_supported(self):
        from agent.audio_routing import _provider_supports_native_audio

        assert _provider_supports_native_audio("gemini") is True
        assert _provider_supports_native_audio("GOOGLE") is True

    def test_other_providers_unsupported(self):
        from agent.audio_routing import _provider_supports_native_audio

        assert _provider_supports_native_audio("openai") is False
        assert _provider_supports_native_audio("anthropic") is False
        assert _provider_supports_native_audio("") is False


# ─── _guess_audio_format ─────────────────────────────────────────────────────


class TestGuessAudioFormat:
    def test_common_extensions(self):
        assert _guess_audio_format(Path("a.ogg")) == "ogg"
        assert _guess_audio_format(Path("a.opus")) == "ogg"
        assert _guess_audio_format(Path("a.mp3")) == "mp3"
        assert _guess_audio_format(Path("a.wav")) == "wav"
        assert _guess_audio_format(Path("a.m4a")) == "m4a"

    def test_unknown_extension_defaults_to_ogg(self):
        assert _guess_audio_format(Path("a.totallyunknown")) == "ogg"

    def test_mimetypes_fallback_for_known_audio_extension(self):
        # .au isn't in _EXT_TO_FORMAT but mimetypes knows it (audio/basic),
        # so the mimetypes branch resolves it rather than defaulting to ogg.
        assert _guess_audio_format(Path("clip.au")) == "basic"

    def test_mp3_mime_subtype_normalized(self):
        assert _guess_audio_format(Path("clip.mpga")) == "mp3"


# ─── build_native_audio_content_parts ────────────────────────────────────────


class TestBuildNativeAudioContentParts:
    def test_emits_text_then_input_audio(self, tmp_path: Path):
        clip = tmp_path / "voice.ogg"
        clip.write_bytes(b"oggbytes")
        parts, skipped = build_native_audio_content_parts("how do I do X?", [str(clip)])
        assert skipped == []
        assert parts[0]["type"] == "text"
        assert "how do I do X?" in parts[0]["text"]
        assert f"[Voice message attached: {clip}]" in parts[0]["text"]
        assert parts[1]["type"] == "input_audio"
        assert parts[1]["input_audio"]["format"] == "ogg"
        assert parts[1]["input_audio"]["data"] == base64.b64encode(b"oggbytes").decode("ascii")

    def test_empty_caption_gets_default_prompt(self, tmp_path: Path):
        clip = tmp_path / "voice.wav"
        clip.write_bytes(b"RIFFdata")
        parts, _ = build_native_audio_content_parts("", [str(clip)])
        assert parts[0]["type"] == "text"
        assert parts[0]["text"].startswith("Listen to this voice message")
        # No raw cache path leaks into the model-visible text when there is no
        # caption — the path hint is parrot-bait the model echoes back verbatim.
        assert "[Voice message attached:" not in parts[0]["text"]
        assert str(clip) not in parts[0]["text"]

    def test_placeholder_caption_is_treated_as_captionless(self, tmp_path: Path):
        """A captionless WhatsApp voice note (the reported bug).

        The Baileys bridge fills body with ``[audio received]`` for a
        captionless clip. That transport placeholder must NOT reach the model
        as the user's text, and the raw cache path must NOT be injected — both
        get parroted straight back as the reply ("[audio received] [Voice
        message attached: ]0:09:00"). The turn must be a clean instruction plus
        the audio bytes only.
        """
        clip = tmp_path / "aud_deadbeef.ogg"
        clip.write_bytes(b"oggbytes")
        parts, skipped = build_native_audio_content_parts(
            "[audio received]", [str(clip)]
        )
        assert skipped == []
        text_part = parts[0]
        assert text_part["type"] == "text"
        # No leaked metadata of any kind.
        assert "[audio received]" not in text_part["text"]
        assert "[Voice message attached:" not in text_part["text"]
        assert str(clip) not in text_part["text"]
        assert text_part["text"] == "Listen to this voice message and respond."
        # The audio bytes are still attached natively.
        assert parts[1]["type"] == "input_audio"

    def test_missing_path_is_skipped(self, tmp_path: Path):
        good = tmp_path / "ok.ogg"
        good.write_bytes(b"data")
        parts, skipped = build_native_audio_content_parts("hi", [str(good), "/nope.ogg"])
        assert skipped == ["/nope.ogg"]
        assert sum(1 for p in parts if p["type"] == "input_audio") == 1

    def test_all_unreadable_falls_back_to_text_only(self):
        parts, skipped = build_native_audio_content_parts("just text", ["/nope.ogg"])
        assert skipped == ["/nope.ogg"]
        assert parts == [{"type": "text", "text": "just text"}]

    def test_empty_file_is_skipped(self, tmp_path: Path):
        clip = tmp_path / "empty.ogg"
        clip.write_bytes(b"")
        parts, skipped = build_native_audio_content_parts("hi", [str(clip)])
        assert skipped == [str(clip)]
        assert all(p["type"] != "input_audio" for p in parts)

    def test_unreadable_file_is_skipped(self, tmp_path: Path):
        # A path that exists as a directory triggers the read OSError branch.
        d = tmp_path / "adir.ogg"
        d.mkdir()
        parts, skipped = build_native_audio_content_parts("hi", [str(d)])
        assert skipped == [str(d)]
        assert all(p["type"] != "input_audio" for p in parts)

    def test_format_inferred_from_extension(self, tmp_path: Path):
        clip = tmp_path / "note.m4a"
        clip.write_bytes(b"m4adata")
        parts, _ = build_native_audio_content_parts("hi", [str(clip)])
        audio = next(p for p in parts if p["type"] == "input_audio")
        assert audio["input_audio"]["format"] == "m4a"


# ─── strip_audio_placeholder_caption ─────────────────────────────────────────


class TestStripAudioPlaceholderCaption:
    def test_strips_bridge_audio_placeholder(self):
        assert strip_audio_placeholder_caption("[audio received]") == ""

    def test_strips_ptt_placeholder(self):
        # The bridge emits "[ptt received]" for a push-to-talk voice note.
        assert strip_audio_placeholder_caption("[ptt received]") == ""

    def test_strips_with_surrounding_whitespace_and_case(self):
        assert strip_audio_placeholder_caption("  [Audio Received]  ") == ""

    def test_keeps_real_caption(self):
        assert strip_audio_placeholder_caption("what did I just record?") == (
            "what did I just record?"
        )

    def test_keeps_caption_that_merely_mentions_audio(self):
        # Only a lone placeholder is stripped, not any text containing the word.
        text = "the audio received was garbled"
        assert strip_audio_placeholder_caption(text) == text

    def test_none_and_empty_normalise_to_empty_string(self):
        assert strip_audio_placeholder_caption(None) == ""
        assert strip_audio_placeholder_caption("   ") == ""
