#!/usr/bin/env python3
"""Native audio output (ADR 0024, Phase 1).

Compose-then-render helpers for conversational *voice note* replies: the brain
(``gemini-3.5-flash``) writes the reply text, then this module renders that text
to an ``opus``-in-``ogg`` voice note via Gemini TTS and hands the gateway a
first-class structured part (:class:`RenderedVoiceNote`) to deliver — no
``MEDIA:`` / ``[[audio_as_voice]]`` magic strings.

Two small policy helpers decide *whether* to speak and *which locale* to render
in; :func:`render_voice_note` does the render and fails loud
(:class:`VoiceRenderError`) so the gateway can fall back to plain text.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Prebuilt Gemini voice for native audio output replies (warm, low-pitch).
DEFAULT_NATIVE_AUDIO_VOICE = "Charon"
# Default spoken locale when text-based detection is inconclusive.
DEFAULT_LOCALE = "en-US"
SPANISH_LOCALE = "es-ES"
# WhatsApp/Telegram render the green voice-note bubble for opus-in-ogg.
VOICE_NOTE_MIME = "audio/ogg; codecs=opus"


# ---------------------------------------------------------------------------
# Locale detection — keep Spanish replies from being read with a gringo accent.
# Phase 1 is a binary en-US / es-ES split; the heuristic is deliberately
# conservative (default to en-US) and easy to extend with more locales later.
# ---------------------------------------------------------------------------

# Characters that only occur in Spanish among the languages we target.
_SPANISH_ONLY_CHARS = ("ñ", "¿", "¡")
_SPANISH_ACCENTS = set("áéíóúü")
# Distinctly-Spanish function words. Ambiguous tokens that are also common
# English words (no, me, mi, son, a, ...) are intentionally excluded to avoid
# false positives on English sentences.
_SPANISH_STOPWORDS = frozenset(
    {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "del",
        "que", "qué", "y", "en", "con", "por", "para", "es", "está",
        "están", "esto", "esta", "este", "eso", "más", "pero", "como",
        "cómo", "muy", "sí", "hola", "gracias", "tú", "te", "nos",
        "porque", "cuando", "dónde", "quién", "hace", "tiene", "puedo",
        "quiero", "también", "ahora", "bien", "hoy", "mañana", "buenos",
        "buenas", "días", "noches", "vale", "nada", "todo", "eres", "soy",
        "somos", "tienes", "quieres", "puedes", "necesito",
    }
)
_WORD_RE = re.compile(r"[a-zñáéíóúü]+")


def detect_voice_locale(text: str, default: str = DEFAULT_LOCALE) -> str:
    """Best-effort BCP-47 locale for spoken output ("es-ES" or ``default``)."""
    if not text or not text.strip():
        return default
    lowered = text.lower()
    if any(ch in lowered for ch in _SPANISH_ONLY_CHARS):
        return SPANISH_LOCALE

    words = _WORD_RE.findall(lowered)
    if not words:
        return default
    distinct_hits = len({w for w in words if w in _SPANISH_STOPWORDS})
    has_accent = any(ch in _SPANISH_ACCENTS for ch in lowered)
    # Two distinct Spanish function words — or one plus a Spanish accent — is a
    # strong enough signal to switch locales.
    if distinct_hits >= 2 or (distinct_hits >= 1 and has_accent):
        return SPANISH_LOCALE
    return default


# ---------------------------------------------------------------------------
# Explicit "reply with a voice note" request detection (selection policy (b)).
# ---------------------------------------------------------------------------
_VOICE_REQUEST_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # English
        r"\b(voice|audio)\s+(note|message|memo|reply|response|recording)\b",
        r"\b(send|reply|respond|answer|record|leave|give)\b[^.?!\n]{0,40}\b(voice|audio)\b",
        r"\breply\b[^.?!\n]{0,20}\b(with|in|using|by|as)\b[^.?!\n]{0,15}\b(voice|audio|speech)\b",
        r"\b(say|read)\s+(it|this|that)\b[^.?!\n]{0,20}\b(out\s+loud|aloud)\b",
        r"\b(speak|talk)\s+(to\s+me|it\s+out)\b",
        # Spanish
        r"\b(nota|mensaje|memo)\s+de\s+voz\b",
        r"\b(m[aá]nd|env[ií]|grab|respond|contest|d[ií])\w*\b[^.?!\n]{0,40}\b(audio|voz)\b",
        r"\b(en|con|por)\s+(un\s+)?(audio|voz)\b",
        r"\ben\s+voz\s+alta\b",
        r"\bh[aá]blame\b",
    )
)


def user_requested_spoken_reply(text: str) -> bool:
    """True when the inbound text explicitly asks for a spoken/voice reply."""
    if not text:
        return False
    return any(pat.search(text) for pat in _VOICE_REQUEST_PATTERNS)


# ---------------------------------------------------------------------------
# Render — text -> opus-in-ogg voice note.
# ---------------------------------------------------------------------------
class VoiceRenderError(RuntimeError):
    """Raised when a reply cannot be rendered to a voice note.

    Callers should fall back to delivering the plain text reply and log loud.
    """


@dataclass
class RenderedVoiceNote:
    """A first-class structured outbound audio part (ADR 0024).

    The gateway hands ``path`` to the platform adapter's ``send_voice`` so the
    audio is delivered as a native voice note — never embedded as a ``MEDIA:``
    or ``[[audio_as_voice]]`` magic string in the reply text.
    """

    path: str
    locale: str
    mime_type: str = VOICE_NOTE_MIME


def render_voice_note(
    text: str,
    output_path: str,
    *,
    voice: str = DEFAULT_NATIVE_AUDIO_VOICE,
    locale: str | None = None,
    base_gemini_config: dict | None = None,
) -> RenderedVoiceNote:
    """Render ``text`` to an opus-in-ogg voice note via Gemini TTS.

    Args:
        text: Final reply text written by the brain.
        output_path: Destination path; forced to ``.ogg``.
        voice: Prebuilt Gemini voice name (default ``Charon``).
        locale: BCP-47 locale; auto-detected from ``text`` when omitted.
        base_gemini_config: The configured ``tts.gemini`` dict (model, base_url,
            persona, ...). Voice / locale / thinking-budget are overridden here.

    Returns:
        A :class:`RenderedVoiceNote` pointing at the opus-in-ogg file.

    Raises:
        VoiceRenderError: on empty input, missing ffmpeg, render failure, or an
            empty/absent output file.
    """
    spoken = (text or "").strip()
    if not spoken:
        raise VoiceRenderError("Cannot render an empty reply to a voice note")

    # Transcoding PCM -> WAV -> opus-in-ogg requires ffmpeg with libopus. Fail
    # loud here so the gateway falls back to text instead of shipping a WAV
    # mislabeled as opus (which WhatsApp would reject from the voice bubble).
    if not shutil.which("ffmpeg"):
        raise VoiceRenderError("ffmpeg not found on PATH; cannot encode opus voice note")

    if not output_path.lower().endswith(".ogg"):
        output_path = os.path.splitext(output_path)[0] + ".ogg"

    resolved_locale = (locale or detect_voice_locale(spoken)).strip() or DEFAULT_LOCALE

    gemini_cfg = dict(base_gemini_config or {})
    gemini_cfg.update(
        {
            "voice": voice,
            "language_code": resolved_locale,
            "thinking_budget": 0,
            # The reply text is already final; skip the audio-tag rewrite pass.
            "audio_tags": False,
        }
    )
    tts_config = {"provider": "gemini", "gemini": gemini_cfg}

    try:
        # Reuse the single Gemini render path (HTTP + PCM->WAV->opus transcode).
        from tools.tts_tool import _generate_gemini_tts

        result_path = _generate_gemini_tts(spoken, output_path, tts_config)
    except Exception as exc:  # noqa: BLE001 — normalize to a single failure type
        raise VoiceRenderError(f"Gemini voice render failed: {exc}") from exc

    if not result_path or not os.path.isfile(result_path) or os.path.getsize(result_path) == 0:
        raise VoiceRenderError("Gemini voice render produced no audio file")

    logger.info(
        "native_audio_out rendered voice note (%s, %s, %d bytes)",
        voice,
        resolved_locale,
        os.path.getsize(result_path),
    )
    return RenderedVoiceNote(path=result_path, locale=resolved_locale)
