"""Routing helpers for inbound user-attached audio (voice notes / audio files).

Two modes, mirroring :mod:`agent.image_routing`:

  native — attach the raw audio as an OpenAI-style ``input_audio`` content
           part on the user turn. The provider adapter translates it into the
           vendor's multimodal format (Gemini ``inlineData``). The model hears
           the bytes: prosody, emotion, hesitation, code-switching, background
           audio — signal a transcript discards.

  stt    — transcribe the audio up front (faster-whisper / Groq / OpenAI /
           Mistral / ElevenLabs) and prepend ``Here's what they said: "…"`` to
           the user's text. The model never hears the audio. This is the
           pre-existing behaviour and the right choice for non-audio models.

The decision is made once per turn by :func:`decide_audio_input_mode`, reading
``stt.native_audio`` from config (``auto`` | ``always`` | ``never``, default
``auto``) and the active model's ``supports_audio_input`` capability metadata.

In ``auto`` mode native is chosen only when the active model reports
``audio`` among its models.dev input modalities; otherwise we fall back to STT.
``always`` forces native regardless of capability (useful for custom/local
models whose metadata is missing); ``never`` preserves STT-only behaviour.

Size/duration guards (:func:`exceeds_native_audio_limits`) let an oversized
clip fall back to STT even when native is selected — audio tokens can dwarf a
transcript on long clips, and Gemini's inline ``inlineData`` has a hard ceiling.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_VALID_MODES = frozenset({"auto", "always", "never"})

# Default guards (also expressed in the ``stt:`` config block). A clip past
# either ceiling falls back to STT. 20 MB is Gemini's documented inline-audio
# ceiling; larger clips need the Files API / GCS URI path (future work).
DEFAULT_NATIVE_AUDIO_MAX_SECONDS = 600
DEFAULT_NATIVE_AUDIO_MAX_BYTES = 20 * 1024 * 1024  # 20 MiB

# Extension → OpenAI ``input_audio`` format token. The adapter maps the token
# to a MIME type; keeping the producer in OpenAI-style keeps every adapter's
# translation uniform (image parts already work this way via ``image_url``).
_EXT_TO_FORMAT = {
    ".ogg": "ogg",
    ".oga": "ogg",
    ".opus": "ogg",
    ".mp3": "mp3",
    ".mpga": "mp3",
    ".wav": "wav",
    ".m4a": "m4a",
    ".mp4": "m4a",
    ".aac": "aac",
    ".flac": "flac",
    ".webm": "webm",
    ".aif": "aiff",
    ".aiff": "aiff",
}

_FORMAT_TO_MIME = {
    "ogg": "audio/ogg",
    "mp3": "audio/mp3",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "webm": "audio/webm",
    "aiff": "audio/aiff",
}


def _coerce_mode(raw: Any) -> str:
    """Normalize a config value into one of the valid modes."""
    if not isinstance(raw, str):
        return "auto"
    val = raw.strip().lower()
    if val in _VALID_MODES:
        return val
    return "auto"


def _read_native_audio_mode(cfg: Optional[Dict[str, Any]]) -> str:
    if not isinstance(cfg, dict):
        return "auto"
    stt_cfg = cfg.get("stt")
    if not isinstance(stt_cfg, dict):
        return "auto"
    return _coerce_mode(stt_cfg.get("native_audio"))


def _lookup_supports_audio(
    provider: str,
    model: str,
) -> Optional[bool]:
    """Return True/False if the model's audio-input capability is known.

    Resolves via ``agent.models_dev.get_model_info(...).supports_audio_input()``
    (the same registry image routing uses for vision). Returns None when the
    model isn't in the catalog so the caller can fall back to STT in ``auto``.
    """
    if not provider or not model:
        return None
    try:
        from agent.models_dev import get_model_info

        info = get_model_info(provider, model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "audio_routing: caps lookup failed for %s:%s — %s", provider, model, exc
        )
        return None
    if info is None:
        return None
    return bool(info.supports_audio_input())


# Providers whose adapter actually translates an OpenAI-style ``input_audio``
# part into the vendor's native audio format. A model can report audio input in
# models.dev while the *adapter* on the active provider has no audio branch and
# would silently drop the part — routing native there loses the message. Today
# only the native Gemini adapter (agent/gemini_native_adapter.py) handles it.
# Other adapters can be added here as they grow the branch.
_NATIVE_AUDIO_PROVIDERS = frozenset({"gemini", "google"})


def _provider_supports_native_audio(provider: str) -> bool:
    """True when the active provider's adapter can ingest native audio parts."""
    return (provider or "").strip().lower() in _NATIVE_AUDIO_PROVIDERS


def decide_audio_input_mode(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]],
) -> str:
    """Return ``"native"`` or ``"stt"`` for the given turn.

    Args:
      provider: active inference provider ID (e.g. ``"gemini"``).
      model:    active model slug as sent to the provider.
      cfg:      loaded config dict, or None. When None, behaves as ``auto``.

    Decision table:
      * ``never``  → always ``stt`` (current default behaviour preserved).
      * ``always`` → ``native`` when the active provider's adapter handles audio
        (the caller asserts the model can hear); else ``stt``, because routing
        native to an adapter without an audio branch would silently drop the
        bytes — worse than transcribing.
      * ``auto``   → ``native`` iff the model reports audio input support **and**
        the provider's adapter handles native audio; else ``stt``.
    """
    mode = _read_native_audio_mode(cfg)
    if mode == "never":
        return "stt"
    if not _provider_supports_native_audio(provider):
        return "stt"
    if mode == "always":
        return "native"

    # auto
    supports = _lookup_supports_audio(provider, model)
    if supports is True:
        return "native"
    return "stt"


def _read_int_guard(cfg: Optional[Dict[str, Any]], key: str, default: int) -> int:
    if not isinstance(cfg, dict):
        return default
    stt_cfg = cfg.get("stt")
    if not isinstance(stt_cfg, dict):
        return default
    raw = stt_cfg.get(key)
    if isinstance(raw, bool):  # bool is an int subclass — reject explicitly
        return default
    if isinstance(raw, (int, float)) and raw > 0:
        return int(raw)
    return default


def native_audio_max_bytes(cfg: Optional[Dict[str, Any]]) -> int:
    return _read_int_guard(cfg, "native_audio_max_bytes", DEFAULT_NATIVE_AUDIO_MAX_BYTES)


def native_audio_max_seconds(cfg: Optional[Dict[str, Any]]) -> int:
    return _read_int_guard(cfg, "native_audio_max_seconds", DEFAULT_NATIVE_AUDIO_MAX_SECONDS)


def probe_audio_duration_seconds(path: str) -> Optional[float]:
    """Best-effort clip duration in seconds. Returns None when it can't be read.

    Cheap, dependency-light, and synchronous: stdlib ``wave`` for WAV, ``mutagen``
    for OGG/Opus (already used by the gateway). Anything else returns None so the
    duration guard simply doesn't fire — the byte ceiling still applies. No
    ffprobe subprocess here; the byte guard is the primary cost/size protection.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".wav":
            import wave

            with wave.open(path, "rb") as wf:
                rate = wf.getframerate() or 0
                if rate <= 0:
                    return None
                return wf.getnframes() / float(rate)
        if ext in (".ogg", ".oga", ".opus"):
            from mutagen.oggopus import OggOpus

            return float(OggOpus(path).info.length)
    except Exception:
        return None
    return None


def exceeds_native_audio_limits(
    audio_paths: List[str],
    cfg: Optional[Dict[str, Any]],
    *,
    durations: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """Return a human-readable reason if any clip is too big for native, else None.

    Checks both guards:
      * byte ceiling — cheap ``os.path.getsize`` (no decode).
      * duration ceiling — uses ``durations`` (path → seconds) when the caller
        supplies it, else falls back to :func:`probe_audio_duration_seconds`.
        A clip whose duration can't be probed is allowed through on duration
        (the byte ceiling still guards cost/size).
    """
    max_bytes = native_audio_max_bytes(cfg)
    max_seconds = native_audio_max_seconds(cfg)
    for raw_path in audio_paths:
        try:
            size = os.path.getsize(raw_path)
        except OSError:
            continue
        if size > max_bytes:
            return f"clip {os.path.basename(raw_path)} is {size}B > {max_bytes}B native ceiling"
        dur = None
        if durations is not None:
            dur = durations.get(raw_path)
        if dur is None:
            dur = probe_audio_duration_seconds(raw_path)
        if isinstance(dur, (int, float)) and dur > max_seconds:
            return (
                f"clip {os.path.basename(raw_path)} is {dur:.0f}s "
                f"> {max_seconds}s native ceiling"
            )
    return None


def _guess_audio_format(path: Path) -> str:
    """Map a path suffix to an OpenAI ``input_audio`` format token."""
    fmt = _EXT_TO_FORMAT.get(path.suffix.lower())
    if fmt:
        return fmt
    mime, _ = mimetypes.guess_type(str(path))
    if isinstance(mime, str) and mime.startswith("audio/"):
        sub = mime.split("/", 1)[1].lower()
        # Normalize a few MIME subtypes to our tokens.
        return {"mpeg": "mp3", "x-wav": "wav", "x-m4a": "m4a"}.get(sub, sub)
    # WhatsApp voice notes are Opus-in-OGG; default there.
    return "ogg"


def build_native_audio_content_parts(
    user_text: str,
    audio_paths: List[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build an OpenAI-style ``content`` list carrying native audio parts.

    Shape (mirrors :func:`agent.image_routing.build_native_content_parts`):

      [{"type": "text", "text": "...\\n\\n[Voice message attached: /path]"},
       {"type": "input_audio", "input_audio": {"data": "<b64>", "format": "ogg"}},
       ...]

    Local paths are read from disk and embedded as base64. A text hint
    (``[Voice message attached: <path>]``) is appended so path-taking tools
    still have a handle on the clip — exactly as the image builder does.

    Returns ``(content_parts, skipped)``. Skipped entries are paths that
    couldn't be read. When nothing attaches, returns a plain text-only part
    (or an empty list) so the caller falls back cleanly.
    """
    skipped: List[str] = []
    audio_parts: List[Dict[str, Any]] = []
    attached_paths: List[str] = []

    for raw_path in audio_paths:
        p = Path(raw_path)
        try:
            if not p.exists() or not p.is_file():
                skipped.append(str(raw_path))
                continue
            raw = p.read_bytes()
        except OSError as exc:
            logger.warning("audio_routing: failed to read %s — %s", raw_path, exc)
            skipped.append(str(raw_path))
            continue
        if not raw:
            skipped.append(str(raw_path))
            continue
        fmt = _guess_audio_format(p)
        audio_parts.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(raw).decode("ascii"),
                    "format": fmt,
                },
            }
        )
        attached_paths.append(str(raw_path))

    text = (user_text or "").strip()

    if attached_paths:
        base_text = text or "Listen to this voice message and respond."
        hint_lines = [f"[Voice message attached: {p}]" for p in attached_paths]
        combined_text = f"{base_text}\n\n" + "\n".join(hint_lines)
        parts: List[Dict[str, Any]] = [{"type": "text", "text": combined_text}]
        parts.extend(audio_parts)
        return parts, skipped

    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    return parts, skipped


__all__ = [
    "decide_audio_input_mode",
    "build_native_audio_content_parts",
    "exceeds_native_audio_limits",
    "probe_audio_duration_seconds",
    "native_audio_max_bytes",
    "native_audio_max_seconds",
    "DEFAULT_NATIVE_AUDIO_MAX_SECONDS",
    "DEFAULT_NATIVE_AUDIO_MAX_BYTES",
]
