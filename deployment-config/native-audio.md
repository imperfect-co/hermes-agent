# Voice-note replies (native audio out) — deployment config

Hermes can answer with a **spoken voice note** (opus-in-ogg) *in place of* the
plain text reply. The brain (`gemini-3.5-flash`) writes the reply text as usual,
then the gateway renders that text to a native voice-note bubble via Gemini TTS
(`Charon`, ADR 0024 Phase 1). When rendering succeeds the voice note is
voice-primary: the duplicate text is **not** re-sent — the only text that
follows is a short link follow-up (see below). Text is used as a fallback only
when the render fails.

## Enabling it

```yaml
voice:
  tts_reply: true          # turn voice-note replies on (default: false)
  native_audio_voice: Charon   # prebuilt Gemini voice used for the reply
```

> **Config key rename.** `voice.tts_reply` was previously called
> `voice.native_audio_out`. The old key is still read for backwards
> compatibility: if `voice.tts_reply` is absent, the gateway falls back to
> `voice.native_audio_out`. When both are present, `tts_reply` wins. Migrate
> your config to `tts_reply` at your convenience.

The flag is read fresh on every turn, so a config reload takes effect without a
restart. It is independent of `voice.auto_tts` and the `/voice` toggle.

## When a voice note is sent

With `tts_reply` on, the gateway speaks the reply when **either**:

- the inbound message was itself a voice note, **or**
- the user explicitly asked for a spoken reply (e.g. "send me a voice note",
  "respóndeme en voz alta", "háblame").

If Gemini TTS fails to render (network error, missing `ffmpeg`, empty audio),
the gateway **falls back to the plain text reply** — the user always gets the
answer.

## Voice character

- **Voice:** `Charon` (warm, low-pitch), configurable via
  `voice.native_audio_voice`.
- **Speed:** natural **1x**. No speaking-rate reduction is applied — Charon
  reads at its native pace.

## Idiomatic accent / locale

The spoken accent is detected from the **inbound message** (what the user
wrote), not from our reply text — so a Spanish question gets a Spanish-voiced
answer even when the reply mixes in English product names.

Each detected language is steered toward the **idiomatic territory** our users
most often mean, and a non-spoken direction cue is prepended to the TTS text:

| Inbound language | Locale (`languageCode`) | Steering prefix prepended to the TTS text |
| --- | --- | --- |
| Spanish | `es-MX` (Mexico, not Castilian `es-ES`) | `[Voice direction: idiomatic Spanish as spoken in Mexico]` |
| Farsi / Persian | `fa-IR` (Iran) | `[Voice direction: idiomatic Farsi as spoken in Iran]` |
| English / other (default) | `en-US` | `[Voice direction: idiomatic English as spoken in the United States]` |

Gemini treats the leading bracketed `[Voice direction: …]` as delivery style,
so it steers the accent without being read aloud. To pin a fixed locale for the
standalone TTS tool instead of per-reply detection, set `tts.gemini.language_code`.

## Voice + text combo (links)

A spoken voice note can't convey a URL. When the reply contains a link:

1. The **URL is stripped from the spoken text** (never read aloud — Charon would
   spell out an unusable string).
2. The voice note is delivered as the reply.
3. A **short follow-up text message carrying the link(s)** is sent so the user
   has a tappable URL.

For a plain reply with no link, only the voice note is sent (voice-primary — the
spoken note *is* the reply, so the duplicate text is suppressed). Media
attachments (images, files) are still delivered as native attachments in all
cases.

## Requirements

- `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) set for the TTS render.
- `ffmpeg` with `libopus` on `PATH` to transcode PCM → opus-in-ogg (the green
  voice-note bubble on WhatsApp/Telegram). Without it, the render fails loud and
  the gateway falls back to the text reply.

## Related code

- `tools/voice_reply.py` — locale/idiomatic detection, URL extraction, render.
- `gateway/run.py` — selection policy (`_should_send_voice_reply`), render +
  delivery (`_send_native_voice_note`), link follow-up (`_send_link_followup`).
- `hermes_cli/config.py` — `voice.tts_reply` / `voice.native_audio_voice`.
