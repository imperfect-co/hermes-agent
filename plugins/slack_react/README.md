# slack_react

Emoji reactions for **Slack** and **WhatsApp**, driven by a `[[react:EMOJI]]`
control directive. Lets the agent react to the triggering message with an emoji
(e.g. `:+1:` for "thanks" / "issue fixed") **instead of** or **in addition to**
a text reply.

## How it works

Two stock hooks, registered by `register(ctx)` in `__init__.py`:

1. **`pre_llm_call`** — on Slack/WhatsApp turns, injects a short *policy*
   teaching the model the reaction vocabulary and when a reaction alone is
   enough. (The model decides; the plugin only gives it the option.)

2. **`transform_llm_output`** — after the model replies, before it is sent:
   - scans the reply for `[[react:EMOJI]]` control directives,
   - fires the platform's native reaction on the **message that triggered this
     turn**, and
   - strips the directives, returning either the cleaned text (react **and**
     reply) or the literal `NO_REPLY` (react **only**).

`NO_REPLY` is the gateway's built-in intentional-silence token
(`gateway/response_filters.py`), so a react-only turn sends no text message and
no "no response generated" fallback.

## How the target message is resolved

The triggering message comes from the **per-turn** session context vars
`HERMES_SESSION_CHAT_ID` / `HERMES_SESSION_MESSAGE_ID`, which the gateway binds
from the current turn's source (`gateway/run.py` `_set_session_env`). This
requires each adapter to pass `message_id` into `build_source` — done for Slack
(`plugins/platforms/slack/adapter.py`) and WhatsApp
(`plugins/platforms/whatsapp/adapter.py`). It deliberately does **not** read
`SessionEntry.origin.message_id`, which is fixed at session creation and would
be stale (or empty) on later turns.

Reaction delivery reuses the live adapter on the gateway runner:

- **Slack** — `adapter._add_reaction(channel, ts, shortcode)` → `reactions.add`.
  Slack takes the shortcode verbatim (no colons).
- **WhatsApp** — `adapter.send_reaction(chat_id, message_id, emoji, ...)` →
  bridge `POST /react` → Baileys `react`. WhatsApp needs a **literal unicode
  emoji**, so shortcodes are mapped via `_SHORTCODE_TO_UNICODE`; in a group the
  sender JID is passed as `participant`.

Because the hook runs in a gateway **worker thread** while the WhatsApp HTTP
session is bound to the gateway event loop, reaction coroutines are scheduled
back onto that loop with `safe_schedule_threadsafe`; a private-loop fallback
keeps CLI/test contexts working.

## Directive format (what the model emits)

```text
[[react:EMOJI]]        # EMOJI = shortcode w/o colons: +1, tada, eyes, ...
```

- React only:        `[[react:+1]] NO_REPLY`  → 👍 added, no text sent
- React + reply:     `[[react:eyes]] Looking into it now.`
- Multiple:          `[[react:+1]][[react:tada]] thanks all`
- Skin tone:         `[[react:middle_finger::skin-tone-5]]`  → 🖕🏾

On WhatsApp, only shortcodes present in `_SHORTCODE_TO_UNICODE` are delivered
(the set advertised in the injected policy is covered); unmapped shortcodes are
skipped. On Slack any valid workspace shortcode works.

### Skin tones

Hand/person emoji (`+1`, `-1`, `raised_hands`, `pray`, `ok_hand`, `clap`,
`thumbsup`, `thumbsdown`, `middle_finger`) accept a Fitzpatrick skin tone via
Slack's native `base::skin-tone-N` suffix, where `N` is `2` (light) … `6`
(dark) — **brown is `skin-tone-5`** (🏾). Slack takes the composite name
verbatim; on WhatsApp the base emoji is suffixed with the matching modifier
codepoint (`_SKIN_TONE_UNICODE`). A skin tone requested on an emoji that can't
carry one (e.g. `tada`) is dropped and the base reacts plain.

## Enable

In `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - slack_react
```

Restart the gateway so the plugin loads at startup.

> **Deployment note.** A **user** plugin at `~/.hermes/plugins/slack_react/`
> overrides this bundled copy (later sources win on name collision). To pick up
> this version on a host that has the old user copy, remove
> `~/.hermes/plugins/slack_react/` (or sync this directory over it) and restart
> the gateway.

## Disable

Remove `slack_react` from `plugins.enabled` (or add it to `plugins.disabled`)
and restart the gateway.
