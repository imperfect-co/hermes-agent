"""slack_react — emoji reactions for Slack and WhatsApp.

Problem
-------
When someone sends a lightweight message ("thanks!", "issue fixed", "lgtm") the
bot always spins up a normal text-reply turn. The natural response is a single
emoji reaction and *no* text. This plugin gives the model that option on Slack
and WhatsApp.

How it works
------------
Two stock hooks:

1. ``pre_llm_call`` — on Slack/WhatsApp turns, inject a short *policy* teaching
   the model the ``[[react:EMOJI]]`` control directive and when a reaction alone
   is enough. The model decides; the plugin only supplies the vocabulary.

2. ``transform_llm_output`` — after the model replies, scan for
   ``[[react:EMOJI]]`` directives, fire the platform's native reaction on the
   message that triggered this turn, strip the directives, and return either:
     * the cleaned text (react **and** reply), or
     * the literal ``NO_REPLY`` (react **instead of** reply). ``NO_REPLY`` is the
       gateway's built-in intentional-silence token
       (``gateway/response_filters.py``), so no text message is sent.

The triggering message is resolved from the per-turn session context vars
``HERMES_SESSION_CHAT_ID`` / ``HERMES_SESSION_MESSAGE_ID`` (set by the gateway
from the *current* turn's source — not the session origin, which is fixed at
session creation). Delivery reuses the live adapter on the gateway runner:
Slack's ``_add_reaction`` (``reactions.add``) or WhatsApp's ``send_reaction``
(Baileys ``react``). WhatsApp's API needs a literal unicode emoji, so Slack
shortcodes are mapped via ``_SHORTCODE_TO_UNICODE``.

Hand/person emoji may carry a skin tone via Slack's native
``base::skin-tone-N`` suffix (e.g. ``middle_finger::skin-tone-5`` → brown
\U0001F595\U0001F3FE). Slack takes that name verbatim; on WhatsApp the base
emoji is suffixed with the Fitzpatrick modifier (``_SKIN_TONE_UNICODE``). A tone
on an emoji that can't carry one (``_SKINTONE_CAPABLE``) is dropped.

Because the hook runs in a gateway worker thread while the WhatsApp adapter's
HTTP session is bound to the gateway event loop, reaction coroutines are
scheduled back onto that loop via ``safe_schedule_threadsafe``; a private-loop
fallback keeps CLI/test contexts working.

Enable in ``~/.hermes/config.yaml``::

    plugins:
      enabled:
        - slack_react
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Callable, List, Optional

logger = logging.getLogger("hermes.plugins.slack_react")

# Matches [[react:EMOJI]] — EMOJI is a shortcode, with or without the
# surrounding colons (e.g. "+1", ":+1:", "white_check_mark", "tada"). A hand or
# person emoji may carry Slack's native skin-tone suffix (e.g.
# "middle_finger::skin-tone-5" → brown 🖕🏾); the "::skin-tone-N" part is
# captured with the base so it survives stripping and can be split off later.
_DIRECTIVE_RE = re.compile(
    r"\[\[\s*react\s*:\s*:?([a-zA-Z0-9_+\-]+(?:::skin-tone-[2-6])?):?\s*\]\]",
    re.IGNORECASE,
)

# The gateway's canonical "say nothing" token. Kept in sync with
# gateway/response_filters.py::SILENT_REPLY_TOKEN.
_SILENT_TOKEN = "NO_REPLY"

# Platforms this plugin reacts on. Both expose a per-turn message id via the
# session context vars and a live adapter with a reaction method.
_SUPPORTED_PLATFORMS = ("slack", "whatsapp")

# Hard cap on distinct reactions fired per response — each is a sequential
# platform call, so bound it even if the model emits many directives.
_MAX_REACTIONS_PER_RESPONSE = 5

# Total wall-clock budget (seconds) for ALL reaction calls in one response.
# Reactions run synchronously in the turn-finalizer thread, so cap the
# cumulative blocking time — not 15s × N — even if individual calls hang.
_REACTION_TOTAL_BUDGET_S = 15.0

# WhatsApp's Baileys `react` payload takes a literal unicode emoji, not a Slack
# shortcode. Map the shortcodes advertised in the policy (plus a few common
# extras) so one directive vocabulary works across both platforms. Unmapped
# shortcodes are skipped on WhatsApp (Slack takes them verbatim).
_SHORTCODE_TO_UNICODE = {
    "+1": "\U0001F44D",            # 👍
    "thumbsup": "\U0001F44D",      # 👍
    "-1": "\U0001F44E",            # 👎
    "thumbsdown": "\U0001F44E",    # 👎
    "tada": "\U0001F389",          # 🎉
    "eyes": "\U0001F440",          # 👀
    "white_check_mark": "✅",  # ✅
    "heavy_check_mark": "✔️",  # ✔️
    "raised_hands": "\U0001F64C",  # 🙌
    "pray": "\U0001F64F",          # 🙏
    "heart": "❤️",       # ❤️
    "fire": "\U0001F525",          # 🔥
    "100": "\U0001F4AF",           # 💯
    "rocket": "\U0001F680",        # 🚀
    "ok_hand": "\U0001F44C",       # 👌
    "clap": "\U0001F44F",          # 👏
    "middle_finger": "\U0001F595",  # 🖕
    "thinking_face": "\U0001F914",  # 🤔
    "joy": "\U0001F602",           # 😂
}

# Skin-tone (Fitzpatrick) modifiers. Slack names a toned reaction
# ``base::skin-tone-N``; WhatsApp needs the base emoji followed by the literal
# modifier codepoint. Keyed by the ``skin-tone-N`` suffix the model emits.
# "brown" is medium-dark (type-5, 🏾) — the tone this plugin was asked for.
_SKIN_TONE_UNICODE = {
    "skin-tone-2": "\U0001F3FB",  # 🏻 light
    "skin-tone-3": "\U0001F3FC",  # 🏼 medium-light
    "skin-tone-4": "\U0001F3FD",  # 🏽 medium
    "skin-tone-5": "\U0001F3FE",  # 🏾 medium-dark / "brown"
    "skin-tone-6": "\U0001F3FF",  # 🏿 dark
}

# Only hand/person emoji accept a skin tone. Applying a modifier to a face or
# object (e.g. 🎉, 😂) yields a broken two-glyph sequence on WhatsApp and an
# invalid name on Slack, so a tone requested on anything else is dropped and the
# base emoji reacts plain.
_SKINTONE_CAPABLE = frozenset({
    "+1",
    "thumbsup",
    "-1",
    "thumbsdown",
    "raised_hands",
    "pray",
    "ok_hand",
    "clap",
    "middle_finger",
})


def _split_skin_tone(shortcode: str) -> tuple[str, Optional[str]]:
    """Split ``base::skin-tone-N`` into ``(base, "skin-tone-N")``.

    Returns ``(shortcode, None)`` when there is no valid skin-tone suffix, or
    when the base emoji does not accept one (the tone is then discarded so the
    base reacts plain).
    """
    base, sep, tone = shortcode.partition("::")
    if sep and tone in _SKIN_TONE_UNICODE and base in _SKINTONE_CAPABLE:
        return base, tone
    return base if sep else shortcode, None


def _resolve_platform(raw: Any) -> Optional[str]:
    """Normalize the turn's platform to 'slack'/'whatsapp', or None if neither.

    ``raw`` may be a plain string or a Platform enum; match leniently on value.
    """
    if raw is None:
        return None
    val = getattr(raw, "value", raw)
    val = str(val).strip().lower()
    return val if val in _SUPPORTED_PLATFORMS else None


def _policy(platform: str) -> str:
    """The reaction policy injected into the turn, tailored to the platform."""
    where = "WhatsApp" if platform == "whatsapp" else "Slack"
    return (
        f"Emoji reaction policy (this turn is on {where}):\n"
        "- For a message that only needs a lightweight acknowledgement — e.g. "
        '"thanks", "ok", "done", "issue fixed", "lgtm", a \U0001F44D, or '
        "similar — prefer reacting with an emoji over writing a text reply.\n"
        "- To react, put a control directive `[[react:EMOJI]]` anywhere in your "
        "response, where EMOJI is an emoji shortcode WITHOUT colons "
        "(e.g. +1, tada, eyes, white_check_mark, raised_hands, pray, "
        "middle_finger). You may include more than one directive to add "
        "multiple reactions.\n"
        "- Hand/person emoji (e.g. +1, raised_hands, pray, clap, ok_hand, "
        "middle_finger) accept a skin tone: append `::skin-tone-N` where N is "
        "2 (light) … 6 (dark) and brown is 5, e.g. "
        "`[[react:middle_finger::skin-tone-5]]`.\n"
        "- If a reaction is all that's warranted, make the ENTIRE rest of your "
        f"response exactly `{_SILENT_TOKEN}` so NO text message is sent — only "
        f"the reaction lands. Example: `[[react:+1]] {_SILENT_TOKEN}`.\n"
        "- When a substantive answer is needed, just reply normally; you may "
        "still add a `[[react:...]]` directive on top if a reaction also fits.\n"
        "- Use judgment: do not react to genuine questions or requests that need "
        "a real answer, and do not over-use reactions."
    )


def _resolve_target(platform: str):
    """Resolve (runner, adapter, chat_id, message_id, participant) for this turn.

    Reads the *current* turn's chat/message ids from the session context vars
    and the live adapter from the gateway runner. Returns None when not
    reachable (CLI, no live gateway, missing ids, adapter not connected).
    """
    try:
        from gateway.config import Platform
        from gateway.run import _gateway_runner_ref
        from gateway.session_context import get_session_env
    except Exception as exc:  # pragma: no cover - import guard
        logger.debug("slack_react: gateway imports unavailable: %s", exc)
        return None

    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    message_id = get_session_env("HERMES_SESSION_MESSAGE_ID", "")
    if not chat_id or not message_id:
        logger.debug("slack_react: no chat_id/message_id in session context")
        return None

    try:
        runner = _gateway_runner_ref()
    except Exception:
        runner = None
    if runner is None:
        return None

    platform_enum = Platform.SLACK if platform == "slack" else Platform.WHATSAPP
    try:
        adapter = runner.adapters.get(platform_enum)
    except Exception:
        adapter = None
    if adapter is None:
        return None

    participant = None
    if platform == "whatsapp" and chat_id.endswith("@g.us"):
        # In a group the react key needs the sender JID of the target message;
        # for a 1:1 chat it is omitted (remoteJid is enough). Without it the
        # Baileys key is incomplete, so skip rather than misfire the reaction.
        participant = get_session_env("HERMES_SESSION_USER_ID", "") or None
        if not participant:
            logger.debug("slack_react: group reaction needs sender JID; skipping")
            return None

    return runner, adapter, chat_id, message_id, participant


def _dispatch(runner: Any, make_coro: Callable[[], Any], timeout: float = 15.0) -> Any:
    """Run an adapter coroutine and return its result (or False on failure).

    The hook fires in a gateway worker thread; the WhatsApp adapter's aiohttp
    session is bound to the gateway loop, so schedule the coroutine there with
    ``safe_schedule_threadsafe`` when that loop is actually running. Otherwise
    fall back to a private loop (CLI/tests). ``make_coro`` is a factory so each
    path awaits a fresh coroutine (never one already closed/consumed).
    ``timeout`` bounds how long the caller blocks on this single reaction.

    Returns the coroutine's own return value on success, or ``False`` when the
    call could not be scheduled, timed out, or raised. Callers interpret the
    adapter's success signal via ``_reaction_ok``.
    """
    loop = getattr(runner, "_gateway_loop", None)
    if loop is not None and not loop.is_closed() and loop.is_running():
        try:
            from agent.async_utils import safe_schedule_threadsafe
        except Exception as exc:  # pragma: no cover - import guard
            logger.debug("slack_react: async_utils unavailable: %s", exc)
            return False
        future = safe_schedule_threadsafe(
            make_coro(),
            loop,
            logger=logger,
            log_message="slack_react: reaction failed to schedule",
        )
        if future is None:
            return False
        try:
            return future.result(timeout=max(0.0, timeout))
        except Exception as exc:
            # Cancel so a slow call can't land a late reaction after we give up.
            future.cancel()
            logger.debug("slack_react: reaction call failed: %s", exc)
            return False

    # Fallback: no live (running) gateway loop (CLI / tests). Honor the same
    # timeout here via wait_for, since _run_async's own deadline is far longer.
    try:
        from model_tools import _run_async

        return _run_async(asyncio.wait_for(make_coro(), max(0.01, timeout)))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("slack_react: fallback dispatch failed: %s", exc)
        return False


def _reaction_ok(result: Any) -> bool:
    """Interpret an adapter reaction result as success.

    Slack ``_add_reaction`` returns a bool; WhatsApp ``send_reaction`` returns a
    ``SendResult`` with ``.success``; a missing return value is treated as
    success (the call ran without error).
    """
    if result is None:
        return True
    success = getattr(result, "success", None)
    if success is not None:
        return bool(success)
    return bool(result)


def _add_reactions(platform: str, shortcodes: List[str]) -> int:
    """Add each reaction to the current turn's message. Returns the count fired."""
    target = _resolve_target(platform)
    if target is None:
        return 0
    runner, adapter, chat_id, message_id, participant = target

    count = 0
    seen = set()
    deadline = time.monotonic() + _REACTION_TOTAL_BUDGET_S
    for raw in shortcodes:
        # Cap on reactions actually fired (not directives seen), so skipped
        # ones — duplicates, or shortcodes with no WhatsApp unicode mapping —
        # never consume the budget and block a later valid reaction.
        if count >= _MAX_REACTIONS_PER_RESPONSE:
            logger.debug(
                "slack_react: reaction limit (%d) reached; skipping the rest",
                _MAX_REACTIONS_PER_RESPONSE,
            )
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.debug("slack_react: reaction time budget exhausted; skipping the rest")
            break
        name = raw.strip().strip(":").strip()
        key = name.lower()
        if not name or key in seen:
            continue
        seen.add(key)
        # Slack emoji names are lowercase — normalize so "TADA" == "tada". Split
        # off an optional skin tone: `middle_finger::skin-tone-5` → base +
        # `skin-tone-5` (dropped for emoji that can't carry one).
        base, tone = _split_skin_tone(key)

        if platform == "slack":
            add_reaction = getattr(adapter, "_add_reaction", None)
            if not callable(add_reaction):
                logger.debug("slack_react: slack adapter has no _add_reaction")
                break
            # Slack's native toned-reaction name is `base::skin-tone-N`.
            slack_name = f"{base}::{tone}" if tone else base
            if _reaction_ok(
                _dispatch(
                    runner,
                    lambda n=slack_name: add_reaction(chat_id, message_id, n),
                    timeout=remaining,
                )
            ):
                count += 1
        else:  # whatsapp
            emoji = _SHORTCODE_TO_UNICODE.get(base)
            if not emoji:
                logger.debug("slack_react: no unicode mapping for %r (whatsapp)", name)
                continue
            # WhatsApp takes a literal emoji: base + Fitzpatrick modifier.
            if tone:
                emoji += _SKIN_TONE_UNICODE[tone]
            send_reaction = getattr(adapter, "send_reaction", None)
            if not callable(send_reaction):
                logger.debug("slack_react: whatsapp adapter has no send_reaction")
                break
            if _reaction_ok(
                _dispatch(
                    runner,
                    lambda e=emoji: send_reaction(
                        chat_id, message_id, e, from_me=False, participant=participant
                    ),
                    timeout=remaining,
                )
            ):
                count += 1

    if count:
        # Aggregate only — no chat/message identifiers at info level.
        logger.info("slack_react: added %d reaction(s) on %s", count, platform)
    return count


# ----------------------------------------------------------------------------
# Hooks
# ----------------------------------------------------------------------------

def _pre_llm_call(**kwargs) -> Optional[dict]:
    """Inject the reaction policy into the turn context (Slack/WhatsApp only)."""
    platform = _resolve_platform(kwargs.get("platform"))
    if not platform:
        return None
    return {"context": _policy(platform)}


def _transform_llm_output(**kwargs) -> Optional[str]:
    """Act on any [[react:...]] directives in the model's reply.

    Returns the cleaned reply text (react + reply), the NO_REPLY silence token
    (react only), or None to leave the response unchanged.
    """
    platform = _resolve_platform(kwargs.get("platform"))
    if not platform:
        return None

    text = kwargs.get("response_text") or ""
    if not isinstance(text, str) or "[[" not in text:
        return None

    shortcodes = _DIRECTIVE_RE.findall(text)
    if not shortcodes:
        return None

    # Fire the reactions (best-effort). Even if delivery fails we still strip
    # the directive so raw markup never reaches the user.
    try:
        _add_reactions(platform, shortcodes)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("slack_react: _add_reactions raised: %s", exc)

    cleaned = _DIRECTIVE_RE.sub("", text).strip()
    if not cleaned or cleaned.upper() == _SILENT_TOKEN:
        # Reaction is the whole response → stay silent.
        return _SILENT_TOKEN
    return cleaned


def register(ctx) -> None:
    """Register the two hooks. Called once when the plugin is enabled."""
    ctx.register_hook("pre_llm_call", _pre_llm_call)
    ctx.register_hook("transform_llm_output", _transform_llm_output)
    logger.info(
        "slack_react plugin registered (pre_llm_call + transform_llm_output; "
        "slack + whatsapp)"
    )
