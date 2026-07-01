"""Session-reset / iteration-limit copy must be warm and metadata-free.

Regression coverage for the quiet-chat-gateway-management skill: daily/idle
resets, context-overflow resets, and iteration-limit fallbacks are human-facing
messaging output and must never leak the model, provider, context-token count,
config paths, or robotic banner glyphs.
"""

import pytest

from agent.chat_completion_helpers import ITERATION_LIMIT_FALLBACK_MESSAGE
from gateway.run import (
    _CONTEXT_OVERFLOW_RESET_NOTICE,
    _auto_reset_chat_notice,
)

# Tokens that mark robotic scaffolding / leaked implementation metadata. None of
# these may appear in any user-facing reset or limit copy on a chat surface.
FORBIDDEN_SUBSTRINGS = [
    "◐",
    "◆",
    "🔄",
    "config.yaml",
    "session_reset",
    "/resume",
    "model:",
    "provider:",
    "context:",
    "token",
    "max_iterations",
    "iteration limit",
    "maximum iterations",
    "automatically reset",
    "history cleared",
    "context size",
    "compress",
    "error:",
]


def _assert_clean(text: str) -> None:
    assert text and text.strip(), "copy must be a non-empty line"
    low = text.lower()
    for bad in FORBIDDEN_SUBSTRINGS:
        assert bad not in low, f"leaked robotic/technical token: {bad!r} in {text!r}"


@pytest.mark.parametrize("reason", ["daily", "idle", "suspended", "inactivity", None, "weird"])
def test_auto_reset_notice_is_warm_and_metadata_free(reason):
    _assert_clean(_auto_reset_chat_notice(reason))


def test_auto_reset_notice_varies_by_reason():
    """Daily / suspended / idle each get their own peer-style line."""
    daily = _auto_reset_chat_notice("daily")
    suspended = _auto_reset_chat_notice("suspended")
    idle = _auto_reset_chat_notice("idle")
    assert len({daily, suspended, idle}) == 3


def test_context_overflow_notice_is_warm_and_metadata_free():
    _assert_clean(_CONTEXT_OVERFLOW_RESET_NOTICE)


def test_iteration_limit_fallback_is_natural():
    _assert_clean(ITERATION_LIMIT_FALLBACK_MESSAGE)
    # It should read as a peer apology that invites a retry, not a diagnostic.
    low = ITERATION_LIMIT_FALLBACK_MESSAGE.lower()
    assert "try again" in low
    assert "couldn't generate a summary" not in low
