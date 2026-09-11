"""
VoiceGuard evaluation/training channel policy: the single source of truth.

**Policy (user directive, 2026-09-11): no eval runs purely on clean audio,
unless the stated application is retraining the model for banks' use.**
The same rule applies to training. VoiceGuard is a phone-call product. v11_seqcnn
was trained `--channel none` and every one of its evals ran on clean audio,
so its headline EER said nothing about real calls. This module makes that
class of blind spot impossible to reintroduce silently:

- every script resolves its channel list through `resolve_channels`, which
  raises `CleanOnlyEvalError` when the resolved set contains no phone
  channel and the application isn't "bank";
- `test_eval_protocol.py` AST-scans every non-test script for a literal
  clean-only channel list (e.g. `channel_recipes=[None]`, or
  `add_argument("--channel", default=[None])`), so a bypass fails CI rather
  than slipping through review.

Channel names are TeleChannel recipe names
(`vaani/telechannel/configs/channels.yaml`). `None` (spelled "none" on the
CLI) means "no degradation at all". TeleChannel's `clean` recipe is **not**
that: it applies RIR, noise, clipping and packet loss, and skips only the
codec. It is a debug recipe and is never allowed here.

See `voice_guard/docs/EVAL-PROTOCOL.md` for the full protocol.
"""
from __future__ import annotations

from typing import Iterable, Sequence

# Every recipe that models a real phone path. `clean` deliberately excluded.
PHONE_CHANNELS: tuple[str, ...] = ("whatsapp", "volte", "cellular_3g", "gsm_2g", "pstn", "tandem_xnet")

# Training sees `none` plus these three. The other three phone channels are
# never trained on, so they form the "unseen channel" eval group.
TRAIN_CHANNELS: tuple[str | None, ...] = (None, "whatsapp", "volte", "cellular_3g")
TRAIN_PHONE_CHANNELS: tuple[str, ...] = tuple(c for c in TRAIN_CHANNELS if c is not None)
UNSEEN_CHANNELS: tuple[str, ...] = tuple(c for c in PHONE_CHANNELS if c not in TRAIN_CHANNELS)

# `none` is a reference row in every report, never an eval on its own.
DEFAULT_EVAL_CHANNELS: tuple[str | None, ...] = (None,) + PHONE_CHANNELS

APPLICATIONS: tuple[str, ...] = ("phone", "bank")
FORBIDDEN_RECIPES: frozenset[str] = frozenset({"clean"})


class CleanOnlyEvalError(ValueError):
    """Raised when a train/eval run would see no phone-channel audio."""


def channel_name(recipe: str | None) -> str:
    """Display/storage name for a recipe: None -> "none"."""
    return "none" if recipe is None else recipe


def parse_channel_arg(items: Iterable[str | None]) -> list[str | None]:
    """CLI strings -> recipes. The literal "none" (any case) maps to None.
    Every other name passes through unchanged. `clean` is not rewritten
    here; `resolve_channels` rejects it."""
    return [None if r is None or str(r).lower() == "none" else str(r) for r in items]


def resolve_channels(
    channels: Sequence[str | None] | None,
    application: str = "phone",
    purpose: str = "eval",
) -> list[str | None]:
    """Validates and returns the channel list for a run.

    channels: recipes (None = undegraded), or None for the default for
        `purpose` (DEFAULT_EVAL_CHANNELS for eval, TRAIN_CHANNELS for train).
    application: "phone" (default) or "bank". Only "bank" may run clean-only.
    purpose: "eval" or "train", used only for the default and error text.

    Raises CleanOnlyEvalError if no phone channel is present and application
    != "bank". Raises ValueError on an unknown application, an empty list,
    the `clean` debug recipe, or a name that isn't a phone channel."""
    if application not in APPLICATIONS:
        raise ValueError(f"application must be one of {APPLICATIONS}, got {application!r}")
    if purpose not in ("eval", "train"):
        raise ValueError(f"purpose must be 'eval' or 'train', got {purpose!r}")
    if channels is None:
        channels = DEFAULT_EVAL_CHANNELS if purpose == "eval" else TRAIN_CHANNELS
    resolved = parse_channel_arg(channels)
    if not resolved:
        raise ValueError(f"empty channel list for {purpose}")

    seen: list[str | None] = []
    for c in resolved:
        if c in FORBIDDEN_RECIPES:
            raise ValueError(
                f"channel {c!r} is TeleChannel's debug recipe (RIR+noise+clipping+loss, "
                "no codec). It is neither undegraded audio nor a phone path. Use 'none' "
                f"for undegraded audio or one of {PHONE_CHANNELS}."
            )
        if c is not None and c not in PHONE_CHANNELS:
            raise ValueError(f"unknown channel {c!r}; expected 'none' or one of {PHONE_CHANNELS}")
        if c not in seen:
            seen.append(c)

    if application != "bank" and not any(c in PHONE_CHANNELS for c in seen):
        raise CleanOnlyEvalError(
            f"{purpose} channel set {[channel_name(c) for c in seen]} has no phone channel. "
            "Policy (2026-09-11): no VoiceGuard eval or training runs purely on clean audio "
            "unless the application is retraining for banks (--application bank). "
            f"Add at least one of {PHONE_CHANNELS}. See voice_guard/docs/EVAL-PROTOCOL.md."
        )
    return seen


def phone_subset(channels: Iterable[str | None]) -> list[str]:
    """The phone channels in `channels`, in order."""
    return [c for c in channels if c in PHONE_CHANNELS]
