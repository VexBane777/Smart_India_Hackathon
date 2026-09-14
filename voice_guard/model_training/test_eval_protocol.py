"""The channel policy (eval_protocol.py): behavior, plus a static scan that
fails if any script bypasses it with a literal clean-only channel list, or
if checkpoint selection ever reads the test split.

Policy (user, 2026-09-11): no VoiceGuard eval or training runs purely on
clean audio unless the application is retraining for banks."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from eval_protocol import (
    ACOUSTIC_CHANNELS,
    ALL_RECIPES,
    DEFAULT_EVAL_CHANNELS,
    PHONE_CHANNELS,
    TRAIN_CHANNELS,
    UNSEEN_CHANNELS,
    CleanOnlyEvalError,
    acoustic_subset,
    parse_channel_arg,
    resolve_channels,
)

HERE = Path(__file__).resolve().parent


def test_default_eval_channels_are_none_plus_every_phone_and_acoustic_channel():
    assert resolve_channels(None, "phone", "eval") == [None, *ALL_RECIPES]
    assert DEFAULT_EVAL_CHANNELS[0] is None
    assert set(ACOUSTIC_CHANNELS) == {"playback"}
    assert set(ALL_RECIPES) == set(PHONE_CHANNELS) | {"playback"}


def test_playback_is_a_valid_acoustic_channel():
    assert resolve_channels(["none", "whatsapp", "playback"], "phone", "eval") == [None, "whatsapp", "playback"]
    assert acoustic_subset(["none", "playback", "whatsapp"]) == ["playback"]
    # acoustic alone is a real capture path but not a phone path: still refused
    # for the phone application (policy: eval/training needs a phone channel).
    with pytest.raises(CleanOnlyEvalError):
        resolve_channels(["playback"], "phone", "eval")


def test_default_train_channels():
    assert resolve_channels(None, "phone", "train") == list(TRAIN_CHANNELS)
    assert set(UNSEEN_CHANNELS) == {"gsm_2g", "pstn", "tandem_xnet"}
    assert not set(UNSEEN_CHANNELS) & set(TRAIN_CHANNELS)


@pytest.mark.parametrize("channels", [["none"], [None], ["NONE", None]])
@pytest.mark.parametrize("purpose", ["eval", "train"])
def test_clean_only_is_refused_for_phone(channels, purpose):
    with pytest.raises(CleanOnlyEvalError):
        resolve_channels(channels, "phone", purpose)


def test_clean_only_is_allowed_for_bank():
    assert resolve_channels(["none"], "bank", "eval") == [None]


def test_clean_recipe_is_refused_even_with_phone_channels():
    with pytest.raises(ValueError, match="debug recipe"):
        resolve_channels(["clean", "whatsapp"], "phone", "eval")


def test_unknown_channel_and_application_refused():
    with pytest.raises(ValueError):
        resolve_channels(["landline"], "phone", "eval")
    with pytest.raises(ValueError):
        resolve_channels(["whatsapp"], "retail", "eval")


def test_duplicates_removed_order_kept():
    assert resolve_channels(["pstn", "none", "pstn"], "phone", "eval") == ["pstn", None]


def test_parse_channel_arg_maps_none_string_to_real_none():
    assert parse_channel_arg(["whatsapp", "None", "NONE", "none"]) == ["whatsapp", None, None, None]
    assert parse_channel_arg(["clean"]) == ["clean"]  # passed through; resolve_channels rejects it


# --- static policy scan ------------------------------------------------------

def _is_clean_only_literal(node: ast.AST) -> bool:
    if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
        return False
    return all(isinstance(e, ast.Constant) and (e.value is None or (isinstance(e.value, str) and e.value.lower() == "none"))
               for e in node.elts)


def _scripts() -> list[Path]:
    return sorted(p for p in HERE.glob("*.py") if not p.name.startswith("test_") and p.name != "conftest.py")


def test_no_script_passes_a_literal_clean_only_channel_list():
    """Catches e.g. build_examples(..., channel_recipes=[None]) or
    add_argument("--channel", default=[None]): the exact pattern every
    v11-era eval script used. Tests are exempt (they exercise mechanics on
    synthetic audio, not model quality)."""
    violations = []
    for path in _scripts():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    if _is_clean_only_literal(arg):
                        violations.append(f"{path.name}:{node.lineno}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                    if _is_clean_only_literal(d):
                        violations.append(f"{path.name}:{node.lineno} (default arg)")
    assert not violations, (
        "clean-only channel literal(s) found; route channels through eval_protocol.resolve_channels "
        f"(policy: no clean-only eval/training unless --application bank): {violations}")


def test_scan_detects_a_planted_violation(tmp_path):
    bad = ast.parse("build_examples(a, b, channel_recipes=[None])\nf(default=('none',))")
    hits = [n for n in ast.walk(bad) if isinstance(n, ast.Call)
            and any(_is_clean_only_literal(a) for a in list(n.args) + [k.value for k in n.keywords])]
    assert len(hits) == 2


def test_checkpoint_selection_never_reads_the_test_split():
    path = HERE / "select_best_checkpoint_seqcnn.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "test" not in literals, "select_best_checkpoint_seqcnn.py references the 'test' split"
    import select_best_checkpoint_seqcnn

    assert select_best_checkpoint_seqcnn.SPLIT == "select"


# Scripts that parse args AND load a model (an eval/training entry point) but
# intentionally do not route channels through resolve_channels. Kept as a
# deliberate, reviewed allowlist so the discovery below does not silently
# exempt a new eval script; each entry needs a one-line, still-true reason.
_MODEL_ENTRY_ALLOWLIST = {
    "eval_playback_loop.py": (
        "scores the shipped ONNX clean and looped per asset; has no channel-"
        "policy dimension (one model, no split, no recipe mix) and is a "
        "supplementary on-device gate, not an eval."),
}


def _eval_entry_points() -> list[Path]:
    """Every non-test script that parses args and loads a model: the scripts
    that must never bypass the channel policy. The v12 code hard-coded five
    names, so a new eval script slipped past the check silently (the hole
    eval_playback_loop exposed). Discovery, not a fixed list."""
    entry_points = []
    for p in _scripts():
        src = p.read_text(encoding="utf-8")
        loads = any(k in src for k in ("onnxruntime", "InferenceSession", "session.run",
                                       "torch.load", "load_scoring_model", "build_model_from_norm_stats"))
        parses_args = "ArgumentParser" in src or "argparse" in src
        if loads and parses_args:
            entry_points.append(p)
    return entry_points


def test_every_eval_entry_point_resolves_channels_through_the_policy():
    """Every script that parses args and loads a model must either call
    eval_protocol.resolve_channels or be explicitly allowlisted with a reason
    (see _MODEL_ENTRY_ALLOWLIST). Catches a new eval script bypassing the
    no-clean-only policy without review."""
    offenders = []
    for p in _eval_entry_points():
        if "resolve_channels(" in p.read_text(encoding="utf-8"):
            continue
        if p.name in _MODEL_ENTRY_ALLOWLIST:
            continue
        offenders.append(p.name)
    assert not offenders, (
        "eval entry point(s) do not route channels through "
        f"resolve_channels: {offenders}. Add resolve_channels, or add an "
        "explicit _MODEL_ENTRY_ALLOWLIST entry with a one-line reason.")


def test_eval_playback_loop_is_intentionally_allowlisted():
    import eval_playback_loop as m
    assert "eval_playback_loop.py" in _MODEL_ENTRY_ALLOWLIST
    assert m is not None
