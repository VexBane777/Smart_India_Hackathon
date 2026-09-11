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
    DEFAULT_EVAL_CHANNELS,
    PHONE_CHANNELS,
    TRAIN_CHANNELS,
    UNSEEN_CHANNELS,
    CleanOnlyEvalError,
    parse_channel_arg,
    resolve_channels,
)

HERE = Path(__file__).resolve().parent


def test_default_eval_channels_are_none_plus_every_phone_channel():
    assert resolve_channels(None, "phone", "eval") == [None, *PHONE_CHANNELS]
    assert DEFAULT_EVAL_CHANNELS[0] is None


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


def test_every_eval_entry_point_resolves_channels_through_the_policy():
    for name in ("evaluate.py", "select_best_checkpoint_seqcnn.py", "train_seq_cnn.py", "build_caches.py",
                 "validate_fp16.py"):
        src = (HERE / name).read_text(encoding="utf-8")
        assert "resolve_channels(" in src, f"{name} does not call eval_protocol.resolve_channels"
