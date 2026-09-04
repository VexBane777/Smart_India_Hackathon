"""
Unit tests for telechannel/manifest.py.

Two concerns are covered:

1. The Parquet logger (write_manifest_row / read_manifest / validation).
2. Leakage-safe split generation (generate_splits / assert_no_leakage /
   the hold-out filters), which is the correctness-critical part of this
   module per master plan section 5.4 / DOC1 1.6 hard rules 2-3: splitting
   must happen at the source_clip level, and additionally at the speaker
   level for real speech.

For (2), `_synthetic_manifest()` below builds a manifest where many ROWS
look independent (different index, different recipe/codec/etc.) but
deliberately share `source_clip` or `speaker_anon` values across what
would otherwise look like unrelated rows -- e.g. the same source_clip
processed through multiple channel recipes (one row per recipe, all
sharing one source_clip), and the same speaker_anon appearing under
several different source_clips. A splitter that only looks at row identity
(or naively hashes the row / clip_id) would scatter these across
train/val/test; generate_splits must not.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from telechannel import manifest as manifest_mod
from telechannel.manifest import (
    ManifestValidationError,
    SPLIT_NAMES,
    assert_no_leakage,
    filter_holdout_real_calls,
    filter_holdout_unseen_codec,
    filter_holdout_unseen_generator,
    filter_holdout_unseen_noise,
    generate_splits,
    get_holdout_sets,
    holdout_mask,
    read_manifest,
    validate_metadata,
    write_manifest_row,
)


def _row(
    source_clip,
    speaker_anon=None,
    generator="melo",
    accent="hi-en",
    codec_chain=None,
    bitrate="7.4k",
    loss_pct=1.0,
    snr_db=15.0,
    rir_id="slr28_sim_001",
    license="CC0",
    consent="c_260312_KP",
    noise="musan_babble",
    **extra,
):
    row = dict(
        source_clip=source_clip,
        speaker_anon=speaker_anon,
        generator=generator,
        accent=accent,
        codec_chain=codec_chain if codec_chain is not None else ["libopencore_amrnb"],
        bitrate=bitrate,
        loss_pct=loss_pct,
        snr_db=snr_db,
        rir_id=rir_id,
        license=license,
        consent=consent,
        noise=noise,
    )
    row.update(extra)
    return row


def _synthetic_manifest():
    """
    Dozens of rows, deliberately NOT one-row-per-independent-clip:

    - source_clip "sc_001".."sc_005" each appear under THREE different
      channel recipes/codecs (as real corpus generation would produce:
      one source recording run through several recipes), i.e. 15 rows
      sharing just 5 distinct source_clip values.
    - Real-speech rows (speaker_anon set) for speaker "spk_001" are spread
      across THREE different source_clips (sc_101, sc_102, sc_103) -- e.g.
      the same donor recorded multiple utterances -- each again run
      through multiple recipes, so "spk_001" backs 9 rows across 3 clips.
      Likewise "spk_002" backs clips sc_201/sc_202.
    - A block of purely synthetic rows (speaker_anon=None, generator=
      "bark"/"rvc_team") each with a unique source_clip, to exercise the
      simple non-leakage-prone case alongside the deliberately-shared one.
    - One row is a held-out-generator row (generator="rvc_team"), one is a
      held-out-codec row (codec_chain contains "amr_wb"), one is a
      held-out-noise row (noise="freesound_horn"), and one is a real-call
      row (origin="real_call"), to exercise the Step 3 filters.
    """
    rows = []
    recipes = ["pstn", "gsm_2g", "cellular_3g"]

    # Synthetic clips reused across recipes (source-clip-level leakage risk).
    for i in range(1, 6):
        clip = f"sc_{i:03d}"
        for recipe in recipes:
            rows.append(
                _row(
                    source_clip=clip,
                    speaker_anon=None,
                    generator="bark",
                    recipe=recipe,
                )
            )

    # Real-speech clips: two speakers, each behind multiple source_clips,
    # each clip run through multiple recipes (speaker-level leakage risk).
    speaker_clip_map = {
        "spk_001": ["sc_101", "sc_102", "sc_103"],
        "spk_002": ["sc_201", "sc_202"],
    }
    for speaker, clips in speaker_clip_map.items():
        for clip in clips:
            for recipe in recipes:
                rows.append(
                    _row(
                        source_clip=clip,
                        speaker_anon=speaker,
                        generator="real",
                        recipe=recipe,
                    )
                )

    # Unique, unrelated synthetic clips (no leakage risk, happy path).
    for i in range(1, 11):
        rows.append(
            _row(
                source_clip=f"sc_unique_{i:03d}",
                speaker_anon=None,
                generator="indic_tts" if i % 2 else "melo",
                recipe="volte",
            )
        )

    # Hold-out exemplars.
    rows.append(_row(source_clip="sc_holdout_gen", generator="rvc_team"))
    rows.append(
        _row(
            source_clip="sc_holdout_codec",
            codec_chain=["libopencore_amrnb", "libvo_amrwbenc_amr_wb"],
        )
    )
    # Use the exact holdout token "amr_wb" as one chain entry for a clean match.
    rows[-1]["codec_chain"] = ["amr_wb"]
    rows.append(_row(source_clip="sc_holdout_noise", noise="freesound_horn"))
    rows.append(_row(source_clip="sc_holdout_real", origin="real_call", speaker_anon="spk_call_1"))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 1: Parquet logger
# ---------------------------------------------------------------------------


def test_validate_metadata_accepts_complete_row():
    validate_metadata(_row(source_clip="sc_001"))  # should not raise


def test_validate_metadata_rejects_missing_field():
    row = _row(source_clip="sc_001")
    del row["license"]
    with pytest.raises(ManifestValidationError):
        validate_metadata(row)


def test_validate_metadata_rejects_blank_license():
    row = _row(source_clip="sc_001", license="")
    with pytest.raises(ManifestValidationError):
        validate_metadata(row)


def test_validate_metadata_allows_null_speaker_anon():
    row = _row(source_clip="sc_001", speaker_anon=None)
    validate_metadata(row)  # should not raise: speaker_anon is nullable


def test_write_manifest_row_creates_and_appends(tmp_path):
    manifest_path = tmp_path / "manifest.parquet"

    write_manifest_row(_row(source_clip="sc_001"), manifest_path=manifest_path)
    df = write_manifest_row(_row(source_clip="sc_002"), manifest_path=manifest_path)

    assert manifest_path.exists()
    assert len(df) == 2
    assert set(df["source_clip"]) == {"sc_001", "sc_002"}

    reloaded = read_manifest(manifest_path)
    assert len(reloaded) == 2


def test_write_manifest_row_raises_on_invalid_metadata(tmp_path):
    manifest_path = tmp_path / "manifest.parquet"
    bad_row = _row(source_clip="sc_001")
    del bad_row["consent"]

    with pytest.raises(ManifestValidationError):
        write_manifest_row(bad_row, manifest_path=manifest_path)

    # Nothing should have been written.
    assert not manifest_path.exists()


# ---------------------------------------------------------------------------
# Step 3: hold-out filters
# ---------------------------------------------------------------------------


def test_filter_holdout_unseen_generator():
    df = _synthetic_manifest()
    result = filter_holdout_unseen_generator(df)
    assert set(result["source_clip"]) == {"sc_holdout_gen"}


def test_filter_holdout_unseen_codec():
    df = _synthetic_manifest()
    result = filter_holdout_unseen_codec(df)
    assert "sc_holdout_codec" in set(result["source_clip"])


def test_filter_holdout_unseen_noise():
    df = _synthetic_manifest()
    result = filter_holdout_unseen_noise(df)
    assert set(result["source_clip"]) == {"sc_holdout_noise"}


def test_filter_holdout_real_calls():
    df = _synthetic_manifest()
    result = filter_holdout_real_calls(df)
    assert set(result["source_clip"]) == {"sc_holdout_real"}


def test_filter_holdout_real_calls_missing_column_returns_empty():
    df = _synthetic_manifest().drop(columns=["origin"])
    result = filter_holdout_real_calls(df)
    assert len(result) == 0


def test_get_holdout_sets_returns_all_four():
    df = _synthetic_manifest()
    sets = get_holdout_sets(df)
    assert set(sets.keys()) == {
        "unseen_generator",
        "unseen_codec",
        "unseen_noise",
        "real_calls",
    }
    for name, subset in sets.items():
        assert len(subset) >= 1, f"expected at least one row in {name!r} holdout"


# ---------------------------------------------------------------------------
# Step 2 / Step 4: leakage-safe splits -- the correctness-critical tests.
# ---------------------------------------------------------------------------


def test_generate_splits_assigns_every_row_a_known_split():
    df = _synthetic_manifest()
    result = generate_splits(df, seed=0)

    assert "split" in result.columns
    assert result["split"].isin(SPLIT_NAMES).all()
    assert len(result) == len(df)


def test_generate_splits_no_source_clip_leakage():
    """The headline requirement: no source_clip appears in more than one
    split, even though many source_clips here have multiple rows (one per
    recipe) that would leak if splitting were done per-row."""
    df = _synthetic_manifest()
    result = generate_splits(df, seed=0)

    clip_split_counts = result.groupby("source_clip")["split"].nunique()
    leaking = clip_split_counts[clip_split_counts > 1]
    assert leaking.empty, f"source_clip(s) leaked across splits: {leaking.index.tolist()}"


def test_generate_splits_no_speaker_leakage_for_real_speech():
    """spk_001 backs 3 different source_clips (9 rows); spk_002 backs 2
    (6 rows). None of those clips/rows may be split across train/test/val,
    even though they look like independent rows."""
    df = _synthetic_manifest()
    result = generate_splits(df, seed=0)

    real_rows = result[result["speaker_anon"].notna()]
    speaker_split_counts = real_rows.groupby("speaker_anon")["split"].nunique()
    leaking = speaker_split_counts[speaker_split_counts > 1]
    assert leaking.empty, f"speaker_anon(s) leaked across splits: {leaking.index.tolist()}"


def test_generate_splits_naive_row_level_shuffle_would_have_leaked():
    """Sanity check that the synthetic fixture actually exercises the
    leakage rule: a naive per-ROW random split (ignoring source_clip
    grouping entirely) demonstrably DOES leak on this fixture, so the
    grouped assert above is testing something real and not vacuously
    passing because the fixture has no shared clips/speakers."""
    df = _synthetic_manifest()
    rng = np.random.default_rng(0)
    naive_split = rng.choice(list(SPLIT_NAMES), size=len(df))
    df_naive = df.copy()
    df_naive["split"] = naive_split

    clip_split_counts = df_naive.groupby("source_clip")["split"].nunique()
    assert (clip_split_counts > 1).any(), (
        "expected the naive per-row shuffle to leak at least one source_clip "
        "across splits on this fixture"
    )


@pytest.mark.parametrize("seed", [0, 1, 7, 123])
def test_generate_splits_no_leakage_across_multiple_seeds(seed):
    """Leakage-safety must hold regardless of RNG seed, not just for one
    lucky shuffle order."""
    df = _synthetic_manifest()
    result = generate_splits(df, seed=seed)
    assert_no_leakage(result)  # should not raise


def test_generate_splits_is_reproducible_given_same_seed():
    df = _synthetic_manifest()
    result1 = generate_splits(df, seed=42)
    result2 = generate_splits(df, seed=42)
    pd.testing.assert_series_equal(
        result1.sort_values("source_clip")["split"].reset_index(drop=True),
        result2.sort_values("source_clip")["split"].reset_index(drop=True),
    )


def test_generate_splits_uses_all_split_names_on_a_large_enough_fixture():
    df = _synthetic_manifest()
    result = generate_splits(df, val_frac=0.2, test_frac=0.2, demo_frac=0.1, seed=3)
    assert set(result["split"].unique()) <= set(SPLIT_NAMES)
    # With non-trivial val/test/demo fractions and dozens of rows spread
    # across many groups, expect more than just "train" to show up.
    assert len(set(result["split"].unique())) > 1


def test_generate_splits_rejects_invalid_fractions():
    df = _synthetic_manifest()
    with pytest.raises(ValueError):
        generate_splits(df, val_frac=0.5, test_frac=0.5, demo_frac=0.1)


def test_generate_splits_requires_source_clip_and_speaker_anon_columns():
    df = _synthetic_manifest().drop(columns=["speaker_anon"])
    with pytest.raises(ValueError):
        generate_splits(df)


def test_assert_no_leakage_detects_injected_source_clip_leakage():
    """Negative control: assert_no_leakage must actually catch leakage when
    it's deliberately injected, not just pass on already-correct input."""
    df = _synthetic_manifest()
    result = generate_splits(df, seed=0)

    # Pick a source_clip with multiple rows and force half of them into a
    # different split than the rest.
    clip = "sc_001"
    clip_rows = result.index[result["source_clip"] == clip].tolist()
    assert len(clip_rows) > 1
    other_split = next(s for s in SPLIT_NAMES if s != result.loc[clip_rows[0], "split"])
    result.loc[clip_rows[0], "split"] = other_split

    with pytest.raises(AssertionError):
        assert_no_leakage(result)


def test_assert_no_leakage_detects_injected_speaker_leakage():
    df = _synthetic_manifest()
    result = generate_splits(df, seed=0)

    speaker_rows = result.index[result["speaker_anon"] == "spk_001"].tolist()
    assert len(speaker_rows) > 1
    other_split = next(s for s in SPLIT_NAMES if s != result.loc[speaker_rows[0], "split"])
    result.loc[speaker_rows[0], "split"] = other_split

    with pytest.raises(AssertionError):
        assert_no_leakage(result)


# ---------------------------------------------------------------------------
# Hold-out exclusion from train: a hold-out-matching group must never be
# assigned to `train`, even by chance -- otherwise the four generalization
# hold-out sets (master plan section 5.4) are silently contaminated.
# ---------------------------------------------------------------------------


def test_holdout_mask_flags_all_four_exemplar_rows():
    df = _synthetic_manifest()
    mask = holdout_mask(df)
    flagged_clips = set(df.loc[mask, "source_clip"])
    assert flagged_clips == {
        "sc_holdout_gen",
        "sc_holdout_codec",
        "sc_holdout_noise",
        "sc_holdout_real",
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7, 42, 123])
def test_generate_splits_never_puts_holdout_rows_in_train(seed):
    """Headline fix: across many seeds, no hold-out-matching row is ever
    assigned to `train`, even though a small single-row group would
    otherwise be assigned by pure chance based on the shuffle order."""
    df = _synthetic_manifest()
    result = generate_splits(df, seed=seed)

    mask = holdout_mask(result)
    holdout_rows = result[mask]
    assert not holdout_rows.empty  # fixture sanity check

    leaked_into_train = holdout_rows[holdout_rows["split"] == "train"]
    assert leaked_into_train.empty, (
        f"hold-out-matching row(s) landed in train at seed={seed}: "
        f"{leaked_into_train['source_clip'].tolist()}"
    )


def test_generate_splits_holdout_exclusion_disableable_and_would_otherwise_leak():
    """Control proving the fixture/test above is not vacuous: with
    `exclude_holdouts_from_train=False`, sweeping the same seeds used by
    the leakage-safety check above, the raw group shuffle DOES place some
    hold-out-matching row in train on at least one seed -- i.e. the
    exclusion logic in generate_splits is doing real work, not defending
    against an impossible case."""
    df = _synthetic_manifest()
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 42, 123]

    leaked_on_any_seed = False
    for seed in seeds:
        result = generate_splits(df, seed=seed, exclude_holdouts_from_train=False)
        mask = holdout_mask(result)
        holdout_rows = result[mask]
        if (holdout_rows["split"] == "train").any():
            leaked_on_any_seed = True
            break

    assert leaked_on_any_seed, (
        "expected exclude_holdouts_from_train=False to leak a hold-out row "
        "into train on at least one seed in the sweep -- if this fails, "
        "the fixture no longer exercises the case the fix guards against"
    )


def test_generate_splits_holdout_route_is_configurable():
    df = _synthetic_manifest()
    result = generate_splits(df, seed=0, holdout_route="demo")

    mask = holdout_mask(result)
    holdout_rows = result[mask]
    assert (holdout_rows["split"] != "train").all()
    # Whichever holdout rows would have been "train" are now "demo"
    # instead of the default "test" reroute target.
    without_exclusion = generate_splits(df, seed=0, exclude_holdouts_from_train=False)
    would_have_been_train = without_exclusion.loc[
        without_exclusion["split"] == "train", "source_clip"
    ]
    rerouted = result[result["source_clip"].isin(would_have_been_train) & mask]
    if not rerouted.empty:
        assert (rerouted["split"] == "demo").all()


def test_generate_splits_holdout_exclusion_preserves_leakage_safety():
    """The hold-out fix must not reintroduce clip/speaker leakage -- rerun
    the full leakage assertion after hold-out rerouting."""
    df = _synthetic_manifest()
    for seed in (0, 1, 2, 3):
        result = generate_splits(df, seed=seed)
        assert_no_leakage(result)  # should not raise


def test_generate_splits_custom_real_speech_fn():
    """A caller-supplied real_speech_fn narrower than the default (e.g.
    only generator == "real") is honored instead of the speaker_anon-based
    default."""
    df = _synthetic_manifest()

    calls = {"count": 0}

    def only_generator_real(row):
        calls["count"] += 1
        return row.get("generator") == "real"

    result = generate_splits(df, seed=0, real_speech_fn=only_generator_real)
    assert calls["count"] > 0
    assert_no_leakage(result, real_speech_fn=only_generator_real)
