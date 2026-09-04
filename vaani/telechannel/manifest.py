"""
manifest.py — Per-clip metadata logging and leakage-safe split generation.

Independent of telechannel/pipeline.py: this module doesn't import from it
and doesn't care how a clip was generated, only what its metadata says. Any
caller (the pipeline, a QA script, a notebook) can call `write_manifest_row`
after producing a clip.

Manifest schema (one row per output clip; see DOC1_TELECHANNEL_SPEC.md
section 1.6 for the full nested JSON shape this flattens):

    source_clip   -- id of the ORIGINAL source recording/utterance this
                     clip was derived from (before windowing/channel sim).
                     This is the leakage-critical field: splitting must
                     happen at this granularity, never at the 2s-window
                     level, or the same source clip leaks across splits.
    speaker_anon   -- anonymized speaker id (e.g. "spk_041"), or None/NaN
                     for synthetic clips with no real human speaker behind
                     them. See "Real speech" below for how this drives the
                     extra speaker-level split rule.
    generator      -- "real" for genuine recordings, or the synthesis
                     model name for generated speech (e.g. "melo", "bark",
                     "rvc_team", "indic_tts"), matching DOC1 section 1.6's
                     `source.generator` field.
    accent         -- language/accent tag, e.g. "hi-en", "ta-en", "en-in".
    codec_chain    -- list (or single string) of codec names applied, in
                     order, e.g. ["libopencore_amrnb", "pcm_mulaw"] for a
                     tandem transcode, or a single-entry chain for a
                     one-hop recipe. Matches telechannel/configs/
                     channels.yaml's `codec` list shape.
    bitrate        -- bitrate string of the (last) codec hop, e.g. "7.4k".
    loss_pct       -- packet loss percentage actually realized on this clip.
    snr_db         -- speech-to-noise loudness ratio in dB.
    rir_id         -- id of the room impulse response profile used
                     (e.g. "slr28_sim_012" or a channels.yaml room name).
    noise          -- noise type/pool used (e.g. "musan_babble",
                     "freesound_horn"), needed for the noise-based hold-out
                     filter (Step 3 / master plan section 5.4).
    license        -- mandatory (DOC1 1.6 hard rule 4): "CC0", "MIT",
                     "apache2", "team_consent_v2", etc. A row with a
                     missing/blank license fails QA and is rejected here.
    consent        -- consent status/id backing this row (e.g. a
                     `consent_id` from the Doc 6 consent register, or a
                     boolean/tag like "public"/"internal_only"). Mandatory
                     for the same reason as `license`.

Additional columns beyond REQUIRED_FIELDS may be passed through
`write_manifest_row`'s `metadata` dict (e.g. `clip_id`, `origin`,
`rng_seed`) and are stored as-is; they're just not validated as required.

Real speech, for the speaker-level split rule
-----------------------------------------------
The master plan (section 5.4) and DOC1 (1.6 hard rule 3) both require that
*real* speech additionally be split at the speaker level, on top of the
source-clip-level rule that applies to every row. This module treats a row
as "real speech" when `speaker_anon` is present (non-null/non-empty) --
i.e. there's an actual anonymized human speaker id backing it. Synthetic
rows are expected to carry `speaker_anon = None` (there is no human speaker
being cloned's identity to protect at the split level; the identity that
matters there, if any, is the source real speaker being impersonated in
some rows, which is out of scope for this leakage rule and instead handled
via the `generator` hold-out filter). This choice is deliberately more
inclusive than gating on `generator == "real"` alone: a voice-cloned clip
that lists both a `generator` (the cloning model) AND a `speaker_anon` (the
donor whose voice was cloned) is still `speaker_anon`-grouped here, which
is the conservative/safe direction for a leakage rule -- it can only ever
merge more rows into the same split, never let a real leak slip through.
Callers who want the stricter/narrower "only rows literally labeled
real" behavior can pass their own `real_speech_fn` to `generate_splits`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "manifest.parquet"

# Fields DOC1 1.6's hard rules treat as mandatory for every manifest row.
# `license` and `consent` failing QA (blank/missing) is DOC1 1.8's explicit
# "any required field missing/invalid" manifest QA gate.
REQUIRED_FIELDS = [
    "source_clip",
    "generator",
    "accent",
    "codec_chain",
    "bitrate",
    "loss_pct",
    "snr_db",
    "rir_id",
    "license",
    "consent",
    "speaker_anon",
    "noise",
]

# speaker_anon is required-as-a-column but is legitimately null for
# synthetic clips with no real speaker behind them, so it's exempt from the
# "must be non-empty" check that the other required fields get.
_NULLABLE_REQUIRED_FIELDS = {"speaker_anon"}

SPLIT_NAMES = ("train", "val", "test", "demo")


class ManifestValidationError(ValueError):
    """Raised when a manifest row is missing a required field or has an
    invalid (blank/None) value for a non-nullable required field."""


def _is_blank(value):
    if value is None:
        return True
    try:
        if isinstance(value, float) and np.isnan(value):
            return True
    except TypeError:
        pass
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def validate_metadata(metadata):
    """
    Validate a single manifest row's metadata dict against REQUIRED_FIELDS.

    Raises ManifestValidationError if a required field is absent, or is
    present but blank/None for a field that isn't allowed to be null
    (everything except `speaker_anon`).
    """
    missing = [f for f in REQUIRED_FIELDS if f not in metadata]
    if missing:
        raise ManifestValidationError(
            f"manifest row missing required field(s): {missing}"
        )

    invalid = [
        f for f in REQUIRED_FIELDS
        if f not in _NULLABLE_REQUIRED_FIELDS and _is_blank(metadata[f])
    ]
    if invalid:
        raise ManifestValidationError(
            f"manifest row has blank/None value for required field(s): {invalid}"
        )


def write_manifest_row(metadata, manifest_path=None):
    """
    Append one clip's metadata as a row to the Parquet manifest.

    Args:
        metadata: dict of column -> value for this row. Must contain every
            key in REQUIRED_FIELDS (see module docstring); extra keys are
            stored as additional columns. `codec_chain`, if a list, is
            stored as a Python list (pyarrow/parquet natively supports
            list-typed columns).
        manifest_path: Path to the manifest.parquet file. Defaults to
            DEFAULT_MANIFEST_PATH. Parent directories are created if
            missing. If the file already exists, the new row is
            concatenated onto the existing table and the whole thing is
            rewritten (Parquet has no native single-row append; for the
            corpus sizes here -- hundreds of thousands of rows -- this is
            fine as a batched/periodic write, not necessarily called once
            per individual clip in the hot path).

    Raises:
        ManifestValidationError: if `metadata` fails validate_metadata.

    Returns:
        pandas.DataFrame: the full manifest after the append.
    """
    validate_metadata(metadata)

    path = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([metadata])

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_parquet(path, index=False)
    return combined


def read_manifest(manifest_path=None):
    """Read and return the manifest.parquet as a pandas DataFrame."""
    path = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST_PATH
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Hold-out filters (Step 3): one-liner filters on manifest fields, per
# DOC1 1.6 hard rule 1 -- "splits filter on manifest fields, never folders" --
# and the four generalization hold-out sets named in master plan section 5.4
# (unseen generator, unseen codec, unseen noise, real recorded calls).
# ---------------------------------------------------------------------------


def _codec_chain_contains(codec_chain, codec_name):
    if isinstance(codec_chain, (list, tuple, np.ndarray)):
        return codec_name in list(codec_chain)
    return codec_chain == codec_name


def filter_holdout_unseen_generator(df, generator="rvc_team"):
    """Rows whose `generator` matches the held-out (never-seen-in-training)
    generator, default "rvc_team" per master plan section 5.4."""
    return df[df["generator"] == generator]


def filter_holdout_unseen_codec(df, codec="amr_wb"):
    """Rows whose `codec_chain` contains the held-out codec, default
    "amr_wb" per master plan section 5.4. Handles both a list-valued
    codec_chain (tandem transcodes) and a single codec-name string."""
    return df[df["codec_chain"].apply(lambda c: _codec_chain_contains(c, codec))]


def filter_holdout_unseen_noise(df, noise="freesound_horn"):
    """Rows whose `noise` matches the held-out noise pool, default
    "freesound_horn" per master plan section 5.4."""
    return df[df["noise"] == noise]


def filter_holdout_real_calls(df, origin_column="origin", origin_value="real_call"):
    """Rows that are real recorded calls (DOC1 1.6's `origin == "real_call"`
    hold-out). `origin` is an optional/extra manifest column (not in
    REQUIRED_FIELDS), so this returns an empty frame if the column isn't
    present rather than raising."""
    if origin_column not in df.columns:
        return df.iloc[0:0]
    return df[df[origin_column] == origin_value]


HOLDOUT_FILTERS = {
    "unseen_generator": filter_holdout_unseen_generator,
    "unseen_codec": filter_holdout_unseen_codec,
    "unseen_noise": filter_holdout_unseen_noise,
    "real_calls": filter_holdout_real_calls,
}


def get_holdout_sets(df):
    """Return {name: DataFrame} for all four hold-out sets, applying each
    filter in HOLDOUT_FILTERS with its documented default value."""
    return {name: fn(df) for name, fn in HOLDOUT_FILTERS.items()}


def holdout_mask(df):
    """
    Boolean Series (indexed like `df`), True for any row matching ANY of
    the four HOLDOUT_FILTERS (with their documented default values).

    This is the single source of truth `generate_splits` uses to decide
    which rows/groups must never end up in `train` -- see that function's
    docstring for why this matters (a hold-out-matching row leaking into
    train silently invalidates the generalization claim the hold-out set
    exists to support).
    """
    mask = pd.Series(False, index=df.index)
    for subset in get_holdout_sets(df).values():
        mask.loc[subset.index] = True
    return mask


# ---------------------------------------------------------------------------
# Step 2: leakage-safe split generation.
# ---------------------------------------------------------------------------


class _UnionFind:
    """Minimal disjoint-set-union, keyed by arbitrary hashable ids."""

    def __init__(self, items):
        self._parent = {item: item for item in items}

    def find(self, item):
        parent = self._parent[item]
        if parent != item:
            parent = self.find(parent)
            self._parent[item] = parent
        return parent

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _default_real_speech_fn(row):
    """Default "real speech" predicate: True when `speaker_anon` is
    present/non-blank. See the module docstring's "Real speech" section
    for the reasoning behind this choice."""
    return not _is_blank(row.get("speaker_anon"))


def _leakage_groups(df, real_speech_fn):
    """
    Build source_clip -> group_id, where two source_clips are forced into
    the same group (and therefore the same split) if:
      - they're literally the same source_clip (trivially, always), or
      - some real-speech row ties them to the same speaker_anon.

    This guarantees BOTH the source-clip-level rule (every row of a given
    source_clip is one group, since a source_clip only ever maps to itself
    unless merged) AND the speaker-level rule for real speech (all
    source_clips ever attributed to the same speaker_anon end up in one
    group, hence one split).
    """
    all_clips = df["source_clip"].unique().tolist()
    uf = _UnionFind(all_clips)

    real_mask = df.apply(real_speech_fn, axis=1)
    real_df = df[real_mask]

    for _, group in real_df.groupby("speaker_anon"):
        clips = group["source_clip"].unique().tolist()
        first = clips[0]
        for other in clips[1:]:
            uf.union(first, other)

    return {clip: uf.find(clip) for clip in all_clips}


def generate_splits(
    df,
    val_frac=0.15,
    test_frac=0.15,
    demo_frac=0.05,
    seed=42,
    real_speech_fn=None,
    exclude_holdouts_from_train=True,
    holdout_mask_fn=None,
    holdout_route=None,
):
    """
    Assign a `split` column (one of SPLIT_NAMES) to every row of `df`,
    leakage-safely: no `source_clip` value is ever assigned to more than
    one split, and no `speaker_anon` value belonging to a "real speech" row
    (see `real_speech_fn` / the module docstring) is ever assigned to more
    than one split either.

    Splitting happens on GROUPS of source_clip (source_clip itself, plus
    any source_clips transitively tied together by a shared real-speech
    speaker_anon -- see `_leakage_groups`), never on individual rows or
    windows, which is the master-plan section 5.4 / DOC1 1.6 hard rule
    this whole module exists to enforce.

    Hold-out exclusion from train
    ------------------------------
    By default (`exclude_holdouts_from_train=True`), any leakage-group
    containing at least one row matching ANY of the four HOLDOUT_FILTERS
    (see `holdout_mask`) is never assigned to `train`, even if the random
    group shuffle would otherwise have put it there. Without this, a
    hold-out-matching clip/speaker could leak into train by chance, which
    would silently invalidate the generalization claims the hold-out sets
    exist to support (master plan section 5.4 calls these "four held-out
    test sets" whose EER numbers ARE the generalization story -- a
    contaminated train set makes those numbers dishonest without anyone
    noticing).

    Design choice: hold-out-matching groups that would have landed in
    `train` are rerouted to `test` (not a new dedicated split name),
    because master plan section 5.4 already frames all four hold-out sets
    as "held-out TEST sets" and downstream training/eval code only needs
    to know about the existing SPLIT_NAMES. Groups that the random
    assignment already put in `val`/`test`/`demo` are left alone (only a
    `train` assignment is overridden) -- being held out of train is the
    hard requirement; which non-train split a hold-out group lands in is
    incidental. This does shift the realized train/test fractions further
    from the requested `val_frac`/`test_frac`/`demo_frac` (see the
    approximate-fractions note below); that's an accepted trade-off,
    documented the same way in `run_corpus.yaml`. Pass
    `exclude_holdouts_from_train=False` to disable this (e.g. for a
    debugging run), or `holdout_route="demo"` (etc.) to reroute elsewhere,
    or `holdout_mask_fn` to override which rows count as hold-out-matching.

    Args:
        df: manifest DataFrame (e.g. from read_manifest()). Must have
            `source_clip` and `speaker_anon` columns.
        val_frac, test_frac, demo_frac: target fraction of ROWS for each
            split; the remainder goes to `train`. Because whole groups
            (not individual rows) are assigned, actual fractions will
            deviate somewhat from these targets when groups are large or
            uneven -- leakage-safety is exact, fraction-matching is
            approximate by construction. Hold-out rerouting (see above)
            adds a further, one-directional deviation (train can only
            shrink, never grow, relative to the un-rerouted assignment).
        seed: RNG seed for the (deterministic, reproducible) group shuffle
            order that fractions are measured against.
        real_speech_fn: optional callable(row) -> bool overriding the
            default "real speech" predicate (`speaker_anon` non-blank).
        exclude_holdouts_from_train: if True (default), enforce the
            hold-out exclusion described above.
        holdout_mask_fn: optional callable(df) -> boolean Series overriding
            the default `holdout_mask` (which ORs together all four
            HOLDOUT_FILTERS).
        holdout_route: split name to reroute hold-out-matching
            `train`-assigned groups to. Defaults to `"test"`.

    Returns:
        A copy of `df` with a new `split` column (values from SPLIT_NAMES).

    Raises:
        ValueError: if val_frac + test_frac + demo_frac >= 1.0, or if
            `df` is missing `source_clip` or `speaker_anon`.
    """
    if val_frac + test_frac + demo_frac >= 1.0:
        raise ValueError(
            "val_frac + test_frac + demo_frac must be < 1.0 "
            f"(got {val_frac + test_frac + demo_frac})"
        )
    for col in ("source_clip", "speaker_anon"):
        if col not in df.columns:
            raise ValueError(f"generate_splits requires a {col!r} column")

    if real_speech_fn is None:
        real_speech_fn = _default_real_speech_fn
    if holdout_mask_fn is None:
        holdout_mask_fn = holdout_mask
    if holdout_route is None:
        holdout_route = "test"

    group_of_clip = _leakage_groups(df, real_speech_fn)

    df = df.copy()
    df["_leak_group"] = df["source_clip"].map(group_of_clip)

    holdout_groups = set()
    if exclude_holdouts_from_train:
        is_holdout_row = holdout_mask_fn(df)
        holdout_groups = set(df.loc[is_holdout_row.values, "_leak_group"].unique())

    group_sizes = df.groupby("_leak_group").size()
    groups = group_sizes.index.to_numpy()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    shuffled_groups = groups[order]
    shuffled_sizes = group_sizes.loc[shuffled_groups].to_numpy()

    total_rows = len(df)
    cum = np.cumsum(shuffled_sizes)

    train_end = (1.0 - val_frac - test_frac - demo_frac) * total_rows
    val_end = train_end + val_frac * total_rows
    test_end = val_end + test_frac * total_rows
    # demo gets the remainder (up to total_rows).

    split_of_group = {}
    for group_id, cum_count in zip(shuffled_groups, cum):
        if cum_count <= train_end:
            split_of_group[group_id] = "train"
        elif cum_count <= val_end:
            split_of_group[group_id] = "val"
        elif cum_count <= test_end:
            split_of_group[group_id] = "test"
        else:
            split_of_group[group_id] = "demo"

    # Enforce hold-out exclusion from train: a hold-out-matching group that
    # the random shuffle assigned to train is rerouted -- see the
    # "Hold-out exclusion from train" section of this function's docstring.
    for group_id in holdout_groups:
        if split_of_group.get(group_id) == "train":
            split_of_group[group_id] = holdout_route

    df["split"] = df["_leak_group"].map(split_of_group)
    df = df.drop(columns=["_leak_group"])
    return df


def assert_no_leakage(df_with_split, real_speech_fn=None):
    """
    Verify a split-assigned DataFrame (as returned by generate_splits) has
    no leakage: no `source_clip` spans more than one split, and no
    `speaker_anon` belonging to a real-speech row spans more than one
    split. Raises AssertionError with details on the first violation type
    found; returns None (silently) if clean.
    """
    if real_speech_fn is None:
        real_speech_fn = _default_real_speech_fn

    clip_splits = df_with_split.groupby("source_clip")["split"].nunique()
    leaked_clips = clip_splits[clip_splits > 1]
    assert leaked_clips.empty, (
        f"source_clip leakage across splits: {leaked_clips.index.tolist()}"
    )

    real_mask = df_with_split.apply(real_speech_fn, axis=1)
    real_df = df_with_split[real_mask & df_with_split["speaker_anon"].notna()]
    speaker_splits = real_df.groupby("speaker_anon")["split"].nunique()
    leaked_speakers = speaker_splits[speaker_splits > 1]
    assert leaked_speakers.empty, (
        f"speaker_anon leakage across splits (real speech): "
        f"{leaked_speakers.index.tolist()}"
    )
