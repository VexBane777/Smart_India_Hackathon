"""Shared schema for the accent/clone data-collection subsystem
(voice_guard/state.md "Open / unresolved" -> accent + clone coverage).

Every prep/gen/ingest/report script in this subsystem imports from here so
the folder layout and manifest columns can't drift between them. Don't
duplicate CELLS or the manifest header inline in another script — import it.

Target matrix (8 cells): {en, hi} x {native, foreign} x {real, fake}.
"native"/"foreign" is a coarse proxy — "does this speaker's accent tag read
as a native-anglophone-region accent (for en) / mainstream-Hindi-belt accent
(for hi)", not a verified L1 claim. Good enough for training-data diversity,
not a linguistic fact about any individual speaker. Documented in data/README.md.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACCENTS_ROOT = HERE / "accents"
MANIFEST_PATH = HERE / "accent_manifest.csv"

LANGS = ("en", "hi")
NATIVITY = ("native", "foreign")
LABELS = ("real", "fake")

CELLS = tuple(f"{lang}_{nat}" for lang in LANGS for nat in NATIVITY)

# Rough per-cell minimum before report_accent_coverage.py stops flagging it.
# Chosen as "enough to matter for a 63-feature MLP's training mix", not a
# statistically derived number — revisit once the training-pipeline step
# (next subsystem) shows what actually moves held-out EER.
TARGET_MINUTES_PER_CELL = 30.0

MANIFEST_FIELDS = [
    "file",        # path relative to ACCENTS_ROOT
    "label",       # "real" | "fake"
    "cell",        # one of CELLS, e.g. "hi_foreign"
    "source",      # e.g. "common_voice_17_0", "xtts_v2_clone", "self_recorded", "manual"
    "speaker_id",  # dataset speaker id, "self", or "unknown"
    "duration_s",  # float
]


def cell_dir(label: str, cell: str) -> Path:
    assert label in LABELS, label
    assert cell in CELLS, cell
    return ACCENTS_ROOT / label / cell


def ensure_dirs() -> None:
    for label in LABELS:
        for cell in CELLS:
            cell_dir(label, cell).mkdir(parents=True, exist_ok=True)


def manifest_writer_append(rows: list[dict]) -> None:
    """Append rows to accent_manifest.csv, writing the header once if new."""
    is_new = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if is_new:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def read_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, newline="") as f:
        return list(csv.DictReader(f))
