"""Report per-cell coverage against TARGET_MINUTES_PER_CELL, and flag which
cells are still thin after the automated (prep_accents_real.py +
gen_accents_fake.py) and self-recording (ingest_self_recordings.py) passes.

This is the literal checklist for the manual-supplement half of the "Approach
C" data plan: whatever's flagged "THIN" here is what to go find/scrape more
of by hand and drop into the matching data/accents/<label>/<cell>/ folder
(plus a manifest row — see accent_common.MANIFEST_FIELDS for the schema; a
"source" of "manual" is fine).

Needs no network/GPU — safe to run any time to check progress.

Usage:
    python report_accent_coverage.py
"""
from __future__ import annotations

from collections import defaultdict

from accent_common import (
    ACCENTS_ROOT,
    CELLS,
    LABELS,
    MANIFEST_PATH,
    TARGET_MINUTES_PER_CELL,
    cell_dir,
    read_manifest,
)


def _duration_minutes(rows: list[dict]) -> float:
    total = 0.0
    for r in rows:
        try:
            total += float(r["duration_s"])
        except (KeyError, ValueError):
            pass  # rows with unknown duration (e.g. gen_accents_fake.py
            # doesn't compute it inline) just don't count toward the total;
            # rerun with a duration backfill pass if that matters later.
    return total / 60.0


def main() -> None:
    rows = read_manifest()
    if not rows:
        print(f"No manifest at {MANIFEST_PATH} yet — nothing has been ingested.")
        return

    by_cell_label: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_cell_label[(r["cell"], r["label"])].append(r)

    print(f"{'cell':<12} {'label':<6} {'clips':>6} {'minutes':>9}  status")
    print("-" * 50)
    thin = []
    for cell in CELLS:
        for label in LABELS:
            cell_rows = by_cell_label.get((cell, label), [])
            minutes = _duration_minutes(cell_rows)
            on_disk = len(list(cell_dir(label, cell).glob("*.wav"))) if cell_dir(label, cell).exists() else 0
            status = "ok" if minutes >= TARGET_MINUTES_PER_CELL else "THIN"
            if status == "THIN":
                thin.append((label, cell, minutes))
            note = "" if on_disk == len(cell_rows) else f"  (manifest={len(cell_rows)} vs on-disk wavs={on_disk}, mismatch!)"
            print(f"{cell:<12} {label:<6} {len(cell_rows):>6} {minutes:>9.1f}  {status}{note}")

    if thin:
        print(f"\n{len(thin)} cell(s) below the {TARGET_MINUTES_PER_CELL:.0f}-minute target - "
              "manual supplement needed:")
        for label, cell, minutes in thin:
            print(f"  - {label}/{cell}: {minutes:.1f} min so far")
    else:
        print("\nAll cells at or above target.")


if __name__ == "__main__":
    main()
