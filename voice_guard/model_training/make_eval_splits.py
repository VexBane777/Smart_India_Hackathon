"""Writes the committed held-out split manifest,
eval_splits/held_out_split_v1.json.

Every core held-out set is split 50/50 by a stable hash of its file id into:
  select: the ONLY data checkpoint selection may read
          (select_best_checkpoint_seqcnn.py);
  test:   read once per candidate model for the final report (evaluate.py).
Test-only sets (MLAAD, accent cells) go entirely to `test`.

Why: v11 selected its checkpoint on fake_itw_held, which was also the fake
half of its headline EER test, so the reported 0.0624 was optimistic.

Known limitation: In-the-Wild's meta.csv (speaker labels) is no longer on
this machine and its HF mirror is empty, so select/test is split by FILE,
not speaker. Train vs held was already speaker-disjoint
(data/prep_in_the_wild.py); only select and test may share speakers. That
makes select a slightly optimistic proxy for test, which is harmless for
selection. The headline number is always computed on test.

Refuses to overwrite an existing manifest (a new split silently invalidates
every earlier comparison). Bump the version instead.

Usage: python make_eval_splits.py [--out eval_splits/held_out_split_v1.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from corpus import EVAL_SETS, SPLIT_MANIFEST, TRAIN_SETS_V12
from dataset import file_id, list_audio_files, stable_seed, stable_unit

SPLIT_SALT = "held_out_split_v1"


def build_manifest() -> dict:
    sets = {}
    for sd in EVAL_SETS:
        files = list_audio_files(sd.path, recursive=sd.recursive)
        if not files:
            raise SystemExit(f"{sd.name}: no audio in {sd.path}")
        ids = [file_id(f) for f in files]
        if sd.cap is not None and len(ids) > sd.cap:
            ids = sorted(ids, key=lambda i: stable_seed(i, SPLIT_SALT, "cap"))[: sd.cap]
        ids.sort()
        if sd.test_only:
            select, test = [], ids
        else:
            select = [i for i in ids if stable_unit(i, SPLIT_SALT) < 0.5]
            test = [i for i in ids if stable_unit(i, SPLIT_SALT) >= 0.5]
        sets[sd.name] = {"rel_dir": sd.rel_dir, "label": sd.label, "group": sd.group,
                         "n_files": len(ids), "select": select, "test": test}
    _check_no_train_overlap(sets)
    return {
        "version": 1, "salt": SPLIT_SALT,
        "method": "stable sha256 hash of data-relative file id; <0.5 -> select, else test",
        "limitation": "In-the-Wild select/test split is file-level, not speaker-level "
                      "(meta.csv unavailable); train vs held-out is speaker-disjoint.",
        "sets": sets,
    }


def _check_no_train_overlap(sets: dict) -> None:
    """No eval file may also be a training file (by data-relative id), and
    no ITW held basename may appear in ITW train."""
    train_ids = set()
    for sd in TRAIN_SETS_V12:
        train_ids |= {file_id(f) for f in list_audio_files(sd.path, recursive=sd.recursive)}
    for name, entry in sets.items():
        clash = train_ids & set(entry["select"] + entry["test"])
        if clash:
            raise SystemExit(f"{name}: {len(clash)} eval files are also training files, e.g. {sorted(clash)[:3]}")
    train_itw = {Path(i).name for i in train_ids if i.startswith(("real_itw_train/", "fake_itw_train/"))}
    for name in ("itw_real", "itw_fake"):
        clash = train_itw & {Path(i).name for i in sets[name]["select"] + sets[name]["test"]}
        if clash:
            raise SystemExit(f"{name}: basenames shared with ITW train: {sorted(clash)[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=SPLIT_MANIFEST)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite. Bump the split version instead.")
    manifest = build_manifest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1))
    for name, e in manifest["sets"].items():
        print(f"{name:26s} files={e['n_files']:5d} select={len(e['select']):5d} test={len(e['test']):5d}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
