"""Builds the attack_type_maps argument for dataset.build_examples from
this project's actual corpus layout — see
docs/superpowers/specs/2026-09-11-attack-type-differentiator-design.md §3
for the source-by-source rationale. Kept separate from train_seq_cnn.py
so check_corpus.py can report coverage without running a training job."""
from __future__ import annotations

from pathlib import Path

from attack_labels import (
    load_asvspoof_2019_train_attack_map,
    load_asvspoof_attack_map,
    register_tts_only_directory,
)

HERE = Path(__file__).resolve().parent


def build_attack_type_maps() -> dict[str, dict[str, str] | None]:
    maps: dict[str, dict[str, str] | None] = {}

    # data/fake2021: per-file labels via ASVspoof2021's trial_metadata.txt
    protocol = HERE / "data" / "asvspoof2021_la" / "LA-keys-full" / "keys" / "LA" / "CM" / "trial_metadata.txt"
    if protocol.exists():
        per_file_map = load_asvspoof_attack_map(protocol)
        maps[str((HERE / "data" / "fake2021").resolve())] = per_file_map

    # Directories where every fake file is TTS by construction (verified
    # against each generation script's own docstring, spec §3) — flat
    # directories of .wav files only (build_examples globs non-recursively,
    # so a nested-by-architecture layout like data/mlaad_en500/fake/en
    # would need per-subdirectory registration, not included here since
    # MLAAD isn't part of the current training corpus scope anyway).
    tts_only_dirs = [
        HERE / "data" / "accents" / "fake" / "en_foreign",
        HERE / "data" / "accents" / "fake" / "hi_native",
        HERE / "data" / "accents" / "fake" / "hi_foreign",
    ]
    for d in tts_only_dirs:
        if d.exists():
            register_tts_only_directory(d)
            maps[str(d.resolve())] = None  # None = use the directory default just registered

    # data/fake (ASVspoof2019 LA train): per-file labels via the
    # protocol-only re-download (see README.md's "Attack-type protocol
    # only" note). Falls back to "unknown" (no entry in maps) until that
    # protocol is actually downloaded onto this machine.
    train_protocol = HERE / "data" / "asvspoof2019_la_protocol_only" / "ASVspoof2019.LA.cm.train.trn.txt"
    if train_protocol.exists():
        maps[str((HERE / "data" / "fake").resolve())] = load_asvspoof_2019_train_attack_map(train_protocol)

    return maps
