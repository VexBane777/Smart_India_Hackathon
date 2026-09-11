"""Builds the attack_type_maps argument for dataset.make_file_specs from
this project's corpus layout. See
docs/superpowers/specs/2026-09-11-attack-type-differentiator-design.md §3
for the source-by-source rationale.

Per-file maps are {utt_id: attack_id} ("A01".."A19"). attack_labels
derives the tts/vc type from the id, and the id itself is what makes
leave-attack-out evaluation possible (v12)."""
from __future__ import annotations

from pathlib import Path

from attack_labels import (
    load_asvspoof_2019_train_attack_id_map,
    load_asvspoof_attack_id_map,
    register_tts_only_directory,
)

HERE = Path(__file__).resolve().parent

ASV2021_PROTOCOL = HERE / "data" / "asvspoof2021_la" / "LA-keys-full" / "keys" / "LA" / "CM" / "trial_metadata.txt"
ASV2019_TRAIN_PROTOCOL = HERE / "data" / "asvspoof2019_la_protocol_only" / "ASVspoof2019.LA.cm.train.trn.txt"


def build_attack_type_maps() -> dict[str, dict[str, str] | None]:
    maps: dict[str, dict[str, str] | None] = {}

    # data/fake2021: ASVspoof2021 LA eval files (LA_E_*) are labeled via
    # trial_metadata.txt. Its ASVspoof2019 LA dev files (LA_D_*, ~12%) have no
    # protocol on this machine and stay "unknown" (masked from the attack loss).
    if ASV2021_PROTOCOL.exists():
        maps[str((HERE / "data" / "fake2021").resolve())] = load_asvspoof_attack_id_map(ASV2021_PROTOCOL)

    # Flat all-TTS directories (each generation script's docstring, spec §3).
    for d in (
        HERE / "data" / "accents" / "fake" / "en_foreign",
        HERE / "data" / "accents" / "fake" / "hi_native",
        HERE / "data" / "accents" / "fake" / "hi_foreign",
    ):
        if d.exists():
            register_tts_only_directory(d)
            maps[str(d.resolve())] = None  # None = use the directory default just registered

    # data/fake (ASVspoof2019 LA train, A01-A06): the protocol-only download
    # (README "Attack-type protocol only").
    if ASV2019_TRAIN_PROTOCOL.exists():
        maps[str((HERE / "data" / "fake").resolve())] = load_asvspoof_2019_train_attack_id_map(ASV2019_TRAIN_PROTOCOL)

    return maps
