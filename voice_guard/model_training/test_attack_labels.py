"""Tests for attack-type labeling (remediation track 4). See
docs/superpowers/specs/2026-09-11-attack-type-differentiator-design.md §3
for the labeling coverage table this implements."""
from __future__ import annotations

from pathlib import Path

from attack_labels import (
    ATTACK_ID_TO_TYPE,
    attack_type_for_file,
    load_asvspoof_attack_map,
)


def test_attack_id_to_type_covers_a01_through_a19():
    for i in range(1, 20):
        assert f"A{i:02d}" in ATTACK_ID_TO_TYPE
        assert ATTACK_ID_TO_TYPE[f"A{i:02d}"] in ("tts", "vc")


def test_attack_id_to_type_known_vc_ids():
    # Empirically confirmed via model_training/test_assets/
    # voice_conversion_asvspoof_a17.wav (see that dir's README).
    assert ATTACK_ID_TO_TYPE["A17"] == "vc"
    assert ATTACK_ID_TO_TYPE["A18"] == "vc"
    assert ATTACK_ID_TO_TYPE["A19"] == "vc"


def test_load_asvspoof_attack_map_parses_trial_metadata_format(tmp_path: Path):
    protocol = tmp_path / "trial_metadata.txt"
    protocol.write_text(
        "LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval\n"
        "LA_0009 LA_E_1428848 alaw ita_tx A17 spoof notrim eval\n"
        "LA_0009 LA_E_0000001 alaw ita_tx - bonafide notrim eval\n"
    )
    result = load_asvspoof_attack_map(protocol)
    assert result["LA_E_9332881"] == "tts"
    assert result["LA_E_1428848"] == "vc"
    assert "LA_E_0000001" not in result  # bonafide rows carry no attack type


def test_attack_type_for_file_uses_per_file_map_first():
    assert attack_type_for_file(
        Path("/x/LA_E_1428848.wav"), "/x", {"LA_E_1428848": "vc"}
    ) == "vc"


def test_attack_type_for_file_falls_back_to_unknown_with_no_map_or_default():
    assert attack_type_for_file(Path("/some/unlabeled/dir/1.wav"), "/some/unlabeled/dir", None) == "unknown"


def test_build_attack_type_maps_finds_fake2021_if_protocol_present():
    from build_attack_type_maps import build_attack_type_maps

    maps = build_attack_type_maps()
    fake2021 = str((Path(__file__).parent / "data" / "fake2021").resolve())
    protocol_path = (
        Path(__file__).parent / "data" / "asvspoof2021_la" / "LA-keys-full"
        / "keys" / "LA" / "CM" / "trial_metadata.txt"
    )
    if protocol_path.exists():
        assert fake2021 in maps
        assert len(maps[fake2021]) > 0
    else:
        assert fake2021 not in maps  # nothing to assert if the real corpus isn't present on this machine
