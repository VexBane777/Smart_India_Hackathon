"""Attack labeling for the attack-type ({tts, vc}) head. See
docs/superpowers/specs/2026-09-11-attack-type-differentiator-design.md §3
for per-source-directory coverage.

The ASVspoof2019 LA attack table below was **verified 2026-09-11** against
Wang et al., "ASVspoof 2019: A large-scale public database of synthesized,
converted and replayed speech" (arXiv:1911.01601), Table 1:
- A01-A04, A07-A12, A16: TTS (A16 is the same system as A04).
- A05, A06, A17-A19: VC (A19 is the same system as A06).
- A13-A15: TTS+VC hybrids (a TTS front end feeding a VC stage). They are
  bucketed as "vc" because the VC stage is the last signal-transforming step.
  This is a labeling choice, not a fact about the systems. Evaluate them as
  their own group if it matters.
A17-A19 are also confirmed empirically:
model_training/test_assets/voice_conversion_asvspoof_a17.wav was produced by
conversion, not synthesis.

Since v12 each fake also carries its **attack ID** (A01-A19, or unknown), not
just its type. That enables leave-attack-out evaluation of the attack-type
head: a system masked out of the attack-type loss can be scored as unseen.
"""
from __future__ import annotations

from pathlib import Path

ATTACK_ID_TO_TYPE: dict[str, str] = {
    "A01": "tts", "A02": "tts", "A03": "tts", "A04": "tts",
    "A05": "vc", "A06": "vc",
    "A07": "tts", "A08": "tts", "A09": "tts", "A10": "tts",
    "A11": "tts", "A12": "tts",
    "A13": "vc",  # TTS+VC hybrid, bucketed as vc (see module docstring)
    "A14": "vc", "A15": "vc",  # TTS+VC hybrids, same bucketing
    "A16": "tts",  # = A04
    "A17": "vc", "A18": "vc",
    "A19": "vc",  # = A06
}
HYBRID_ATTACK_IDS: frozenset[str] = frozenset({"A13", "A14", "A15"})

# Integer codes stored in feature caches: 0 = none/unknown, 1..19 = A01..A19.
ATTACK_ID_UNKNOWN = 0


def attack_id_to_int(attack_id: str | None) -> int:
    if attack_id and len(attack_id) == 3 and attack_id[0] == "A" and attack_id[1:].isdigit():
        n = int(attack_id[1:])
        if 1 <= n <= 19:
            return n
    return ATTACK_ID_UNKNOWN


def int_to_attack_id(code: int) -> str | None:
    return f"A{int(code):02d}" if 1 <= int(code) <= 19 else None


# Directories where every fake file is TTS by construction (verified
# against each source's own generation script docstring, design spec §3).
# Keyed by resolved absolute path string.
DIRECTORY_DEFAULT_ATTACK_TYPE: dict[str, str] = {}


def register_tts_only_directory(path: Path) -> None:
    """Call once per known-all-TTS directory (MLAAD, XTTS/YourTTS/MMS
    accent-expansion fakes) during corpus setup, so attack_type_for_file
    can resolve it without a per-file map."""
    DIRECTORY_DEFAULT_ATTACK_TYPE[str(Path(path).resolve())] = "tts"


def load_asvspoof_attack_id_map(protocol_path: Path) -> dict[str, str]:
    """ASVspoof2021 trial_metadata.txt format (speaker utt codec tx
    attack_id key trim subset) -> {utt_id: "Axx"}. Spoof rows only."""
    result: dict[str, str] = {}
    with open(protocol_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            utt_id, attack_id, key = parts[1], parts[4], parts[5]
            if key == "spoof" and attack_id in ATTACK_ID_TO_TYPE:
                result[utt_id] = attack_id
    return result


def load_asvspoof_2019_train_attack_id_map(protocol_path: Path) -> dict[str, str]:
    """ASVspoof2019 LA train protocol format (speaker utt - attack_id key)
    -> {utt_id: "Axx"}. Spoof rows only."""
    result: dict[str, str] = {}
    with open(protocol_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            utt_id, attack_id, key = parts[1], parts[3], parts[4]
            if key == "spoof" and attack_id in ATTACK_ID_TO_TYPE:
                result[utt_id] = attack_id
    return result


def load_asvspoof_attack_map(protocol_path: Path) -> dict[str, str]:
    """{utt_id: "tts"|"vc"} from a trial_metadata.txt-format protocol."""
    return {u: ATTACK_ID_TO_TYPE[a] for u, a in load_asvspoof_attack_id_map(protocol_path).items()}


def load_asvspoof_2019_train_attack_map(protocol_path: Path) -> dict[str, str]:
    """{utt_id: "tts"|"vc"} from the ASVspoof2019 LA train protocol."""
    return {u: ATTACK_ID_TO_TYPE[a] for u, a in load_asvspoof_2019_train_attack_id_map(protocol_path).items()}


def attack_type_for_file(
    wav_path: Path, source_dir: str, per_file_map: dict[str, str] | None
) -> str:
    """Resolves one file's attack type: per-file map first (keyed by filename
    stem; values may be types or attack IDs), then the directory-level default
    (e.g. MLAAD, accent-expansion TTS dirs), else "unknown" (In-the-Wild,
    CodecFake: no ground truth)."""
    if per_file_map is not None:
        v = per_file_map.get(Path(wav_path).stem)
        if v is not None:
            return ATTACK_ID_TO_TYPE.get(v, v)
    resolved_dir = str(Path(source_dir).resolve())
    return DIRECTORY_DEFAULT_ATTACK_TYPE.get(resolved_dir, "unknown")


def attack_id_for_file(wav_path: Path, per_file_map: dict[str, str] | None) -> str | None:
    """The file's ASVspoof attack ID if the per-file map carries IDs, else None."""
    if per_file_map is None:
        return None
    v = per_file_map.get(Path(wav_path).stem)
    return v if v in ATTACK_ID_TO_TYPE else None
