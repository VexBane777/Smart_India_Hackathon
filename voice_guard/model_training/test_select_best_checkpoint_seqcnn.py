"""select_best_checkpoint_seqcnn.py: the selection policy (no torch, no splits).

Covers `select_best` (argmin / earliest-within-tolerance / multi-objective floor /
channel-balanced objective), `_objective`, `attack_bacc_from_history` and
`_epoch_of`. The script itself still selects on the `select` split only; these
helpers are pure and split-agnostic.
"""
from select_best_checkpoint_seqcnn import (_epoch_of, _objective, attack_bacc_from_history,
                                           select_best)


def test_epoch_of_parses_names():
    assert _epoch_of("epoch_01.pt") == 1
    assert _epoch_of("epoch_12.pt") == 12
    assert _epoch_of("weird.pt") == -1


def test_select_best_default_is_raw_argmin():
    sweep = {"epoch_03.pt": {"pooled_eer": 0.31},
             "epoch_06.pt": {"pooled_eer": 0.30},
             "epoch_22.pt": {"pooled_eer": 0.32}}
    assert select_best(sweep, 0.0) == "epoch_06.pt"
    # default tolerance (omitted) is also raw argmin
    assert select_best(sweep) == "epoch_06.pt"


def test_select_best_stability_picks_earliest_within_tolerance():
    # argmin is the late epoch_20; within tolerance epoch_10 is earlier and allowed ->
    # stability selection should prefer the earlier epoch instead of the chance-dip.
    sweep = {"epoch_05.pt": {"pooled_eer": 0.3500},
             "epoch_10.pt": {"pooled_eer": 0.3170},
             "epoch_20.pt": {"pooled_eer": 0.3160}}
    assert select_best(sweep, 0.002) == "epoch_10.pt"


def test_select_best_too_small_tolerance_returns_argmin():
    sweep = {"epoch_06.pt": {"pooled_eer": 0.3160},
             "epoch_22.pt": {"pooled_eer": 0.3170}}
    assert select_best(sweep, 1e-9) == "epoch_06.pt"


# ---------------------------------------------------------------------------
# Post-v13 rework axes D1 (channel-balanced objective) and H (multi-objective
# floor): selection was chasing the two worst codecs and could pick an EER-min
# checkpoint with a weak attack head.
# ---------------------------------------------------------------------------
def test_channel_balanced_objective_debiases_the_worst_codec():
    # A wins the pooled argmin only because one bad codec drags everything;
    # B is uniformly better across channels.
    sweep = {"epoch_04.pt": {"pooled_eer": 0.30, "eer_whatsapp": 0.10, "eer_gsm_2g": 0.90},
             "epoch_09.pt": {"pooled_eer": 0.32, "eer_whatsapp": 0.28, "eer_gsm_2g": 0.36}}
    chans = ["whatsapp", "gsm_2g"]
    assert _objective(sweep["epoch_04.pt"], "channel_balanced", chans) == 0.5
    assert _objective(sweep["epoch_09.pt"], "channel_balanced", chans) == 0.32
    assert select_best(sweep, 0.0, "pooled", chans) == "epoch_04.pt"
    assert select_best(sweep, 0.0, "channel_balanced", chans) == "epoch_09.pt"


def test_channel_balanced_objective_skips_missing_and_nan_channels():
    import math

    row = {"pooled_eer": 0.3, "eer_a": 0.2, "eer_b": float("nan")}
    assert _objective(row, "channel_balanced", ["a", "b", "missing"]) == 0.2
    # no usable channel at all -> NaN (which select_best treats as +inf, never best)
    assert math.isnan(_objective({"pooled_eer": 0.3}, "channel_balanced", ["a"]))


def test_rows_without_a_usable_objective_are_never_selected():
    sweep = {"epoch_02.pt": {"pooled_eer": 0.9, "eer_a": float("nan")},   # inf objective
             "epoch_07.pt": {"pooled_eer": 0.5, "eer_a": 0.4}}
    assert select_best(sweep, 0.0, "channel_balanced", ["a"]) == "epoch_07.pt"


def test_min_attack_bacc_floor_excludes_weak_attack_heads():
    sweep = {"epoch_03.pt": {"pooled_eer": 0.300, "val_attack_bacc": 0.60},
             "epoch_08.pt": {"pooled_eer": 0.320, "val_attack_bacc": 0.90}}
    # without the floor the (better-EER, weak-head) epoch wins
    assert select_best(sweep, 0.0, "pooled", (), None) == "epoch_03.pt"
    # with it, only the strong-head epoch is eligible
    assert select_best(sweep, 0.0, "pooled", (), 0.8) == "epoch_08.pt"
    # a floor nothing clears is dropped rather than failing the run
    assert select_best(sweep, 0.0, "pooled", (), 0.99) == "epoch_03.pt"


def test_floor_and_stability_tolerance_compose():
    # epoch_02 has the best EER but a weak attack head; epoch_09/20 are both strong
    sweep = {"epoch_02.pt": {"pooled_eer": 0.310, "val_attack_bacc": 0.40},
             "epoch_09.pt": {"pooled_eer": 0.318, "val_attack_bacc": 0.90},
             "epoch_20.pt": {"pooled_eer": 0.317, "val_attack_bacc": 0.90}}
    # no floor: the argmin itself is inside the tolerance -> stability keeps it
    assert select_best(sweep, 0.005, "pooled", (), None) == "epoch_02.pt"
    # floor 0.8 removes epoch_02, then stability picks the earliest of the rest
    assert select_best(sweep, 0.005, "pooled", (), 0.8) == "epoch_09.pt"
    # a floor nothing clears is dropped instead of failing the selection
    assert select_best(sweep, 0.005, "pooled", (), 0.99) == "epoch_02.pt"


def test_attack_bacc_from_history_maps_epoch_names():
    history = [{"epoch": 1, "val_attack_bacc": 0.55},
               {"epoch": 30, "val_attack_bacc": 0.91},
               {"epoch": 0, "val_attack_bacc": float("nan")}]  # not a real epoch row
    assert attack_bacc_from_history(history) == {"epoch_01.pt": 0.55, "epoch_30.pt": 0.91}
