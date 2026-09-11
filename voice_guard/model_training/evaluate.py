"""
The VoiceGuard evaluation harness. It replaces every one-off eval script:
eval_held_out*.py, measure_confound*.py and eval_accent_cells.py were
retired 2026-09-11. Protocol: voice_guard/docs/EVAL-PROTOCOL.md.

What it does, for one or more models, on the committed held-out split
(eval_splits/held_out_split_v1.json) under the channel policy
(eval_protocol.py: no clean-only eval unless --application bank):

- Operating threshold: the phone-pooled EER threshold on the `select`
  split (core sets). Never fitted on the split being reported.
- Headline: EER on `test`, core sets, pooled over phone channels, with a
  95% file-level bootstrap CI (every window and channel of a file resamples
  together); FPR/FNR at the select threshold and at the app default 0.60.
- Per channel (none = reference row), seen vs unseen channel groups,
  padded (short-clip) windows only, core+MLAAD, per eval set, accent cells.
- Confound v2, real AND fake side, for pauseRatio, energyVariance,
  zcrVariance, jitter, shimmer, HNR and pad_fraction: half-means, absolute
  gap, high/low ratio, Spearman rho, and FPR (reals) / FNR (fakes) per half
  at the operating threshold. Gates, fixed before any model was scored on
  the new protocol, pooled over phone channels:
    |rho| <= 0.10; and for the worse/better half FPR (reals) / FNR (fakes):
    a row fails only if the ratio > 1.25 AND the absolute difference > 1
    point AND the difference is significant (file-level bootstrap CI,
    Bonferroni-corrected over all rows). The significance clause was added
    when a synthetic model with no confound at all failed the bare ratio
    gate by chance (test_evaluate.py); it was added before any real model
    was scored.
- Attack-type head: MLAAD (all TTS) share predicted "tts"; optionally
  (--attack-val-run) in-distribution and leave-attack-out (A11/A18)
  balanced accuracy on a training run's val fakes; plus coverage and
  accuracy of the sub-label the UI actually shows (p_fake >= 0.60 and
  attack confidence >= 0.70). Gates: leave-out balanced accuracy >= 0.70,
  MLAAD tts share >= 0.70.
- Writes report.json and report.md into --out.

Usage (the v12 comparison):
    python evaluate.py --split test --out runs/eval_v12_test \
        --model v9=runs/voice_guard_v9_noisefix_final/model.pt \
        --model v11=runs/voice_guard_v11_seqcnn_selected/model.pt \
        --model v12=runs/voice_guard_v12_selected/model.pt \
        --attack-val-run runs/voice_guard_v12 --candidate v12 --reference v11
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from eval_protocol import (
    APPLICATIONS,
    PHONE_CHANNELS,
    TRAIN_PHONE_CHANNELS,
    UNSEEN_CHANNELS,
    channel_name,
    phone_subset,
    resolve_channels,
)
from eval_stats import (
    auc_separability,
    balanced_accuracy,
    bootstrap_eer_ci,
    cis_overlap,
    compute_eer_threshold,
    confusion_matrix,
    rates_at_threshold,
    spearman_rho,
)

APP_THRESHOLD = 0.60  # settings_provider.dart default alert threshold (applied to an EMA in the app)
UI_ATTACK_CONF = 0.70  # call_screen.dart attack-type sub-label confidence gate
RHO_GATE = 0.10
RATE_RATIO_GATE = 1.25
RATE_ABS_FLOOR = 0.01
ATTACK_LEAVE_OUT_BACC_GATE = 0.70
ATTACK_MLAAD_TTS_GATE = 0.70
CONFOUND_FEATURES = (("pauseRatio", 0), ("energyVariance", 1), ("zcrVariance", 2),
                     ("jitter", 3), ("shimmer", 4), ("hnr_db", 5), ("pad_fraction", None))
ACCENT_PAIRS = {"en_native": ("accent_real_en_native", "accent_fake_en_native"),
                "hi_native": ("accent_real_hi_native", "accent_fake_hi_native")}
HYBRID_CODES = (13, 14, 15)


def load_eval_collection(split: str, channels, cache_root: Path, set_names=None, workers: int = 10):
    """CacheCollection over the committed manifest's `split`, building any
    missing cache unit first."""
    from build_attack_type_maps import build_attack_type_maps
    from build_caches import eval_units
    from feature_cache import CacheCollection, build_unit

    units = eval_units(split, channels, set_names, build_attack_type_maps())
    return CacheCollection([build_unit(n, s, c, cache_root, workers=workers) for n, s, c in units])


def _eer_block(p, y, fid, n_boot: int) -> dict:
    out = {"n_windows": int(len(y)), "n_files": int(len(np.unique(fid))) if len(fid) else 0}
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {**out, "eer": float("nan")}
    eer, thr = compute_eer_threshold(p, y)
    out.update(eer=eer, eer_threshold=thr)
    if n_boot:
        ci = bootstrap_eer_ci(p, y, fid, n_bootstrap=n_boot)
        out.update(ci_lo=ci["ci_lo"], ci_hi=ci["ci_hi"], ci_std=ci["std"])
    return out


def _rate_diff_ci(flag, hi, fid, n_boot: int, alpha: float, seed: int = 0) -> tuple[float, float]:
    """File-level bootstrap CI of rate(high half) - rate(low half)."""
    _u, inv = np.unique(fid, return_inverse=True)
    F = int(inv.max()) + 1
    n_hi = np.bincount(inv, weights=hi.astype(float), minlength=F)
    k_hi = np.bincount(inv, weights=(flag & hi).astype(float), minlength=F)
    n_lo = np.bincount(inv, weights=(~hi).astype(float), minlength=F)
    k_lo = np.bincount(inv, weights=(flag & ~hi).astype(float), minlength=F)
    rng = np.random.default_rng(seed)
    diffs = []
    for start in range(0, n_boot, 100):
        S = rng.integers(0, F, size=(min(100, n_boot - start), F))
        diffs.append(k_hi[S].sum(1) / np.maximum(n_hi[S].sum(1), 1) - k_lo[S].sum(1) / np.maximum(n_lo[S].sum(1), 1))
    d = np.concatenate(diffs)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def confound_table(p, y, scalars, pad, mask, threshold, fid=None, n_boot: int = 1000) -> dict:
    """Median-split confound analysis per class and feature, with gates.
    A rate row fails only if ALL of: worse/better ratio > RATE_RATIO_GATE,
    |difference| > RATE_ABS_FLOOR, and the file-level bootstrap CI of the
    difference excludes 0 at a Bonferroni-corrected level (alpha 0.05 split
    over every rate row). Sampling noise alone must not fail a clean model:
    on ~1k files a 7%-vs-9% split is within noise."""
    fid = np.arange(len(p)) if fid is None else np.asarray(fid)
    n_rows = 2 * len(CONFOUND_FEATURES)
    alpha = 0.05 / n_rows
    rows = []
    for cls, cname, rate_name in ((0, "real", "fpr"), (1, "fake", "fnr")):
        m = mask & (y == cls)
        pp = p[m]
        if len(pp) < 20:
            continue
        flag = (pp >= threshold) if cls == 0 else (pp < threshold)
        for feat, col in CONFOUND_FEATURES:
            x = (scalars[m, col] if col is not None else pad[m]).astype(np.float64)
            med = float(np.median(x))
            hi = x > med
            lo = ~hi
            rho = spearman_rho(x, pp)
            row = {"class": cname, "feature": feat, "median": med, "n_low": int(lo.sum()),
                   "n_high": int(hi.sum()), "spearman_rho": rho, "rate": rate_name}
            if hi.any() and lo.any():
                ml, mh = float(pp[lo].mean()), float(pp[hi].mean())
                rl, rh = float(flag[lo].mean()), float(flag[hi].mean())
                worse, better = max(rl, rh), min(rl, rh)
                rr = worse / better if better > 0 else (math.inf if worse > 0 else 1.0)
                ci = _rate_diff_ci(flag, hi, fid[m], n_boot, alpha) if n_boot else (-math.inf, math.inf)
                significant = ci[0] > 0 or ci[1] < 0
                row.update(mean_p_low=ml, mean_p_high=mh, abs_gap=abs(mh - ml),
                           mean_p_ratio=(max(ml, mh) / min(ml, mh)) if min(ml, mh) > 0 else math.inf,
                           rate_low=rl, rate_high=rh, rate_ratio_worse_better=rr,
                           rate_diff_ci=list(ci), rate_diff_significant=bool(significant))
                gate_rate = rr <= RATE_RATIO_GATE or abs(rh - rl) <= RATE_ABS_FLOOR or not significant
            else:
                gate_rate = True
                row["note"] = "degenerate median split (feature constant above/below median)"
            gate_rho = (not np.isfinite(rho)) or abs(rho) <= RHO_GATE
            row.update(gate_rho=bool(gate_rho), gate_rate=bool(gate_rate), passed=bool(gate_rho and gate_rate))
            rows.append(row)
    return {"threshold": float(threshold), "bonferroni_alpha": alpha, "rows": rows,
            "passed": bool(all(r["passed"] for r in rows)) if rows else None,
            "failed": [f"{r['class']}:{r['feature']}" for r in rows if not r["passed"]]}


def _attack_block(p, att, y_type, mask_extra=None) -> dict:
    """Attack-type metrics on fakes with known type (y_type 0=tts, 1=vc)."""
    if att is None or len(y_type) == 0:
        return {"n": int(len(y_type))}
    pred = att.argmax(1)
    conf = att.max(1)
    shown = (p >= APP_THRESHOLD) & (conf >= UI_ATTACK_CONF)
    return {
        "n": int(len(y_type)), "accuracy": float((pred == y_type).mean()),
        "balanced_accuracy": balanced_accuracy(y_type, pred), "confusion_tts_vc": confusion_matrix(y_type, pred),
        "ui_coverage": float(shown.mean()),
        "ui_accuracy_when_shown": float((pred[shown] == y_type[shown]).mean()) if shown.any() else float("nan"),
    }


def evaluate_model(p, att, coll, threshold, channels, n_boot) -> dict:
    y, fid, ch, sset = coll.label, coll.file_id, coll.channel, coll.source_set
    from corpus import CORE_EVAL_SETS

    core = np.isin(sset, CORE_EVAL_SETS)
    phone_ch = phone_subset(channels)
    pooled_ch = phone_ch if phone_ch else ["none"]  # bank application: clean-only allowed
    pooled = np.isin(ch, pooled_ch)
    m = core & pooled
    res = {"operating_threshold": float(threshold), "pooled_channels": pooled_ch}
    res["headline"] = {**_eer_block(p[m], y[m], fid[m], n_boot),
                       "at_select_threshold": rates_at_threshold(p[m], y[m], threshold),
                       "at_app_threshold": rates_at_threshold(p[m], y[m], APP_THRESHOLD)}
    res["per_channel"] = {}
    for c in channels:
        mc = core & (ch == channel_name(c))
        res["per_channel"][channel_name(c)] = {**_eer_block(p[mc], y[mc], fid[mc], n_boot // 5),
                                              "at_select_threshold": rates_at_threshold(p[mc], y[mc], threshold)}
    groups = {"seen_phone": [c for c in phone_ch if c in TRAIN_PHONE_CHANNELS],
              "unseen_phone": [c for c in phone_ch if c in UNSEEN_CHANNELS], "none_reference": ["none"]}
    res["channel_groups"] = {g: {"channels": cs, **_eer_block(p[core & np.isin(ch, cs)], y[core & np.isin(ch, cs)],
                                                              fid[core & np.isin(ch, cs)], 0)}
                             for g, cs in groups.items() if cs}
    padded = m & (coll.pad_fraction > 0)
    res["padded_windows_only"] = {**_eer_block(p[padded], y[padded], fid[padded], 0),
                                  "share_of_pooled_windows": float(padded.sum() / max(m.sum(), 1))}
    unpadded = m & (coll.pad_fraction == 0)
    res["unpadded_windows_only"] = _eer_block(p[unpadded], y[unpadded], fid[unpadded], 0)
    res["pad_diagnostics"] = {
        "padded_share_real": float((coll.pad_fraction[m & (y == 0)] > 0).mean()) if (m & (y == 0)).any() else None,
        "padded_share_fake": float((coll.pad_fraction[m & (y == 1)] > 0).mean()) if (m & (y == 1)).any() else None,
        "pad_fraction_label_auc": auc_separability(coll.pad_fraction[m], y[m]),
    }
    if "mlaad_fake" in set(sset):
        mm = (core | (sset == "mlaad_fake")) & pooled
        res["core_plus_mlaad"] = _eer_block(p[mm], y[mm], fid[mm], 0)
    res["per_set"] = {}
    for s in sorted(set(sset)):
        ms = (sset == s) & pooled
        lab = int(y[ms][0]) if ms.any() else None
        r = rates_at_threshold(p[ms], y[ms], threshold)
        r_app = rates_at_threshold(p[ms], y[ms], APP_THRESHOLD)
        key = "fpr" if lab == 0 else "fnr"
        res["per_set"][s] = {"label": lab, "n_windows": int(ms.sum()), "n_files": int(len(np.unique(fid[ms]))),
                             f"{key}_at_select_threshold": r[key], f"{key}_at_app_threshold": r_app[key],
                             "mean_p_fake": float(p[ms].mean()) if ms.any() else float("nan")}
    res["accent_cells"] = {}
    for cell, (rs, fs) in ACCENT_PAIRS.items():
        ma = np.isin(sset, [rs, fs]) & pooled
        if ma.any() and len(np.unique(y[ma])) == 2:
            res["accent_cells"][cell] = _eer_block(p[ma], y[ma], fid[ma], 0)
    res["confound"] = {"phone_pooled": confound_table(p, y, coll.scalars, coll.pad_fraction, m, threshold, fid),
                       "none_reference": confound_table(p, y, coll.scalars, coll.pad_fraction,
                                                        core & (ch == "none"), threshold, fid)}
    if att is not None and "mlaad_fake" in set(sset):
        ml = (sset == "mlaad_fake") & pooled
        pred_tts = att[ml].argmax(1) == 0
        shown = (p[ml] >= APP_THRESHOLD) & (att[ml].max(1) >= UI_ATTACK_CONF)
        res["attack_mlaad"] = {
            "n": int(ml.sum()), "tts_share": float(pred_tts.mean()),
            "ui_coverage": float(shown.mean()),
            "ui_tts_share_when_shown": float(pred_tts[shown].mean()) if shown.any() else float("nan"),
            "per_channel_tts_share": {c: float((att[(sset == "mlaad_fake") & (ch == c)].argmax(1) == 0).mean())
                                      for c in pooled_ch if ((sset == "mlaad_fake") & (ch == c)).any()},
        }
    return res


def attack_val_eval(models_scored: dict, coll, leave_codes, phone_ch) -> dict:
    """Attack-type head on a training run's val fakes with known type."""
    out = {}
    y_type = coll.attack_type
    aid = coll.attack_id
    groups = {"in_distribution": ~np.isin(aid, leave_codes), "leave_attack_out": np.isin(aid, leave_codes),
              "hybrids_A13_A15": np.isin(aid, HYBRID_CODES)}
    for name, (p, att) in models_scored.items():
        if att is None:
            continue
        r = {}
        for g, gm in groups.items():
            r[g] = _attack_block(p[gm], att[gm], y_type[gm])
            gp = gm & np.isin(coll.channel, phone_ch)
            r[g]["phone_only"] = _attack_block(p[gp], att[gp], y_type[gp])
        r["leave_out_per_attack_accuracy"] = {
            f"A{c:02d}": float((att[aid == c].argmax(1) == y_type[aid == c]).mean())
            for c in leave_codes if (aid == c).any()}
        out[name] = r
    return out


def _fmt(x, pct=False, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a" if x is None or np.isnan(x) else "inf"
    return f"{100 * x:.1f}%" if pct else f"{x:.{nd}f}"


def write_markdown(report: dict, path: Path) -> None:
    L = [f"# VoiceGuard evaluation report: split `{report['split']}`", "",
         f"Generated {report['generated']} by `evaluate.py`. Protocol: `voice_guard/docs/EVAL-PROTOCOL.md`. "
         f"Channels: {', '.join(report['channels'])}. Headline pooled over: {', '.join(report['pooled_channels'])}.",
         f"Files: {report['n_files']} ({report['n_windows']} windows across channels). "
         "Operating threshold per model = phone-pooled EER threshold on the `select` split.", ""]
    names = list(report["models"])
    M = report["models"]
    L += ["## Headline (core sets, phone-pooled)", "",
          "| model | EER | 95% CI (file bootstrap) | threshold | FPR @thr | FNR @thr | FPR @0.60 | FNR @0.60 |",
          "|---|---|---|---|---|---|---|---|"]
    for n in names:
        h = M[n]["headline"]
        L.append(f"| {n} | {_fmt(h['eer'])} | [{_fmt(h.get('ci_lo'))}, {_fmt(h.get('ci_hi'))}] | "
                 f"{_fmt(M[n]['operating_threshold'], nd=3)} | {_fmt(h['at_select_threshold']['fpr'], True)} | "
                 f"{_fmt(h['at_select_threshold']['fnr'], True)} | {_fmt(h['at_app_threshold']['fpr'], True)} | "
                 f"{_fmt(h['at_app_threshold']['fnr'], True)} |")
    L += ["", "## EER per channel (core sets; `none` is a reference row, never a result on its own)", "",
          "| channel | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for c in report["channels"]:
        L.append(f"| {c} | " + " | ".join(_fmt(M[n]["per_channel"][c]["eer"]) for n in names) + " |")
    for g in ("seen_phone", "unseen_phone", "none_reference"):
        if g in M[names[0]]["channel_groups"]:
            L.append(f"| **{g}** | " + " | ".join(_fmt(M[n]["channel_groups"][g]["eer"]) for n in names) + " |")
    L.append("| **padded windows only** | " + " | ".join(_fmt(M[n]["padded_windows_only"]["eer"]) for n in names) + " |")
    L.append("| **unpadded windows only** | " + " | ".join(_fmt(M[n]["unpadded_windows_only"]["eer"]) for n in names) + " |")
    if "core_plus_mlaad" in M[names[0]]:
        L.append("| **core + MLAAD (unseen TTS)** | " + " | ".join(_fmt(M[n]["core_plus_mlaad"]["eer"]) for n in names) + " |")
    for cell in ACCENT_PAIRS:
        if cell in M[names[0]]["accent_cells"]:
            L.append(f"| **accent {cell}** | " + " | ".join(_fmt(M[n]["accent_cells"].get(cell, {}).get("eer")) for n in names) + " |")
    L += ["", "## Per eval set, phone-pooled (reals: FPR, fakes: FNR, at each model's select threshold)", "",
          "| set | windows | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for s, v in M[names[0]]["per_set"].items():
        key = "fpr_at_select_threshold" if v["label"] == 0 else "fnr_at_select_threshold"
        L.append(f"| {s} ({'FPR' if v['label'] == 0 else 'FNR'}) | {v['n_windows']} | "
                 + " | ".join(_fmt(M[n]["per_set"][s][key], True) for n in names) + " |")
    L += ["", "## Confound v2 (phone-pooled, core sets)", "",
          f"Gates: |rho| <= {RHO_GATE}; a rate row fails only if worse/better half rate ratio > {RATE_RATIO_GATE} "
          f"AND |diff| > {RATE_ABS_FLOOR:.0%} AND the file-level bootstrap CI of the difference excludes 0 "
          f"(Bonferroni alpha 0.05 over {2 * len(CONFOUND_FEATURES)} rows). "
          "Rate = FPR for reals, FNR for fakes, at the select threshold.", ""]
    for n in names:
        cf = M[n]["confound"]["phone_pooled"]
        L += [f"### {n}: {'PASS' if cf['passed'] else 'FAIL ' + ', '.join(cf['failed'])}", "",
              "| class | feature | mean p low/high | gap | ratio | rho | rate low/high | rate ratio | pass |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in cf["rows"]:
            L.append(f"| {r['class']} | {r['feature']} | {_fmt(r.get('mean_p_low'), nd=3)} / {_fmt(r.get('mean_p_high'), nd=3)} | "
                     f"{_fmt(r.get('abs_gap'), nd=3)} | {_fmt(r.get('mean_p_ratio'), nd=2)} | {_fmt(r['spearman_rho'], nd=3)} | "
                     f"{_fmt(r.get('rate_low'), True)} / {_fmt(r.get('rate_high'), True)} | "
                     f"{_fmt(r.get('rate_ratio_worse_better'), nd=2)} | {'yes' if r['passed'] else '**NO**'} |")
        L.append("")
    pd_rows = [(n, M[n]["pad_diagnostics"]) for n in names[:1]]
    for _n, d in pd_rows:
        L += [f"Pad diagnostics (eval data, same for every model): padded share real "
              f"{_fmt(d['padded_share_real'], True)}, fake {_fmt(d['padded_share_fake'], True)}, "
              f"pad_fraction label AUC {_fmt(d['pad_fraction_label_auc'], nd=3)}.", ""]
    L += ["## Attack-type head", ""]
    for n in names:
        a = M[n].get("attack_mlaad")
        if a:
            L.append(f"- **{n}** MLAAD (all TTS, unseen systems): predicted tts {_fmt(a['tts_share'], True)}; "
                     f"UI sub-label shown on {_fmt(a['ui_coverage'], True)} of windows, tts when shown "
                     f"{_fmt(a['ui_tts_share_when_shown'], True)}.")
    av = report.get("attack_val")
    if av:
        L += ["", f"Training-run val fakes (`{av['run']}`, leave-out {av['leave_out']}). {av['caveat']}", "",
              "| model | group | n | balanced acc | acc | UI coverage | UI acc when shown |", "|---|---|---|---|---|---|---|"]
        for n, r in av["models"].items():
            for g in ("in_distribution", "leave_attack_out", "hybrids_A13_A15"):
                b = r[g]
                L.append(f"| {n} | {g} | {b['n']} | {_fmt(b.get('balanced_accuracy'), nd=3)} | {_fmt(b.get('accuracy'), nd=3)} | "
                         f"{_fmt(b.get('ui_coverage'), True)} | {_fmt(b.get('ui_accuracy_when_shown'), True)} |")
    if report.get("fp16_validation"):
        L += ["", "## fp16 feature-cache validation", "", "```", json.dumps(report["fp16_validation"], indent=1), "```"]
    d = report.get("decision")
    if d:
        L += ["", "## Gates and deploy decision", ""]
        for k, v in d.items():
            L.append(f"- **{k}**: {v}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="NAME=path/to/model.pt (repeatable)")
    ap.add_argument("--split", default="test", choices=["test", "select"])
    ap.add_argument("--channels", nargs="*", default=None, help="default: none + all phone channels")
    ap.add_argument("--application", default="phone", choices=APPLICATIONS)
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--attack-val-run", type=Path, default=None, help="training run dir (train_config.json)")
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--fp16-report", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    import torch

    from attack_labels import attack_id_to_int
    from build_caches import default_cache_root
    from corpus import CORE_EVAL_SETS
    from dataset import stable_unit
    from feature_cache import CacheCollection
    from model import load_scoring_model
    from train_seq_cnn import score

    channels = resolve_channels(args.channels, args.application, "eval")
    phone_ch = phone_subset(channels)
    pooled_ch = phone_ch if phone_ch else ["none"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    cache_root = args.cache_root or default_cache_root()
    models = dict(m.split("=", 1) for m in args.model)

    sel = load_eval_collection("select", channels, cache_root, CORE_EVAL_SETS, args.workers)
    ev = sel if args.split == "select" else load_eval_collection(args.split, channels, cache_root, None, args.workers)
    sel_m = np.isin(sel.channel, pooled_ch)
    report = {"split": args.split, "generated": time.strftime("%Y-%m-%d %H:%M"), "application": args.application,
              "channels": [channel_name(c) for c in channels], "pooled_channels": pooled_ch,
              "n_windows": int(ev.n), "n_files": int(len(np.unique(ev.file_id))),
              "feature_version": ev.manifests[0]["feature_version"], "models": {}, "model_paths": models}
    if args.split == "select":
        report["warning"] = "threshold fitted on the split being reported (select); diagnostic only"
    loaded = {}
    for name, path in models.items():
        model, arch = load_scoring_model(path)
        loaded[name] = model.to(device)
        p_sel, _ = score(loaded[name], sel, np.arange(sel.n), device)
        _eer, thr = compute_eer_threshold(p_sel[sel_m], sel.label[sel_m])
        p, att = score(loaded[name], ev, np.arange(ev.n), device)
        report["models"][name] = {"arch": arch, **evaluate_model(p, att, ev, thr, channels, args.n_bootstrap)}
        h = report["models"][name]["headline"]
        print(f"{name} ({arch}): headline EER={h['eer']:.4f} [{h.get('ci_lo', float('nan')):.4f}, "
              f"{h.get('ci_hi', float('nan')):.4f}] thr={thr:.3f} confound="
              f"{'PASS' if report['models'][name]['confound']['phone_pooled']['passed'] else 'FAIL'}", flush=True)

    if args.attack_val_run:
        cfg = json.loads((args.attack_val_run / "train_config.json").read_text())
        tc = CacheCollection([Path(d) for d in cfg["unit_dirs"]])
        is_val = np.array([stable_unit(f, cfg["seed"], "val") < cfg["val_fraction"] for f in tc.file_id])
        idx = np.flatnonzero(is_val & (tc.label == 1) & (tc.attack_type >= 0))
        sub = _Subset(tc, idx)
        scored = {n: score(m, tc, idx, device) for n, m in loaded.items()}
        codes = [attack_id_to_int(a) for a in cfg["leave_out_attacks"]]
        report["attack_val"] = {
            "run": str(args.attack_val_run), "leave_out": cfg["leave_out_attacks"], "n_windows": int(len(idx)),
            "caveat": ("Only models trained with this run's val split held out are out-of-sample here; "
                       "v9/v11 trained on (most of) these files, and v11's attack head saw A11/A18 labels."),
            "models": attack_val_eval(scored, sub, codes, pooled_ch)}

    if args.fp16_report and args.fp16_report.exists():
        report["fp16_validation"] = json.loads(args.fp16_report.read_text())

    if args.candidate and args.reference:
        c, r = report["models"][args.candidate], report["models"][args.reference]
        conf = c["confound"]["phone_pooled"]
        better = c["headline"]["eer"] < r["headline"]["eer"]
        overlap = cis_overlap({"ci_lo": c["headline"]["ci_lo"], "ci_hi": c["headline"]["ci_hi"]},
                              {"ci_lo": r["headline"]["ci_lo"], "ci_hi": r["headline"]["ci_hi"]})
        dec = {
            "confound gates (candidate)": "PASS" if conf["passed"] else f"FAIL: {', '.join(conf['failed'])}",
            "beats reference on test headline EER": f"{better} ({c['headline']['eer']:.4f} vs {r['headline']['eer']:.4f}; "
                                                   f"CIs {'overlap' if overlap else 'do not overlap'})",
        }
        am = c.get("attack_mlaad")
        lo = report.get("attack_val", {}).get("models", {}).get(args.candidate, {}).get("leave_attack_out", {})
        if am is not None:
            a_ok = am["tts_share"] >= ATTACK_MLAAD_TTS_GATE and lo.get("balanced_accuracy", 0.0) >= ATTACK_LEAVE_OUT_BACC_GATE
            dec["attack-type head gates"] = (
                f"{'PASS' if a_ok else 'FAIL'} (leave-out balanced acc {_fmt(lo.get('balanced_accuracy'), nd=3)} "
                f"vs >= {ATTACK_LEAVE_OUT_BACC_GATE}; MLAAD tts share {_fmt(am['tts_share'], True)} vs >= "
                f"{ATTACK_MLAAD_TTS_GATE:.0%})")
            dec["attack_head_ok"] = bool(a_ok)
        dec["deploy"] = bool(conf["passed"] and better)
        report["decision"] = dec

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    write_markdown(report, args.out / "report.md")
    print(f"wrote {args.out / 'report.json'} and report.md")


class _Subset:
    """Row subset of a CacheCollection's metadata (for attack_val_eval)."""

    def __init__(self, coll, idx):
        for k in ("label", "attack_type", "attack_id", "channel", "file_id"):
            setattr(self, k, getattr(coll, k)[idx])


if __name__ == "__main__":
    main()
