"""OC-Softmax fine-tune directly on the deployed v13 SeqTCN (Idea 6,
standalone -- see docs/superpowers/plans/2026-09-17-vaani-model-
architecture-phase0-spike.md Task 15). Does NOT depend on the AASIST spike
(Tasks 1-11): v13's pooled->frame-level embedding already exists, so this
tests Idea 6's core hypothesis (one-class objective vs. two-class BCE)
directly on the model running in production today.

`embed_v13` is a straight read of VoiceGuardSeqTCN.forward's own body
(model.py) up to and including trunk_out -- model.py is not modified, this
only calls its already-public submodules in the same order forward() does.

--freeze-backbone (default True) trains ONLY OCSoftmaxLoss's own w0 on top
of the frozen v13 embedding, isolating "does changing the objective alone
move the confound gate" from "does more training move it" -- the same
one-variable-at-a-time discipline that already found the 'clean'-recipe
regression (state.md, "Attempt 2 + follow-up ablations").

Usage:
    python finetune_oc_softmax_v13.py \
        --checkpoint runs/voice_guard_v13_selected/model.pt \
        --out docs/2026-09-18-oc-softmax-v13-result.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from oc_softmax import OCSoftmaxLoss, oc_softmax_score


def embed_v13(model, seq: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
    """(B, 184, 60) sequence + (B, 6) scalars -> (B, hidden) trunk_out --
    the shared embedding VoiceGuardSeqTCN.forward feeds to real_fake_head/
    attack_type_head, computed by calling the SAME public submodules in the
    SAME order forward() does. No no_grad() here: callers control whether
    gradients flow into the backbone via requires_grad, not via this
    function (needed for the --freeze-backbone=False escalation)."""
    h = model.blocks(model.stem(model.normalize_sequence(seq).transpose(1, 2)))
    mean = h.mean(dim=2)
    std = torch.sqrt(((h - mean.unsqueeze(2)) ** 2).mean(dim=2) + 1e-5)
    pooled = torch.cat([mean, std, h.amax(dim=2)], dim=1)
    return model.trunk(torch.cat([pooled, model.scalar_normalize(scalars)], dim=1))


def _oc_p_fake(model, oc_loss_fn: OCSoftmaxLoss, coll, idx: np.ndarray, device: str, batch: int = 2048) -> np.ndarray:
    """OC-Softmax fake-probability for windows idx of a CacheCollection,
    rescaled from oc_softmax_score's [-1, 1] cosine similarity to [0, 1]
    (higher = more fake) so it drops into evaluate.confound_table exactly
    like a BCE model's real_fake_head softmax output does."""
    out = []
    model.eval()
    with torch.no_grad():
        for _b, seq, scal in coll.iter_batches(idx, batch):
            emb = embed_v13(model, torch.from_numpy(seq).to(device), torch.from_numpy(scal).to(device))
            sim = oc_softmax_score(emb, oc_loss_fn.w0)
            out.append(((1.0 - sim) / 2.0).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="an existing v13 (seqtcn_v2) run's model.pt")
    ap.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True,
                    help="default True: train only OCSoftmaxLoss's w0 on the frozen v13 embedding, isolating "
                         "the objective-function question from a re-training question (see module docstring)")
    ap.add_argument("--out", type=Path, required=True, help="result markdown path (a sibling .json is also written)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--m-real", type=float, default=0.9)
    ap.add_argument("--m-fake", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--application", default="phone")
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from build_caches import default_cache_root
    from corpus import CORE_EVAL_SETS
    from eval_protocol import APPLICATIONS, phone_subset, resolve_channels
    from eval_stats import bootstrap_eer_ci, compute_eer_threshold
    from evaluate import confound_table, load_eval_collection
    from model import load_scoring_model

    if args.application not in APPLICATIONS:
        raise SystemExit(f"--application must be one of {APPLICATIONS}, got {args.application!r}")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model, arch = load_scoring_model(args.checkpoint)
    if arch != "seqtcn_v2":
        raise SystemExit(f"finetune_oc_softmax_v13.py targets the deployed v13 architecture (seqtcn_v2), "
                          f"got {arch!r} from {args.checkpoint}")
    model = model.to(device)
    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad_(False)

    embedding_dim = model.trunk[0].out_features
    oc_loss_fn = OCSoftmaxLoss(embedding_dim, m_real=args.m_real, m_fake=args.m_fake, alpha=args.alpha).to(device)

    channels = resolve_channels(args.channels, args.application, "eval")
    cache_root = args.cache_root or default_cache_root()
    print(f"Loading `select` split (channels={[c or 'none' for c in channels]})...")
    sel = load_eval_collection("select", channels, cache_root, CORE_EVAL_SETS, args.workers)
    phone_ch = phone_subset(channels)
    pooled_ch = phone_ch if phone_ch else ["none"]
    sel_mask = np.isin(sel.channel, pooled_ch)
    print(f"{sel.n} select-split windows ({int(sel_mask.sum())} phone-pooled) from {len(set(sel.file_id))} files")

    params = list(oc_loss_fn.parameters())
    if not args.freeze_backbone:
        params += [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)

    model.train(mode=not args.freeze_backbone)
    idx_all = np.arange(sel.n)
    history = []
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed + epoch)
        perm = rng.permutation(idx_all)
        total_loss, n_seen = 0.0, 0
        for b_idx, seq, scal in sel.iter_batches(perm, args.batch_size):
            seq_t = torch.from_numpy(seq).to(device)
            scal_t = torch.from_numpy(scal).to(device)
            y_t = torch.from_numpy(sel.label[b_idx]).to(device)
            opt.zero_grad()
            emb = embed_v13(model, seq_t, scal_t)
            loss = oc_loss_fn(emb, y_t)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(b_idx)
            n_seen += len(b_idx)
        avg_loss = total_loss / max(n_seen, 1)
        history.append({"epoch": epoch + 1, "train_loss": avg_loss})
        print(f"epoch {epoch + 1}/{args.epochs}  oc_softmax_loss={avg_loss:.4f}")

    model.eval()

    print("Scoring `select` for the operating threshold...")
    p_sel = _oc_p_fake(model, oc_loss_fn, sel, idx_all, device)
    _eer_sel, threshold = compute_eer_threshold(p_sel[sel_mask], sel.label[sel_mask])

    print("Loading and scoring `test` split...")
    test = load_eval_collection("test", channels, cache_root, None, args.workers)
    core = np.isin(test.source_set, CORE_EVAL_SETS)
    headline_mask = core & np.isin(test.channel, pooled_ch)
    p_test = _oc_p_fake(model, oc_loss_fn, test, np.arange(test.n), device)

    eer, _thr = compute_eer_threshold(p_test[headline_mask], test.label[headline_mask])
    ci = bootstrap_eer_ci(p_test[headline_mask], test.label[headline_mask], test.file_id[headline_mask],
                          n_bootstrap=args.n_bootstrap)
    conf = confound_table(p_test, test.label, test.scalars, test.pad_fraction, headline_mask, threshold,
                          test.file_id, n_boot=args.n_bootstrap)

    result = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "checkpoint": str(args.checkpoint),
        "arch": arch,
        "freeze_backbone": bool(args.freeze_backbone),
        "epochs": args.epochs,
        "oc_softmax_params": {"m_real": args.m_real, "m_fake": args.m_fake, "alpha": args.alpha},
        "channels": [c or "none" for c in channels],
        "pooled_channels": pooled_ch,
        "operating_threshold": float(threshold),
        "headline": {"eer": eer, "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "n_windows": int(headline_mask.sum()),
                     "n_files": int(len(np.unique(test.file_id[headline_mask])))},
        "confound": conf,
        "train_history": history,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=1, default=float))

    lines = [
        "# OC-Softmax on the deployed v13 SeqTCN (Idea 6, standalone) - result",
        "",
        f"Generated {result['generated']} by `finetune_oc_softmax_v13.py`. Checkpoint: `{args.checkpoint}`. "
        f"Backbone: {'frozen (w0 only trained)' if args.freeze_backbone else 'fine-tuned jointly with w0'}, "
        f"{args.epochs} epochs on the `select` split. m_real={args.m_real}, m_fake={args.m_fake}, alpha={args.alpha}.",
        "",
        "## Headline (test split, core sets, phone-pooled)",
        "",
        "| model | EER | 95% CI (file bootstrap) | confound gate |",
        "|---|---|---|---|",
        f"| v13 BCE (existing, `runs/eval_v13_test/report.md`) | 0.3149 | [0.3050, 0.3250] | "
        f"FAIL (9/14 rows) |",
        f"| v13 + OC-Softmax ({'frozen' if args.freeze_backbone else 'unfrozen'} backbone) | {eer:.4f} | "
        f"[{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}] | "
        f"{'PASS' if conf['passed'] else 'FAIL (' + ', '.join(conf['failed']) + ')'} |",
        "",
        "## Confound v2 rows (phone-pooled, core sets, test split)",
        "",
        "| class | feature | mean p low/high | gap | rho | rate low/high | rate ratio | pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in conf["rows"]:
        lines.append(
            f"| {r['class']} | {r['feature']} | {r.get('mean_p_low', float('nan')):.3f} / "
            f"{r.get('mean_p_high', float('nan')):.3f} | {r.get('abs_gap', float('nan')):.3f} | "
            f"{r['spearman_rho']:.3f} | {r.get('rate_low', float('nan')):.1%} / "
            f"{r.get('rate_high', float('nan')):.1%} | {r.get('rate_ratio_worse_better', float('nan')):.2f} | "
            f"{'yes' if r['passed'] else '**NO**'} |")
    lines += ["", "## Train loss by epoch", "", "| epoch | oc_softmax_loss |", "|---|---|"]
    lines += [f"| {h['epoch']} | {h['train_loss']:.4f} |" for h in history]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"headline EER={eer:.4f} [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]  "
          f"confound={'PASS' if conf['passed'] else 'FAIL: ' + ', '.join(conf['failed'])}")
    print(f"wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
