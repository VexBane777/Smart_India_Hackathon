"""Helper: append the rest of dedup.py (D2 near-dup dedup completion)."""
from pathlib import Path

P = Path("dedup.py")
text = P.read_text(encoding="utf-8")
if "find_cross_split_duplicates" in text:
    print("already complete")
    raise SystemExit(0)

APPEND = """
# Allowed group names in the manifest for this check.
_SPLIT_GROUPS = ("select", "test", "train")


def find_cross_split_duplicates(
    manifest_path,
    *,
    data_root=None,
    perceptual=False,
):
    \"\"\"Find near-duplicate audio crossing train/select/test split boundaries.

    Returns list of dicts:
        kind: "exact" | "perceptual"
        detail: md5 hex or fingerprint hex
        groups: list of (split, set_name, file_id, path)
    \"\"\"
    data_root = data_root or DATA_ROOT
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exact_index = defaultdict(list)
    perc_index = defaultdict(list)
    for split_name, split_groups in manifest.items():
        if split_name not in _SPLIT_GROUPS:
            continue
        for set_name, file_ids in split_groups.items():
            for fid in file_ids:
                p = data_root / fid
                if not p.is_file():
                    continue
                m = _md5_of_path(p)
                if m:
                    exact_index[m].append((split_name, set_name, fid, p))
                if perceptual:
                    f = _perceptual_fp(p)
                    if f:
                        perc_index[f].append((split_name, set_name, fid, p))
    out = []
    for md5, recs in exact_index.items():
        ss = {(r[0], r[1]) for r in recs}
        if len(ss) > 1:
            out.append({"kind": "exact", "detail": md5, "groups": recs})
    if perceptual:
        for fp, recs in perc_index.items():
            ss = {(r[0], r[1]) for r in recs}
            if len(ss) > 1:
                out.append({"kind": "perceptual", "detail": fp, "groups": recs})
    out.sort(key=lambda d: (0 if d["kind"] == "exact" else 1, -len(d["groups"])))
    return out


def _format_dup(d):
    kind = d["kind"].upper()
    lines = [f"[{kind}] {d['detail']}"]
    for split_name, set_name, fid, path in d["groups"]:
        try:
            rel = path.relative_to(DATA_ROOT)
        except ValueError:
            rel = path.name
        lines.append(f"  {split_name:6s} {set_name:22s} {fid:38s} {rel}")
    return "\\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="Near-duplicate dedup across train/select/test splits (axis D2).")
    ap.add_argument("--splits", type=Path, default=SPLIT_MANIFEST,
                    help="split manifest JSON (default: %(default)s)")
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT,
                    help="audio root (default: %(default)s)")
    ap.add_argument("--perceptual", action="store_true",
                    help="also flag perceptual near-dups")
    args = ap.parse_args(argv)
    if not args.splits.is_file():
        raise FileNotFoundError(f"split manifest not found: {args.splits}")
    dups = find_cross_split_duplicates(args.splits, data_root=args.data_root,
                                       perceptual=args.perceptual)
    if not dups:
        print("No cross-split duplicates found.")
        return 0
    print(f"Found {len(dups)} cross-split duplicate group(s):\\n")
    for d in dups:
        print(_format_dup(d))
        print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
"""

P.write_text(text + APPEND.lstrip("\\n"), encoding="utf-8")
print("done; total lines now:", len(P.read_text(encoding="utf-8").splitlines()))
