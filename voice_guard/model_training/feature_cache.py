"""
On-disk feature cache: the fix for v11's OOM (`to_arrays` held every window
in RAM, which forced `--channel none`).

A cache *unit* is (a list of FileSpecs, one channel recipe, a base seed).
Its directory holds:
  seq.npy       float16 (N, 184, 60)   memory-mapped on read
  scalars.npy   float32 (N, 6)
  meta.npz      file_id, label, attack_type, attack_id, window_index,
                pad_fraction, duration, channel, source_set (per window)
  manifest.json feature-version hash + components, unit identity, yield report

Build: workers each process a shard of files and write shard files, and
the parent concatenates shards into the final memmap one at a time. Peak
RAM is about one shard per worker. Finished shards survive a crash, so a
re-run resumes where it stopped.

**Staleness is loud, never silent.** The manifest stores a feature-version
hash over everything that determines window contents: the code (AST,
docstrings stripped) of features.py, dataset.py and TeleChannel's
pipeline/stages, channels.yaml's parsed content, the
numpy/soundfile/librosa versions and the ffmpeg build. Opening a unit whose
hash differs from the current code raises StaleCacheError. Building over
one raises too, unless rebuild=True.

float16 storage: LFCC magnitudes reach a few hundred for c0, so fp16 keeps
~3 significant digits. The effect on model outputs is measured, not
assumed: `validate_fp16.py` reports the max |delta prob| for v11 between
fp32 and fp16 inputs.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dataset import (
    STATUS_OK,
    VAANI_ROOT,
    FileSpec,
    process_file,
    report_yield,
    stable_seed,
)
from eval_protocol import channel_name
from features import N_LFCC

CACHE_FORMAT_VERSION = 2  # v1 hashed unit contents without the per-file pad policy;
# convert_prob/keep_short_prob change which windows a file yields but were
# not in unit_key, so a pad-policy change silently reused stale units. v2
# hashes (convert_prob, keep_short_prob) per file. Bumped 2026-09-12 when
# the training pad-policy pre-pass was parallelized (corpus.py).
N_FRAMES = 184
N_SCALARS = 6
SEQ_DTYPE = np.float16
MODEL_TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_ROOT = MODEL_TRAINING_DIR / "cache"
META_FIELDS = ("file_id", "label", "attack_type", "attack_id", "window_index",
               "pad_fraction", "duration", "channel", "source_set")


class StaleCacheError(RuntimeError):
    """A cache built by different feature/windowing/channel code."""


def _code_fingerprint(path: Path) -> bytes:
    """What actually determines behavior, not formatting: Python files hash
    as their AST with docstrings stripped (comments never reach the AST);
    YAML hashes as its parsed content. So doc/comment edits and CRLF vs LF
    checkouts don't invalidate hours of cache, but any code change does."""
    if path.suffix == ".py":
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body
                    and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        return ast.dump(tree, annotate_fields=False, include_attributes=False).encode()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        return json.dumps(yaml.safe_load(path.read_text(encoding="utf-8")), sort_keys=True).encode()
    return path.read_bytes().replace(b"\r\n", b"\n")


def _ffmpeg_version() -> str:
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=30)
        return out.stdout.splitlines()[0].strip() if out.stdout else "unknown"
    except Exception:
        return "missing"


_FEATURE_VERSION: dict | None = None  # legacy single-version cache, kept for tests
_VERSION_CACHE: dict[str, dict] = {}
_ALL_KEY = "<all-recipes>"
_NONE_KEY = "none"



def _json_bytes(obj) -> bytes:
    """Canonical bytes for a parsed config subtree (sorted keys, compact)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()


def _channels_config() -> dict:
    import yaml

    return yaml.safe_load((VAANI_ROOT / "telechannel" / "configs" / "channels.yaml").read_text(encoding="utf-8"))


def _recipe_yaml_scope(recipe: str | None) -> dict | None:
    """The parts of channels.yaml that process_clip actually consults for ONE
    recipe: its own `recipes.<recipe>` block, the `rooms` entry(ies) its rir
    references, plus shared sample_rate/version. `None` (undegraded) runs no
    channel stages, so it returns None (no yaml consumed). Sorted-key JSON, so
    a recipe edit that only reorders keys does not bump the hash."""
    if recipe is None:
        return None
    cfg = _channels_config()
    if recipe not in cfg.get("recipes", {}):
        raise KeyError(f"unknown recipe {recipe!r}")
    r = cfg["recipes"][recipe]
    scope = {"sample_rate": cfg.get("sample_rate"), "version": cfg.get("version"),
             "recipe": {recipe: r}}
    room = (r.get("rir") or {}).get("room")
    if room in cfg.get("rooms", {}):
        scope["rooms"] = {room: cfg["rooms"][room]}
    return scope


def _version_common_components() -> dict:
    """Code + runtime components shared by every recipe scope. YAML is added
    per-scope: the whole file for the all-recipes version, just the recipe's
    own subtree for a scoped one."""
    import librosa
    import soundfile

    tc = VAANI_ROOT / "telechannel"
    files = [MODEL_TRAINING_DIR / "features.py", MODEL_TRAINING_DIR / "dataset.py",
             tc / "pipeline.py"] + sorted((tc / "stages").glob("*.py"))
    comps = {
        "cache_format": CACHE_FORMAT_VERSION,
        "numpy": np.__version__, "soundfile": soundfile.__version__, "librosa": librosa.__version__,
        "ffmpeg": _ffmpeg_version(),
    }
    for f in files:
        rel = f.relative_to(VAANI_ROOT.parent).as_posix() if VAANI_ROOT.parent in f.parents else f.name
        comps[rel] = hashlib.sha256(_code_fingerprint(f)).hexdigest()[:16]
    return comps

def feature_version(recipe: str | None = None, all_recipes: bool = True) -> dict:
    """Feature-version dict for a channel recipe's scope.

    - `feature_version()` (no args, the default) hashes the UNION over every
      recipe: whole channels.yaml + all code + the runtime. "Did anything
      pipeline-shaped change". This is what tests probe and is the pre-v13
      global-hash behavior.
    - `feature_version(recipe, all_recipes=False)` hashes only that recipe's
      scope: features.py + dataset.py + pipeline.py + stages + lib versions +
      the recipe's OWN yaml block and the rooms entry(ies) it references.
      **A change to one recipe must not invalidate every other unit** — that
      false positive is a whole-corpus rebuild (98 eval + 32 train units,
      hours) and is exactly the v13 post-v12 plan step 3. Docstring/comment
      changes still hash to nothing (AST, docstrings stripped).

    `recipe=None` (undegraded) runs no channel stages, so it hashes only the
    code/runtime (no channel yaml).
    """
    key = _ALL_KEY if all_recipes else (_NONE_KEY if recipe is None else str(recipe))
    cached = _VERSION_CACHE.get(key)
    if cached is not None:
        return cached
    comps = _version_common_components()
    if all_recipes:
        comps["channels.yaml"] = hashlib.sha256(_json_bytes(_channels_config())).hexdigest()[:16]
    else:
        scope = _recipe_yaml_scope(recipe)
        # undegraded (None) consumes no channel yaml: add no channels.yaml key
        if scope is not None:
            comps[f"channels.yaml[{key}]"] = hashlib.sha256(_json_bytes(scope)).hexdigest()[:16]
    h = hashlib.sha256(json.dumps(comps, sort_keys=True).encode()).hexdigest()[:16]
    ver = {"hash": h, "components": comps, "recipe": key}
    _VERSION_CACHE[key] = ver
    return ver




def unit_key(specs: list[FileSpec], recipe: str | None, seed: int) -> str:
    h = hashlib.sha256()
    h.update(f"{channel_name(recipe)}|{seed}|{CACHE_FORMAT_VERSION}".encode())
    for s in sorted(specs, key=lambda s: s.file_id):
        h.update(f"|{s.file_id}:{s.label}:{s.attack_type}:{s.attack_id}:{s.convert_prob:.6f}:{s.keep_short_prob:.6f}:{s.source_set}".encode())
    return h.hexdigest()[:12]


def unit_dir(cache_root: Path, name: str, specs: list[FileSpec], recipe: str | None, seed: int) -> Path:
    return Path(cache_root) / f"{name}__{channel_name(recipe)}__{unit_key(specs, recipe, seed)}"


# ---------------------------------------------------------------- build

def _build_shard(task) -> dict:
    shard_path, specs, recipe, seed = task
    shard_path = Path(shard_path)
    if shard_path.exists():
        with np.load(shard_path) as z:
            return json.loads(str(z["summary"]))
    windows, statuses = [], []
    for spec in specs:
        ws, status, _dur = process_file(spec, recipe, seed)
        windows += ws
        statuses.append((spec.file_id, spec.source_set, status, len(ws)))
    n = len(windows)
    seq = np.zeros((n, N_FRAMES, N_LFCC), dtype=SEQ_DTYPE)
    for i, w in enumerate(windows):
        if w.lfcc_seq.shape != (N_FRAMES, N_LFCC):
            raise ValueError(f"{w.file_id}: lfcc_seq shape {w.lfcc_seq.shape}, expected {(N_FRAMES, N_LFCC)}")
        seq[i] = w.lfcc_seq
    summary = json.loads(json.dumps({"n": n, "statuses": statuses}))  # same shape fresh or resumed
    tmp = shard_path.with_suffix(".tmp.npz")
    np.savez(
        tmp, seq=seq,
        scalars=np.stack([w.scalars for w in windows]).astype(np.float32) if n else np.zeros((0, N_SCALARS), np.float32),
        file_id=np.array([w.file_id for w in windows], dtype=str),
        label=np.array([w.label for w in windows], dtype=np.int8),
        attack_type=np.array([w.attack_type for w in windows], dtype=np.int16),
        attack_id=np.array([w.attack_id for w in windows], dtype=np.int8),
        window_index=np.array([w.window_index for w in windows], dtype=np.int16),
        pad_fraction=np.array([w.pad_fraction for w in windows], dtype=np.float32),
        duration=np.array([w.duration for w in windows], dtype=np.float32),
        channel=np.array([w.channel for w in windows], dtype=str),
        source_set=np.array([w.source_set for w in windows], dtype=str),
        summary=np.array(json.dumps(summary)),
    )
    os.replace(tmp, shard_path)
    return summary


def build_unit(
    name: str, specs: list[FileSpec], recipe: str | None, cache_root: Path = DEFAULT_CACHE_ROOT,
    seed: int = 0, workers: int = 8, shard_files: int = 256, rebuild: bool = False, pool=None,
) -> Path:
    """Builds (or verifies) one cache unit and returns its directory."""
    d = unit_dir(cache_root, name, specs, recipe, seed)
    manifest_path = d / "manifest.json"
    fv = feature_version(recipe, all_recipes=False)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("feature_version") == fv["hash"]:
            return d
        if not rebuild:
            raise StaleCacheError(
                f"{d} was built with feature version {manifest.get('feature_version')}, current is "
                f"{fv['hash']}. Features/windowing/channel code changed since. Rebuild it (rebuild=True "
                "/ --rebuild-stale) or delete it; stale features are never reused."
            )
        shutil.rmtree(d)
    shards_dir = d / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    fv_marker = shards_dir / "feature_version.txt"
    if fv_marker.exists() and fv_marker.read_text().strip() != fv["hash"]:
        shutil.rmtree(shards_dir)  # half-built shards from older code: never mixed in
        shards_dir.mkdir(parents=True)
    fv_marker.write_text(fv["hash"])

    specs = sorted(specs, key=lambda s: s.file_id)
    tasks = [(str(shards_dir / f"shard_{i // shard_files:05d}.npz"), specs[i:i + shard_files], recipe, seed)
             for i in range(0, len(specs), shard_files)]
    t0 = time.time()
    print(f"[cache] {d.name}: {len(specs)} files, {len(tasks)} shards, channel={channel_name(recipe)}",
          file=sys.stderr)
    if workers <= 1 or len(tasks) <= 1:
        summaries = [_build_shard(t) for t in tasks]
    else:
        own_pool = pool is None
        pool = pool or ProcessPoolExecutor(max_workers=workers)
        try:
            summaries = []
            for i, s in enumerate(pool.map(_build_shard, tasks)):
                summaries.append(s)
                if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                    el = time.time() - t0
                    print(f"[cache]   {d.name}: {i + 1}/{len(tasks)} shards, {el / 60:.1f} min", file=sys.stderr)
        finally:
            if own_pool:
                pool.shutdown()

    _consolidate(d, tasks, summaries, specs, recipe, name, seed, fv)
    shutil.rmtree(shards_dir)
    print(f"[cache] {d.name}: done in {(time.time() - t0) / 60:.1f} min", file=sys.stderr)
    return d


def _consolidate(d: Path, tasks, summaries, specs, recipe, name, seed, fv) -> None:
    n_total = sum(s["n"] for s in summaries)
    seq_mm = np.lib.format.open_memmap(d / "seq.tmp.npy", mode="w+", dtype=SEQ_DTYPE,
                                       shape=(n_total, N_FRAMES, N_LFCC))
    parts = {k: [] for k in META_FIELDS}
    scalars = []
    off = 0
    for (shard_path, *_), s in zip(tasks, summaries):
        with np.load(shard_path) as z:
            n = int(s["n"])
            seq_mm[off:off + n] = z["seq"]
            scalars.append(z["scalars"])
            for k in META_FIELDS:
                parts[k].append(z[k])
            off += n
    seq_mm.flush()
    del seq_mm
    os.replace(d / "seq.tmp.npy", d / "seq.npy")
    np.save(d / "scalars.npy", np.concatenate(scalars) if scalars else np.zeros((0, N_SCALARS), np.float32))
    np.savez(d / "meta.npz", **{k: np.concatenate(v) if v else np.array([]) for k, v in parts.items()})

    spec_by_id = {s.file_id: s for s in specs}
    results = [(spec_by_id[fid], recipe, n, status) for s in summaries for fid, _set, status, n in s["statuses"]]
    import io
    buf = io.StringIO()
    yield_report = report_yield(results, stream=buf)
    sys.stderr.write(buf.getvalue())
    status_by_file = {fid: status for s in summaries for fid, _set, status, _n in s["statuses"]}
    manifest = {
        "name": name, "channel": channel_name(recipe), "seed": seed,
        "feature_version": fv["hash"], "feature_version_components": fv["components"],
        "n_files": len(specs), "n_windows": n_total,
        "n_files_with_windows": sum(1 for v in status_by_file.values() if v == STATUS_OK),
        "yield": yield_report,
        "dropped_files": {fid: st for fid, st in status_by_file.items() if st != STATUS_OK},
        # per-file pad policy: what revalidate() needs to re-render a sample
        # with EXACTLY the same pad params (and thus the same windows).
        "pad_policy": {s.file_id: [s.convert_prob, s.keep_short_prob] for s in specs},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "first_spec": asdict(specs[0]) if specs else None,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))


# ---------------------------------------------------------------- read

class CacheCollection:
    """Several cache units viewed as one window table. Metadata and scalars
    are in RAM (small); LFCC sequences stay memory-mapped on disk."""

    def __init__(self, dirs: list[Path], check_version: bool = True):
        self.dirs = [Path(d) for d in dirs]
        self._seqs, offsets, meta_parts, scalars = [], [0], {k: [] for k in META_FIELDS}, []
        self.manifests = []
        for d in self.dirs:
            mp = d / "manifest.json"
            if not mp.exists():
                raise FileNotFoundError(f"{d}: no manifest.json (unbuilt or interrupted cache unit)")
            manifest = json.loads(mp.read_text())
            if check_version:
                # Each unit's manifest is stamped with the feature version of
                # ITS OWN recipe's scope (recipe-scoped invalidation), so a
                # change to an unrelated recipe must not flag it stale.
                ch = manifest.get("channel", "none")
                recipe = None if ch == "none" else ch
                fv = feature_version(recipe, all_recipes=False)["hash"]
                if manifest.get("feature_version") != fv:
                    raise StaleCacheError(
                        f"{d}: feature version {manifest.get('feature_version')} != current {fv} "
                        f"for recipe {recipe!r}")
            self.manifests.append(manifest)
            seq = np.load(d / "seq.npy", mmap_mode="r")
            self._seqs.append(seq)
            offsets.append(offsets[-1] + len(seq))
            scalars.append(np.load(d / "scalars.npy"))
            with np.load(d / "meta.npz") as z:
                for k in META_FIELDS:
                    meta_parts[k].append(z[k])
        self._offsets = np.array(offsets)
        self.n = int(offsets[-1])
        self.scalars = np.concatenate(scalars).astype(np.float32) if scalars else np.zeros((0, N_SCALARS), np.float32)
        for k in META_FIELDS:
            vals = [v for v in meta_parts[k] if len(v)]
            setattr(self, k, np.concatenate(vals) if vals else np.array([]))
        self.label = self.label.astype(np.int64)
        self.attack_type = self.attack_type.astype(np.int64)
        self.attack_id = self.attack_id.astype(np.int64)
        for k in META_FIELDS:
            assert len(getattr(self, k)) == self.n, f"meta field {k} length mismatch"

    def __len__(self) -> int:
        return self.n

    def get_seq(self, idx: np.ndarray) -> np.ndarray:
        """float32 (len(idx), 184, 60) in the order of idx."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((len(idx), N_FRAMES, N_LFCC), dtype=np.float32)
        order = np.argsort(idx, kind="stable")
        sidx = idx[order]
        unit = np.searchsorted(self._offsets, sidx, side="right") - 1
        for u in np.unique(unit):
            m = unit == u
            local = sidx[m] - self._offsets[u]
            out[order[m]] = self._seqs[u][local]
        return out

    def iter_batches(self, idx: np.ndarray | None = None, batch: int = 4096):
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            yield b, self.get_seq(b), self.scalars[b]
# ---------------------------------------------------------------- revalidate

def _reconstruct_specs_from_meta(unit_dir: Path, recipe: str | None) -> list:
    """Rebuild the FileSpecs that produced a cache unit from its meta so a
    sample can be re-rendered with EXACTLY the pad policy that was used.
    v13+ goes through the stored per-file `pad_policy` (exact). Legacy units
    (no `pad_policy`) re-derive it by name: `train_` units use the v12
    raw-header pre-pass, everything else the natural (0.0, 1.0) defaults.
    Best-effort for legacy sets whose policy was min() capped per set; a miss
    reads as not-equivalent (safe: rebuild instead of re-stamp)."""
    from dataset import DATA_ROOT, FileSpec, compute_pad_policy, estimate_duration_seconds

    with np.load(unit_dir / "meta.npz") as z:
        fid = [str(x) for x in z["file_id"]]
        label = [int(x) for x in z["label"]]
        attack_type = [int(x) for x in z["attack_type"]]
        attack_id = [int(x) for x in z["attack_id"]]
        source_set = [str(x) for x in z["source_set"]]
        duration = [float(x) for x in z["duration"]]
    manifest = json.loads((unit_dir / "manifest.json").read_text())
    pad_policy = manifest.get("pad_policy")

    per = {}
    first_dur = {}
    for i, f in enumerate(fid):
        per.setdefault(f, (label[i], attack_type[i], attack_id[i], source_set[i]))
        first_dur.setdefault(f, duration[i])

    order = sorted(set(fid), key=lambda k: fid.index(k))
    specs_by_id = {}
    for f in order:
        lbl, at, aid, ss = per[f]
        if pad_policy and f in pad_policy:
            conv, keep = pad_policy[f]
        else:
            conv, keep = 0.0, 1.0  # default; recomputed below for legacy
        specs_by_id[f] = FileSpec(
            path=str(DATA_ROOT / f), file_id=f, label=lbl, source_set=ss,
            attack_type=at, attack_id=aid,
            convert_prob=conv, keep_short_prob=keep)

    if not pad_policy:
        by_set = {}
        for s in specs_by_id.values():
            by_set.setdefault(s.source_set, []).append(s)
        unit_name = str(manifest.get("name", ""))
        for ss, grp in by_set.items():
            if unit_name.startswith("train_"):
                # v12 training units: raw-header durations (same as the
                # training pad-policy pre-pass). Eval units: natural defaults.
                conv, keep, _frac = compute_pad_policy([estimate_duration_seconds(Path(s.path)) for s in grp])
            else:
                conv, keep = 0.0, 1.0
            for g in grp:
                g.convert_prob = conv
                g.keep_short_prob = keep
    return [specs_by_id[f] for f in order]


def revalidate(unit_dir_path, *, seed: int = 0, max_sample_files: int = 12,
               rel_tol: float = 1e-2, atol: float = 1e-2, scalars_atol: float = 1e-3) -> dict:
    """Re-render a deterministic sample of a stale cache unit under the current
    code and compare it to the stored features.

    Purpose (post-v12 plan step 3b): when a code change bumps `feature_version`,
    decide between re-stamping the manifest (output truly unchanged) and a full
    rebuild — by MEASUREMENT, not assumption. Re-renders a hash-selected,
    deterministic sample of the unit's files with the SAME pad policy stored in
    the manifest, and compares LFCC sequences (stored fp16) and scalars
    (stored fp32) to freshly extracted ones within tolerance.

    - equivalent=True  -> manifest re-stamped to the current recipe version,
      recording `revalidated_from` (old hash, sample_n, max_abs_diff).
    - equivalent=False -> unit left as-is (still stale); caller should rebuild.
      Per-file max diffs are returned for diagnostics.
    """
    unit_dir_path = Path(unit_dir_path)
    manifest_path = unit_dir_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ch = manifest.get("channel", "none")
    recipe = None if ch == "none" else ch
    seed = int(manifest.get("seed", seed))

    specs = _reconstruct_specs_from_meta(unit_dir_path, recipe)
    with np.load(unit_dir_path / "meta.npz") as z:
        fid = [str(x) for x in z["file_id"]]
        win_idx = [int(x) for x in z["window_index"]]

    stored_seq = np.load(unit_dir_path / "seq.npy", mmap_mode="r")
    stored_scalars = np.load(unit_dir_path / "scalars.npy")

    sample = sorted(specs, key=lambda s: (stable_seed(s.file_id, seed, "revalidate"), s.file_id))
    sample = sorted(sample[:max_sample_files], key=lambda s: s.file_id)

    max_abs_diff = 0.0
    per_file = {}
    for spec in sample:
        windows, status, _dur = process_file(spec, recipe, seed)
        if status != STATUS_OK:
            per_file[spec.file_id] = {"status": status}
            continue
        rows = sorted([i for i, f in enumerate(fid) if f == spec.file_id], key=lambda i: win_idx[i])
        if len(rows) != len(windows):
            per_file[spec.file_id] = {"status": "window_count_mismatch",
                                      "stored": len(rows), "fresh": len(windows)}
            continue
        ok, diff = _compare_windows(stored_seq[rows], stored_scalars[rows],
                                    windows, rel_tol, atol, scalars_atol)
        per_file[spec.file_id] = {"equivalent": bool(ok), "max_abs_diff": float(diff)}
        max_abs_diff = max(max_abs_diff, float(diff))

    equivalent = bool(sample) and all(v.get("equivalent") for v in per_file.values())
    info = {"equivalent": equivalent, "sample_n": len(sample),
            "max_abs_diff": float(max_abs_diff), "per_file": per_file, "recipe": recipe}

    if equivalent:
        old_hash = manifest.get("feature_version")
        new_fv = feature_version(recipe, all_recipes=False)
        manifest["feature_version"] = new_fv["hash"]
        manifest["feature_version_components"] = new_fv["components"]
        manifest["revalidated_from"] = {"old_hash": old_hash, "sample_n": len(sample),
                                        "max_abs_diff": float(max_abs_diff)}
        manifest_path.write_text(json.dumps(manifest, indent=1))
        info["restamped_to"] = new_fv["hash"]
    return info


def _compare_windows(stored_seqs, stored_scalars, windows, rel_tol, atol, scalars_atol):
    """Compare stored rows (already in window_index order) to freshly rendered
    windows. Returns (all_ok, max_abs_diff)."""
    import numpy as _np

    diffs = []
    for i, w in enumerate(windows):
        fresh_fp16 = _np.asarray(w.lfcc_seq, dtype=_np.float16).astype(_np.float32)
        seq_ok = _np.allclose(stored_seqs[i], fresh_fp16, rtol=rel_tol, atol=atol)
        sc_ok = _np.allclose(stored_scalars[i], _np.asarray(w.scalars, dtype=_np.float32),
                             atol=scalars_atol)
        if not (seq_ok and sc_ok):
            return False, float(_np.max(_np.abs(stored_seqs[i].astype(_np.float32) - fresh_fp16)))
        diffs.append(float(_np.max(_np.abs(stored_seqs[i].astype(_np.float32) - fresh_fp16))))
    return True, (max(diffs) if diffs else 0.0)
