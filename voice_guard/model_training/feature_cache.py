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


_FEATURE_VERSION: dict | None = None


def feature_version() -> dict:
    """{"hash": ..., "components": {...}} over everything that shapes a window."""
    global _FEATURE_VERSION
    if _FEATURE_VERSION is not None:
        return _FEATURE_VERSION
    import librosa
    import soundfile

    tc = VAANI_ROOT / "telechannel"
    files = [MODEL_TRAINING_DIR / "features.py", MODEL_TRAINING_DIR / "dataset.py",
             tc / "configs" / "channels.yaml", tc / "pipeline.py"] + sorted((tc / "stages").glob("*.py"))
    components = {
        "cache_format": CACHE_FORMAT_VERSION,
        "numpy": np.__version__, "soundfile": soundfile.__version__, "librosa": librosa.__version__,
        "ffmpeg": _ffmpeg_version(),
    }
    for f in files:
        rel = f.relative_to(VAANI_ROOT.parent).as_posix() if VAANI_ROOT.parent in f.parents else f.name
        components[rel] = hashlib.sha256(_code_fingerprint(f)).hexdigest()[:16]
    h = hashlib.sha256(json.dumps(components, sort_keys=True).encode()).hexdigest()[:16]
    _FEATURE_VERSION = {"hash": h, "components": components}
    return _FEATURE_VERSION


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
    fv = feature_version()
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
        fv = feature_version()["hash"] if check_version else None
        self._seqs, offsets, meta_parts, scalars = [], [0], {k: [] for k in META_FIELDS}, []
        self.manifests = []
        for d in self.dirs:
            mp = d / "manifest.json"
            if not mp.exists():
                raise FileNotFoundError(f"{d}: no manifest.json (unbuilt or interrupted cache unit)")
            manifest = json.loads(mp.read_text())
            if check_version and manifest["feature_version"] != fv:
                raise StaleCacheError(f"{d}: feature version {manifest['feature_version']} != current {fv}")
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
