#!/usr/bin/env python3
"""Shared, dependency-light utilities for the Persian Dataset V2 pipeline.

Design goals: Windows/Python 3.12 compatibility, streaming I/O, bounded RAM,
atomic writes, deterministic hashes, explicit logging and clear errors.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

DOC_SEPARATOR = "<|endoftext|>"
DEFAULT_ENCODING = "utf-8"


# --------------------------------------------------------------------------- #
# Paths / configuration
# --------------------------------------------------------------------------- #

def repo_root_from_here() -> Path:
    """Return the repository root assuming this file is dataset_v2/pipeline/common.py."""
    return Path(__file__).resolve().parents[2]


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding=DEFAULT_ENCODING) as f:
        return json.load(f)


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = repo_root_from_here() / "dataset_v2" / "configs" / "dataset_config.json"
    return load_json(path)


def _is_windows_drive_path(p: Path) -> bool:
    return len(p.parts) > 1 and ":" in p.parts[0]


def config_repo_root(config: dict | None = None) -> Path:
    """Return the actual V2 checkout root, ignoring machine-specific settings.

    ``dataset_config.json`` deliberately has no ``repo_root``.  A stale
    ``D:/GPT/dataset/repo`` value would resolve against V1 and is unsafe.  All
    repository-relative configuration is therefore discovered from this file's
    location: ``.../dataset_v2/pipeline/common.py`` -> two parents up.
    ``config`` is accepted for call-site compatibility but intentionally unused.
    """
    return repo_root_from_here()


def resolve_config_path(value: str | Path, repo_root: str | Path | None = None) -> Path:
    """Resolve a config path.

    Relative paths always resolve against the actual checkout root returned by
    :func:`repo_root_from_here`.  A supplied ``repo_root`` is ignored: it may be
    a stale V1 path from an older config.  Absolute paths are returned as-is;
    catalog/config paths in this repository are relative.
    """
    p = Path(value)
    if p.is_absolute() or _is_windows_drive_path(p):
        return p
    return config_repo_root() / p


def config_source_metadata(config: dict) -> Path:
    return resolve_config_path(config["source_metadata"])


def config_policy(config: dict) -> Path:
    return resolve_config_path(config["source_policy"])


def config_tokenizer(config: dict) -> Path:
    return resolve_config_path(config["tokenizer_baseline"])


def assert_safe_output_dir(outdir: str | Path, repo_root: str | Path,
                           v1_filenames: Iterable[str] | None = None,
                           allow_existing: bool = False) -> Path:
    """Refuse to write into the repository root or over V1's root files."""
    out = Path(outdir).resolve()
    repo = Path(repo_root).resolve()
    if out == repo:
        raise RuntimeError(
            f"Refusing to use the repository root as --outdir ({out}). "
            "V1's train.txt/validation.txt live there. Use a separate directory, "
            "e.g. D:/GPT/dataset/v2_final.")
    names = list(v1_filenames or ["train.txt", "validation.txt", "README.txt",
                                  "sources.json", "dataset_statistics.json"])
    collisions = [n for n in names if (out / n).exists()]
    if collisions and not allow_existing:
        raise RuntimeError(
            "Refusing to write: V1-named files already exist in --outdir: "
            + ", ".join(collisions) + ". Choose a clean V2 output directory.")
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# Logging / time / memory / disk
# --------------------------------------------------------------------------- #

def configure_logging(workdir: str | Path, name: str = "dataset_v2") -> logging.Logger:
    Path(workdir).mkdir(parents=True, exist_ok=True)
    log_path = Path(workdir) / "pipeline.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    fmt.converter = time.gmtime
    fh = logging.FileHandler(log_path, encoding=DEFAULT_ENCODING)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def peak_rss_mb() -> float | None:
    """Best-effort peak resident memory; None when the platform cannot report it."""
    try:
        import resource  # type: ignore
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":           # bytes on macOS
            return round(v / (1024 * 1024), 2)
        return round(v / 1024, 2)              # KiB on Linux
    except Exception:
        pass
    if os.name == "nt":
        try:
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            counters = PROCESS_MEMORY_COUNTERS()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                ctypes.windll.kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
                ctypes.byref(counters), ctypes.sizeof(counters))
            if ok:
                return round(counters.PeakWorkingSetSize / (1024 * 1024), 2)
        except Exception:
            pass
    return None


def disk_free_bytes(path: str | Path) -> int:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


def require_free_space(path: str | Path, min_gb: float, logger: logging.Logger | None = None) -> int:
    free = disk_free_bytes(path)
    need = int(min_gb * 1024 ** 3)
    if free < need:
        raise RuntimeError(
            f"Not enough free disk space at {path}: {free / 1024**3:.1f} GiB free, "
            f"{min_gb:.1f} GiB required. Free space or change --workdir/--outdir.")
    if logger:
        logger.info("disk check: %.1f GiB free at %s (required %.1f GiB)",
                    free / 1024 ** 3, path, min_gb)
    return free


# --------------------------------------------------------------------------- #
# Deterministic hashing / checksums / atomic writes
# --------------------------------------------------------------------------- #

def stable_hash(text: str, seed: str = "", digest_size: int = 16) -> str:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=digest_size,
                        person=b"persian-v2"[:16])
    if seed:
        h.update(seed.encode("utf-8"))
    return h.hexdigest()


def stable_int(text: str, seed: str = "") -> int:
    """Deterministic 63-bit positive integer (safe for SQLite INTEGER)."""
    value = int.from_bytes(hashlib.blake2b(
        (seed + "\x1f" + text).encode("utf-8"), digest_size=8,
        person=b"persian-v2"[:16]).digest(), "big")
    return value & ((1 << 63) - 1)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_git_blob_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Compute Git's blob SHA-1 without loading the file into RAM."""
    size = os.path.getsize(path)
    h = hashlib.sha1()
    h.update(b"blob " + str(size).encode("ascii") + b"\0")
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dumps(obj: Any, *, pretty: bool = True) -> str:
    if pretty:
        return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def atomic_write_text(path: str | Path, text: str, encoding: str = DEFAULT_ENCODING) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    with open(tmp, "w", encoding=encoding, newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json_dumps(obj))


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding=DEFAULT_ENCODING) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{lineno}: {e}") from e


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".part")
    n = 0
    with open(tmp, "w", encoding=DEFAULT_ENCODING, newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    return n


def append_jsonl(handle, row: dict) -> int:
    data = json.dumps(row, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode(DEFAULT_ENCODING)
    handle.write(data + b"\n")
    return len(data) + 1


def read_jsonl_at(shard_path: str | Path, offset: int, byte_len: int) -> dict:
    with open(shard_path, "rb") as f:
        f.seek(offset)
        raw = f.read(byte_len)
    return json.loads(raw.decode(DEFAULT_ENCODING))


def safe_excerpt(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")
