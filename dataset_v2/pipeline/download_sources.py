#!/usr/bin/env python3
"""Resumable, checksummed downloader for Persian Dataset V2 sources.

* Writes to ``<workdir>/raw/<source_id>/...``; never to repository root files.
* Uses ``.part`` files plus HTTP Range for resume; falls back to a clean restart
  when a server ignores Range.
* Verifies Wikimedia SHA-1 when available and records SHA-256 for every file.
* Writes a checkpointed ``download_manifest.json`` after every file.
* ``--max-bytes`` is a safety brake for small test machines; use 0 (default)
  on the user's Windows PC for full dumps.

Only sources marked ``download=true`` in source_metadata.json are considered.
PROFILE C (quarantine) is never downloaded by this script.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    config_source_metadata, configure_logging, load_config, now_utc,
    require_free_space, sha1_file, sha256_file,
)

CHUNK = 1 << 20
USER_AGENT = "PersianDatasetV2/2.0 (resumable downloader; contact via repository)"


def _parse_total(resp, partial: int) -> int | None:
    cr = resp.headers.get("Content-Range", "")
    if cr and "/" in cr:
        try:
            return int(cr.rsplit("/", 1)[1])
        except ValueError:
            pass
    cl = resp.headers.get("Content-Length")
    if cl and cl.isdigit():
        return int(cl) + (partial if resp.status == 206 else 0)
    return None


def download_one(url: str, dest: Path, *, expected_size: int | None = None,
                 expected_sha1: str | None = None, max_bytes: int = 0,
                 timeout: float = 60.0, logger=None) -> dict:
    """Download one file with resume support. Returns a manifest record."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if dest.exists():
        size = dest.stat().st_size
        if expected_sha1 and sha1_file(dest) == expected_sha1:
            return {"path": str(dest), "size": size, "sha1": expected_sha1,
                    "sha256": sha256_file(dest), "status": "already_verified",
                    "resumed": False, "url": url}
        if expected_size and size == expected_size:
            return {"path": str(dest), "size": size, "sha1": sha1_file(dest),
                    "sha256": sha256_file(dest), "status": "already_present_by_size",
                    "resumed": False, "url": url}
        logger and logger.warning("%s exists but did not verify; re-downloading", dest)
        dest.unlink()

    partial = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if partial:
        headers["Range"] = f"bytes={partial}-"
    req = urllib.request.Request(url, headers=headers)
    resumed = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 206 and partial:
                mode = "ab"
                resumed = True
            else:
                mode = "wb"
                partial = 0
            total = _parse_total(resp, partial)
            if max_bytes and total and total > max_bytes:
                raise RuntimeError(
                    f"Refusing to download {url}: reported size {total:,} bytes exceeds "
                    f"--max-bytes {max_bytes:,}. Increase --max-bytes on the user PC.")
            written = partial
            with open(part, mode) as f:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if logger and (written // (10 * CHUNK)) != ((written - len(chunk)) // (10 * CHUNK)):
                        logger.info("  %s: %.1f MiB", dest.name, written / 1024 ** 2)
                    if max_bytes and written > max_bytes:
                        raise RuntimeError(
                            f"Download exceeded --max-bytes {max_bytes:,}: {url}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} while downloading {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error while downloading {url}: {e.reason}. "
                           "The .part file is kept; rerun to resume.") from e

    size = part.stat().st_size
    if expected_size and size != expected_size:
        raise RuntimeError(f"Size mismatch for {url}: got {size:,}, expected {expected_size:,}. "
                           "The .part file is kept for inspection; delete it to restart.")
    digest1 = sha1_file(part)
    if expected_sha1 and digest1 != expected_sha1:
        raise RuntimeError(f"SHA-1 mismatch for {url}: got {digest1}, expected {expected_sha1}. "
                           "The .part file is kept; delete it to restart.")
    os.replace(part, dest)
    rec = {"path": str(dest), "size": size, "sha1": digest1,
           "sha256": sha256_file(dest), "status": "downloaded",
           "resumed": resumed, "url": url, "downloaded_at_utc": now_utc()}
    logger and logger.info("downloaded %s (%s bytes, resumed=%s)", dest.name, f"{size:,}", resumed)
    return rec


def github_api_parts(api_url: str, source_dir: Path, pattern: str, *,
                     max_bytes: int, timeout: float, logger) -> list[dict]:
    """Download the split 7z parts listed by a GitHub contents API URL."""
    req = urllib.request.Request(api_url, headers={
        "User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            items = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Cannot list GitHub 7z parts at {api_url}: {e}") from e
    if not isinstance(items, list):
        raise RuntimeError(f"GitHub API did not return a directory listing: {api_url}")
    picks = [it for it in items if it.get("type") == "file" and fnmatch.fnmatch(it.get("name", ""), pattern)]
    picks.sort(key=lambda it: it["name"])
    if not picks:
        raise RuntimeError(f"No GitHub files matched {pattern!r} at {api_url}")
    parts_dir = source_dir / "parts"
    records = []
    for it in picks:
        rec = download_one(it["download_url"], parts_dir / it["name"],
                           expected_size=it.get("size"), max_bytes=max_bytes,
                           timeout=timeout, logger=logger)
        rec["part_name"] = it["name"]
        records.append(rec)
        logger.info("7z part %s (%s bytes)", it["name"], f"{rec['size']:,}")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Resumably download approved V2 sources")
    ap.add_argument("--config", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--profiles", nargs="*", default=None,
                    help="override configured profiles, e.g. A or A B")
    ap.add_argument("--only-sources", nargs="*", default=None)
    ap.add_argument("--max-bytes", type=int, default=0,
                    help="safety cap per file; 0 disables. Use 104857600 on a small test machine.")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--force", action="store_true", help="re-download even if a file verifies")
    ap.add_argument("--include-optional", action="store_true",
                    help="also download optional recovery sources (e.g. the V1 2021 Wikipedia mirror)")
    ap.add_argument("--min-free-gb", type=float, default=None,
                    help="override configured disk-space safety threshold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    meta_path = Path(args.metadata) if args.metadata else config_source_metadata(cfg)
    workdir = Path(args.workdir) if args.workdir else Path(cfg["workdir"])
    raw_root = workdir / "raw"
    logger = configure_logging(workdir, "dataset_v2.download")
    repo_root = Path(cfg.get("repo_root", "."))
    if workdir.resolve() == repo_root.resolve():
        raise RuntimeError("Refusing to use the repository root as --workdir; "
                           "V1 files must remain untouched. Use D:/GPT/dataset/v2_work.")
    min_free = args.min_free_gb if args.min_free_gb is not None else float(cfg.get("min_free_gb", 20))
    require_free_space(workdir, min_free, logger)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    profiles = args.profiles or cfg.get("default_profiles", ["A"])
    manifest_path = workdir / "download_manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {"version": "2.0.0", "sources": {}})

    selected = [s for s in meta["sources"]
                if s.get("download") and s["profile"] in profiles
                and (not args.only_sources or s["id"] in args.only_sources)]
    logger.info("selected %d downloadable sources for profiles %s", len(selected), profiles)
    for s in selected:
        source_dir = raw_root / s["id"]
        source_dir.mkdir(parents=True, exist_ok=True)
        records = []
        logger.info("[source] %s (%s)", s["id"], s["name"])
        for f in s.get("files", []):
            if f.get("required") is False and not args.include_optional:
                logger.info("[skip optional] %s / %s (use --include-optional to download)",
                            s["id"], f["filename"])
                continue
            if f.get("format") == "github_api_7z":
                recs = github_api_parts(f["url"] + "&per_page=100", source_dir,
                                        f["options"]["filename_pattern"],
                                        max_bytes=args.max_bytes, timeout=args.timeout,
                                        logger=logger)
                records.extend(recs)
            else:
                dest = source_dir / f["filename"]
                if args.force and dest.exists():
                    dest.unlink()
                rec = download_one(f["url"], dest,
                                   expected_size=f.get("reported_size_bytes"),
                                   expected_sha1=f.get("sha1"),
                                   max_bytes=args.max_bytes, timeout=args.timeout,
                                   logger=logger)
                rec["format"] = f["format"]
                rec["extractor"] = f["extractor"]
                records.append(rec)
            manifest["sources"][s["id"]] = {
                "profile": s["profile"], "decision": s["decision"],
                "license_content": s["license_content"], "files": records,
                "updated_at_utc": now_utc(),
            }
            # Checkpoint after every file group.
            tmp = manifest_path.with_name(manifest_path.name + ".part")
            tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
            os.replace(tmp, manifest_path)
    manifest["generated_at_utc"] = now_utc()
    manifest["config_version"] = cfg.get("config_version")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    logger.info("download manifest: %s", manifest_path)
    print(f"Downloaded/verified {sum(len(v['files']) for v in manifest['sources'].values())} file(s).")


if __name__ == "__main__":
    main()
