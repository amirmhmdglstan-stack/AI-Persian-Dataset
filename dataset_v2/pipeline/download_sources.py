#!/usr/bin/env python3
"""Resumable, checksummed downloader for Persian Dataset V2 sources.

* Writes to ``<workdir>/raw/<source_id>/...``; never to repository root files.
* Uses ``.part`` files plus HTTP Range for resume; falls back to a clean restart
  when a server ignores Range.
* Loads ``download_manifest.json`` and matches a completed destination to its
  source URL AND filename, then verifies the ACTUAL SHA-256 and size before
  skipping.  A filename alone is never accepted as file identity.
* Records SHA-256/size and, for mutable weekly Tatoeba snapshots, the observed
  HTTP snapshot date/checksum metadata.  Ordinary resume does not refresh a
  verified weekly snapshot; ``--force`` is the explicit refresh mechanism.
* ``--max-bytes`` is a safety brake for small test machines; use 0 (default)
  on the user's Windows PC for full dumps.

Only sources marked ``download=true`` in source_metadata.json are considered.
PROFILE C (quarantine) is never downloaded by this script.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    config_source_metadata, configure_logging, load_config, now_utc,
    repo_root_from_here, require_free_space, sha1_file, sha256_file,
)

CHUNK = 1 << 20
USER_AGENT = "PersianDatasetV2/2.1 (resumable downloader; contact via repository)"
MANIFEST_VERSION = "2.1.0"


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


def _manifest_filename(rec: dict) -> str:
    """Extract a filename from a manifest record written on Windows or Linux."""
    name = rec.get("filename")
    if name:
        return str(name)
    path = str(rec.get("path", ""))
    return PureWindowsPath(path).name or Path(path).name


def manifest_record_for(manifest: dict, source_id: str, url: str, dest: Path) -> dict | None:
    """Find a checkpoint record by BOTH source URL and destination filename.

    This deliberately does not match on filename alone: a mutable weekly export
    or a reused filename from another URL is not the same snapshot.
    """
    files = manifest.get("sources", {}).get(source_id, {}).get("files", [])
    for rec in files:
        if rec.get("url") == url and _manifest_filename(rec) == dest.name:
            return rec
    return None


def _snapshot_fields(headers, mutable: bool, catalog_snapshot_date: str | None = None) -> dict:
    out = {"mutable_weekly_snapshot": bool(mutable)}
    if catalog_snapshot_date:
        out["catalog_snapshot_date"] = catalog_snapshot_date
    if not mutable:
        return out
    last_modified = headers.get("Last-Modified") or headers.get("last-modified") or ""
    etag = headers.get("ETag") or headers.get("etag") or ""
    out["snapshot_last_modified"] = last_modified
    out["snapshot_etag"] = etag
    if last_modified:
        try:
            out["snapshot_date"] = parsedate_to_datetime(last_modified).date().isoformat()
        except (TypeError, ValueError, OverflowError):
            out["snapshot_date"] = last_modified[:10]
    return out


def _manifest_snapshot_fields(rec: dict) -> dict:
    keys = ("mutable_weekly_snapshot", "catalog_snapshot_date", "snapshot_date",
            "snapshot_last_modified", "snapshot_etag", "verified_at_utc")
    return {k: rec[k] for k in keys if k in rec}


def download_one(url: str, dest: Path, *, expected_size: int | None = None,
                 expected_sha1: str | None = None, max_bytes: int = 0,
                 timeout: float = 60.0, logger=None, manifest_record: dict | None = None,
                 force: bool = False, mutable_weekly_snapshot: bool = False,
                 catalog_snapshot_date: str | None = None,
                 source_id: str | None = None) -> dict:
    """Download one file with manifest-verified resume support.

    A manifest record is trusted only when its URL and filename match this
    destination and the completed file's actual SHA-256 AND size match the
    checkpoint.  ``--force`` bypasses the skip and performs a clean refresh.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    sidecar = part.with_name(part.name + ".json")
    label = f"{source_id or 'source'}/{dest.name}"

    if manifest_record is not None and not (
            manifest_record.get("url") == url
            and _manifest_filename(manifest_record) == dest.name):
        logger and logger.warning(
            "%s: ignoring a manifest record that does not match URL + filename", label)
        manifest_record = None

    if force:
        logger and logger.warning("%s: --force requested; performing explicit refresh", label)
        if dest.exists():
            dest.unlink()
        if part.exists():
            part.unlink()
        sidecar.unlink(missing_ok=True)
    elif dest.exists():
        if manifest_record is not None:
            try:
                manifest_size = int(manifest_record["size"])
            except (KeyError, TypeError, ValueError):
                manifest_size = None
            manifest_sha256 = str(manifest_record.get("sha256", "")).lower()
            actual_size = dest.stat().st_size
            actual_sha256 = sha256_file(dest).lower()
            if (manifest_size is not None and manifest_sha256
                    and actual_size == manifest_size and actual_sha256 == manifest_sha256):
                rec = {
                    "path": str(dest), "filename": dest.name, "source_id": source_id,
                    "url": url, "size": actual_size, "sha1": manifest_record.get("sha1"),
                    "sha256": actual_sha256, "status": "already_verified_from_manifest",
                    "resumed": False, "force_refresh": False,
                    "checksum_algorithm": "sha256",
                    "verified_at_utc": now_utc(),
                }
                rec.update(_manifest_snapshot_fields(manifest_record))
                logger and logger.info(
                    "[skip] %s: manifest SHA-256 + size verified; no ordinary-resume refresh", label)
                return rec
            logger and logger.warning(
                "%s exists but manifest identity failed (size %s vs %s, sha256 %s vs %s); "
                "refusing to trust filename alone; re-downloading",
                label, actual_size, manifest_size, actual_sha256[:12], manifest_sha256[:12])
            dest.unlink()
            if part.exists():
                part.unlink()
            sidecar.unlink(missing_ok=True)
        elif expected_sha1:
            actual_size = dest.stat().st_size
            digest1 = sha1_file(dest)
            if digest1 == expected_sha1 and (expected_size is None or actual_size == expected_size):
                rec = {
                    "path": str(dest), "filename": dest.name, "source_id": source_id,
                    "url": url, "size": actual_size, "sha1": digest1,
                    "sha256": sha256_file(dest), "status": "already_verified_by_catalog_sha1",
                    "resumed": False, "force_refresh": False,
                    "checksum_algorithm": "sha256", "verified_at_utc": now_utc(),
                    "mutable_weekly_snapshot": False,
                }
                logger and logger.info("[skip] %s: immutable catalog SHA-1 verified", label)
                return rec
            logger and logger.warning(
                "%s exists but immutable catalog SHA-1/size failed; re-downloading", label)
            dest.unlink()
            if part.exists():
                part.unlink()
            sidecar.unlink(missing_ok=True)
        else:
            logger and logger.warning(
                "%s exists but has no matching URL+filename manifest and no catalog SHA-1; "
                "refusing filename-only trust; re-downloading", label)
            dest.unlink()
            if part.exists():
                part.unlink()
            sidecar.unlink(missing_ok=True)

    partial = part.stat().st_size if part.exists() else 0
    validator = None
    if partial and mutable_weekly_snapshot:
        try:
            state = json.loads(sidecar.read_text(encoding="utf-8"))
            validator = (state.get("etag") if not state.get("etag", "").startswith("W/")
                         else None) or state.get("last_modified")
            if (state.get("url") != url or state.get("size") != partial or
                    state.get("sha256") != sha256_file(part) or not validator):
                raise ValueError("partial identity or validator missing")
        except (ValueError, OSError, json.JSONDecodeError):
            part.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            partial = 0
            validator = None
    headers = {"User-Agent": USER_AGENT}
    if partial:
        headers["Range"] = f"bytes={partial}-"
        if validator:
            headers["If-Range"] = validator
    req = urllib.request.Request(url, headers=headers)
    resumed = False
    snapshot: dict = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            snapshot = _snapshot_fields(resp.headers, mutable_weekly_snapshot,
                                        catalog_snapshot_date)
            if resp.status == 206:
                import re
                cr = resp.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", cr)
                if (not partial or not match or int(match.group(1)) != partial or
                        int(match.group(2)) < partial or
                        (match.group(3) != "*" and int(match.group(3)) <= int(match.group(2)))):
                    raise ValueError("invalid Content-Range for partial download")
                for key, header in (("snapshot_etag", "ETag"),
                                    ("snapshot_last_modified", "Last-Modified")):
                    if mutable_weekly_snapshot and state.get(key.replace("snapshot_", "")) and \
                            (resp.headers.get(header) not in (None, state[key.replace("snapshot_", "")]) or
                             (validator == state[key.replace("snapshot_", "")] and
                              resp.headers.get(header) is None)):
                        raise ValueError("snapshot validator changed during resume")
                mode = "ab"
                resumed = True
            elif resp.status == 200:
                mode = "wb"
                partial = 0
            else:
                raise ValueError(f"unexpected HTTP response {resp.status}")
            total = _parse_total(resp, partial)
            if max_bytes and total and total > max_bytes:
                raise RuntimeError(f"Refusing to download {url}: reported size {total:,} exceeds --max-bytes")
            written = partial
            try:
                with open(part, mode) as f:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        written += len(chunk)
                        if max_bytes and written > max_bytes:
                            raise RuntimeError(f"Download exceeded --max-bytes {max_bytes:,}: {url}")
            finally:
                if part.exists():
                    sidecar.write_text(json.dumps({"url": url, "size": part.stat().st_size,
                        "sha256": sha256_file(part),
                        "etag": resp.headers.get("ETag", ""),
                        "last_modified": resp.headers.get("Last-Modified", "")}), encoding="utf-8")
    except ValueError as e:
        part.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        # A clean request is safer than appending an inconsistent range.
        if partial:
            return download_one(url, dest, expected_size=expected_size, expected_sha1=expected_sha1,
                max_bytes=max_bytes, timeout=timeout, logger=logger, force=True,
                mutable_weekly_snapshot=mutable_weekly_snapshot,
                catalog_snapshot_date=catalog_snapshot_date, source_id=source_id)
        raise RuntimeError(str(e)) from e
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} while downloading {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error while downloading {url}: {e.reason}. "
                           "The .part file is kept; rerun to resume.") from e

    size = part.stat().st_size
    if total is not None and size != total:
        raise RuntimeError(f"Incomplete HTTP body for {url}: got {size}, expected {total}; partial kept")
    if expected_size and size != expected_size:
        raise RuntimeError(f"Size mismatch for {url}: got {size:,}, expected {expected_size:,}. "
                           "The .part file is kept for inspection; delete it to restart.")
    digest1 = sha1_file(part)
    if expected_sha1 and digest1 != expected_sha1:
        raise RuntimeError(f"SHA-1 mismatch for {url}: got {digest1}, expected {expected_sha1}. "
                           "The .part file is kept; delete it to restart.")
    os.replace(part, dest)
    sidecar.unlink(missing_ok=True)
    rec = {
        "path": str(dest), "filename": dest.name, "source_id": source_id,
        "url": url, "size": size, "sha1": digest1, "sha256": sha256_file(dest),
        "status": "downloaded", "resumed": resumed, "force_refresh": force,
        "checksum_algorithm": "sha256", "downloaded_at_utc": now_utc(),
        "verified_at_utc": now_utc(),
    }
    rec.update(snapshot)
    logger and logger.info("downloaded %s (%s bytes, resumed=%s, snapshot=%s)",
                           dest.name, f"{size:,}", resumed,
                           snapshot.get("snapshot_date") or "immutable")
    return rec


def github_api_parts(api_url: str, source_dir: Path, pattern: str, *,
                     max_bytes: int, timeout: float, logger, force: bool = False,
                     manifest: dict | None = None, source_id: str | None = None) -> list[dict]:
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
    picks = [it for it in items if it.get("type") == "file"
             and fnmatch.fnmatch(it.get("name", ""), pattern)]
    picks.sort(key=lambda it: it["name"])
    if not picks:
        raise RuntimeError(f"No GitHub files matched {pattern!r} at {api_url}")
    parts_dir = source_dir / "parts"
    records = []
    for it in picks:
        dest = parts_dir / it["name"]
        mrec = manifest_record_for(manifest or {}, source_id or "", it["download_url"], dest)
        rec = download_one(it["download_url"], dest,
                           expected_size=it.get("size"), max_bytes=max_bytes,
                           timeout=timeout, logger=logger, manifest_record=mrec,
                           force=force, source_id=source_id)
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
    ap.add_argument("--force", action="store_true",
                    help="explicitly refresh/download even when a file verifies")
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
    repo_root = repo_root_from_here()
    if workdir.resolve() == repo_root.resolve():
        raise RuntimeError("Refusing to use the repository root as --workdir; "
                           "V1 files must remain untouched. Use D:/GPT/dataset/v2_work.")
    min_free = args.min_free_gb if args.min_free_gb is not None else float(cfg.get("min_free_gb", 20))
    require_free_space(workdir, min_free, logger)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    profiles = args.profiles or cfg.get("default_profiles", ["A"])
    manifest_path = workdir / "download_manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {"version": MANIFEST_VERSION, "sources": {}})
    manifest["version"] = MANIFEST_VERSION
    manifest["mutable_snapshot_policy"] = {
        "identity": "source URL + destination filename + actual SHA-256 + actual size",
        "ordinary_resume": "skip a verified mutable weekly snapshot; no automatic refresh",
        "explicit_refresh": "use --force to download and record a new weekly snapshot",
    }

    selected = [s for s in meta["sources"]
                if s.get("download") and s["profile"] in profiles
                and (not args.only_sources or s["id"] in args.only_sources)]
    logger.info("selected %d downloadable sources for profiles %s", len(selected), profiles)
    for s in selected:
        source_dir = raw_root / s["id"]
        source_dir.mkdir(parents=True, exist_ok=True)
        records = []
        fallback_needed = False
        primary_available = False
        logger.info("[source] %s (%s)", s["id"], s["name"])
        for f in s.get("files", []):
            if f.get("fallback_only") and s["id"] == "tatoeba_fa_cc_by":
                if primary_available or not fallback_needed:
                    continue
            elif f.get("required") is False and not args.include_optional:
                logger.info("[skip optional] %s / %s (use --include-optional to download)",
                            s["id"], f["filename"])
                continue
            if f.get("format") == "github_api_7z":
                recs = github_api_parts(f["url"] + "&per_page=100", source_dir,
                                        f["options"]["filename_pattern"],
                                        max_bytes=args.max_bytes, timeout=args.timeout,
                                        logger=logger, force=args.force,
                                        manifest=manifest, source_id=s["id"])
                records.extend(recs)
            else:
                dest = source_dir / f["filename"]
                mrec = manifest_record_for(manifest, s["id"], f["url"], dest)
                try:
                    rec = download_one(
                    f["url"], dest,
                    expected_size=f.get("reported_size_bytes"),
                    expected_sha1=f.get("sha1"),
                    max_bytes=args.max_bytes, timeout=args.timeout, logger=logger,
                    manifest_record=mrec, force=args.force,
                    mutable_weekly_snapshot=bool(f.get("mutable_weekly_snapshot", False)),
                    catalog_snapshot_date=f.get("snapshot_date"),
                    source_id=s["id"])
                except RuntimeError as exc:
                    if (s["id"] == "tatoeba_fa_cc_by" and not f.get("fallback_only") and
                            (str(exc).startswith("HTTP 404 ") or str(exc).startswith("HTTP 410 "))):
                        fallback_needed = True
                        logger.warning("primary Tatoeba export permanently unavailable: %s", exc)
                        continue
                    raise
                if s["id"] == "tatoeba_fa_cc_by" and not f.get("fallback_only"):
                    primary_available = True
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
