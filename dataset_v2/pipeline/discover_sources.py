#!/usr/bin/env python3
"""Discover remote source metadata without downloading data.

Makes HEAD (or a 1-byte ranged GET fallback) requests and records
Content-Length and availability under ``--workdir``.  Probe results are written
to ``<workdir>/source_metadata_probe.json``; the tracked ``source_metadata.json``
and derived CSVs are NOT modified by normal launcher runs.  ``--export-only`` is
the explicit opt-in that regenerates the derived CSVs.  Use ``--offline`` to
skip all network access.  This script never downloads a corpus file.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    atomic_write_text, config_source_metadata, configure_logging, json_dumps,
    load_config, now_utc, require_free_space,
)


def _load_builder():
    path = Path(__file__).resolve().parents[1] / "sources" / "_build_metadata.py"
    spec = importlib.util.spec_from_file_location("v2_catalog_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load catalog builder at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe_url(url: str, timeout: float = 12.0) -> dict:
    """Return {status, size_bytes, final_url, error} using HEAD/Range only."""
    headers = {"User-Agent": "PersianDatasetV2/2.1 (metadata-only probe)"}
    for method, extra in (("HEAD", {}), ("GET", {"Range": "bytes=0-0"})):
        h = {**headers, **extra}
        req = urllib.request.Request(url, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                size = None
                if method == "GET" and resp.status == 206:
                    cr = resp.headers.get("Content-Range", "")
                    if "/" in cr:
                        try:
                            size = int(cr.rsplit("/", 1)[1])
                        except ValueError:
                            pass
                if size is None:
                    cl = resp.headers.get("Content-Length")
                    if cl and cl.isdigit():
                        size = int(cl)
                return {"status": "ok", "http_status": resp.status,
                        "size_bytes": size, "final_url": resp.geturl(),
                        "method": method, "error": None}
        except urllib.error.HTTPError as e:
            if method == "HEAD" and e.code in (403, 405, 501):
                continue  # retry with a 1-byte ranged GET
            return {"status": "http_error", "http_status": e.code,
                    "size_bytes": None, "final_url": url, "method": method,
                    "error": str(e)}
        except Exception as e:  # DNS, TLS, timeout, blocked host
            if method == "HEAD":
                continue
            return {"status": "unreachable", "http_status": None,
                    "size_bytes": None, "final_url": url, "method": method,
                    "error": f"{type(e).__name__}: {e}"}
    return {"status": "unreachable", "size_bytes": None, "error": "no response"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe V2 source URLs and refresh the source catalog")
    ap.add_argument("--config", default=None)
    ap.add_argument("--metadata", default=None, help="override source_metadata.json path")
    ap.add_argument("--workdir", default=None, help="override configured workdir (logs/report)")
    ap.add_argument("--probe", action="store_true", help="make metadata-only HTTP requests")
    ap.add_argument("--offline", action="store_true", help="skip all network access")
    ap.add_argument("--timeout", type=float, default=12.0)
    ap.add_argument("--export-only", action="store_true", help="only regenerate CSVs")
    ap.add_argument("--min-free-gb", type=float, default=None,
                    help="override configured disk-space safety threshold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    meta_path = Path(args.metadata) if args.metadata else config_source_metadata(cfg)
    workdir = Path(args.workdir) if args.workdir else Path(cfg["workdir"])
    logger = configure_logging(workdir, "dataset_v2.discover")
    min_free = args.min_free_gb if args.min_free_gb is not None else float(cfg.get("min_free_gb", 20))
    require_free_space(workdir, min_free, logger)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    checked = 0
    skipped = 0
    if args.probe and not args.offline and not args.export_only:
        for s in meta["sources"]:
            if not s.get("download"):
                continue
            for f in s.get("files", []):
                if f.get("format") == "github_api_7z":
                    # The API listing is small; still use a metadata-only GET.
                    result = probe_url(f["url"] + "&per_page=100", args.timeout)
                else:
                    result = probe_url(f["url"], args.timeout)
                f["remote_probe"] = {**result, "checked_at_utc": now_utc()}
                if result.get("size_bytes") is not None:
                    f["remote_size_bytes"] = result["size_bytes"]
                checked += 1
                logger.info("[probe] %-42s %-10s size=%s", s["id"], result["status"],
                            result.get("size_bytes"))
    else:
        skipped = sum(len(s.get("files", [])) for s in meta["sources"] if s.get("download"))
        logger.info("offline/export-only: skipped %d remote probes", skipped)

    meta["last_discovery_utc"] = now_utc()
    meta["discovery_mode"] = ("export_only" if args.export_only else
                              "offline" if args.offline else
                              "remote_probe" if args.probe else "no_probe")
    # Keep the tracked source catalog immutable during normal launcher runs.
    probe_path = workdir / "source_metadata_probe.json"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(probe_path, json_dumps(meta))
    if args.export_only:
        _load_builder().export_csvs(meta)

    report = {
        "generated_at_utc": now_utc(),
        "mode": meta["discovery_mode"],
        "urls_checked": checked,
        "urls_skipped": skipped,
        "sources": len(meta["sources"]),
        "downloadable_sources": sum(1 for s in meta["sources"] if s.get("download")),
        "probe_output": str(probe_path),
        "note": ("Metadata-only. No corpus file was downloaded. Probe results are written "
                 "under --workdir; tracked source_metadata.json is not overwritten. "
                 "Derived CSVs are regenerated only with --export-only."),
    }
    out = workdir / "discovery_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    logger.info("discovery report: %s", out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
