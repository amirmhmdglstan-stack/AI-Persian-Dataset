#!/usr/bin/env python3
"""Print dependency versions and check only the optional packages actually enabled.

Default Profile A needs only ``tokenizers``.  ``pyarrow`` is checked only when an
enabled source uses Parquet, and ``py7zr`` only when the optional V1 2021 7z
mirror is enabled.  The default launchers never auto-install optional packages.
This script is stdlib-only apart from the explicitly checked packages; it does
not import torch, numpy, pandas, tensorflow or model weights.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DOC_SEPARATOR, config_source_metadata, config_tokenizer, load_config,
)

CORE_PACKAGE = "tokenizers"
CORE_MIN = (0, 20)
CORE_MAX_EXCLUSIVE = (0, 24)
OPTIONAL_BY_SOURCE_TYPE = {
    "pyarrow": {"parquet"},
    "py7zr": {"github_api_7z", "7z", "csv_from_7z"},
}


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_version(value: str) -> tuple[int, ...]:
    parts = tuple(int(n) for n in re.findall(r"\d+", value)[:4])
    return parts or (0,)


def version_in_range(value: str | None) -> bool:
    if not value:
        return False
    v = parse_version(value)
    # (0, 23, 3) < (0, 24) is True; no downgrade to <0.23 is requested.
    return CORE_MIN <= v < CORE_MAX_EXCLUSIVE


def enabled_files(meta: dict, profiles: list[str], include_optional: bool,
                  only_sources: list[str] | None = None) -> list[tuple[dict, dict]]:
    out: list[tuple[dict, dict]] = []
    for s in meta.get("sources", []):
        if not s.get("download") or s.get("profile") not in profiles:
            continue
        if only_sources and s.get("id") not in only_sources:
            continue
        for f in s.get("files", []):
            if f.get("required") is False and not include_optional:
                continue
            out.append((s, f))
    return out


def required_optional_packages(files: list[tuple[dict, dict]]) -> list[str]:
    needed: set[str] = set()
    for _s, f in files:
        markers = {
            str(f.get("format", "")).lower(),
            str(f.get("extractor", "")).lower(),
        }
        for package, triggers in OPTIONAL_BY_SOURCE_TYPE.items():
            if any(t in marker for t in triggers for marker in markers):
                needed.add(package)
    return sorted(needed)


def check_baseline_tokenizer(path: Path) -> dict:
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as e:
        raise RuntimeError("The 'tokenizers' package is required for exact token counting.") from e
    if not path.exists():
        raise FileNotFoundError(f"Baseline tokenizer not found: {path}")
    tok = Tokenizer.from_file(str(path))
    ids = tok.encode(DOC_SEPARATOR, add_special_tokens=False).ids
    special_id = tok.token_to_id(DOC_SEPARATOR)
    if special_id != 0 or ids != [0]:
        raise RuntimeError(
            f"Baseline tokenizer check failed: {DOC_SEPARATOR} id={special_id}, encoded ids={ids}; "
            "expected ID 0 and one single token.")
    return {"path": str(path), "vocab_size": tok.get_vocab_size(),
            "special_token": DOC_SEPARATOR, "special_token_id": special_id,
            "special_token_is_single_token": ids == [0]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Check V2 core and enabled optional dependencies")
    ap.add_argument("--config", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--tokenizer-baseline", default=None)
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--only-sources", nargs="*", default=None)
    ap.add_argument("--include-optional", action="store_true",
                    help="check optional 7z/Parquet sources too")
    ap.add_argument("--skip-baseline-check", action="store_true",
                    help="only print versions; do not load the V1 32K tokenizer")
    args = ap.parse_args()

    cfg = load_config(args.config)
    meta_path = Path(args.metadata) if args.metadata else config_source_metadata(cfg)
    tok_path = Path(args.tokenizer_baseline) if args.tokenizer_baseline else config_tokenizer(cfg)
    profiles = args.profiles or cfg.get("default_profiles", ["A"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    files = enabled_files(meta, profiles, args.include_optional, args.only_sources)
    optional_needed = required_optional_packages(files)

    print("Persian Dataset V2 requirements check")
    print(f"  Python                 : {sys.version.split()[0]} ({sys.platform})")
    print(f"  Profiles               : {', '.join(profiles)}")
    print(f"  Repository metadata    : {meta_path}")
    print(f"  Enabled source files   : {len(files)}")
    print(f"  Optional packages      : {', '.join(optional_needed) if optional_needed else 'none required for enabled sources'}")

    core_version = installed_version(CORE_PACKAGE)
    if not core_version:
        print(f"  {CORE_PACKAGE:<23}: MISSING")
        print("ERROR: install core requirements first:")
        print("  D:\\GPT\\venv\\Scripts\\python.exe -m pip install -r dataset_v2\\requirements.txt")
        return 1
    range_ok = version_in_range(core_version)
    print(f"  {CORE_PACKAGE:<23}: {core_version} "
          f"[supported >=0.20,<0.24: {'OK' if range_ok else 'OUT OF RANGE'}]")
    if not range_ok:
        print("ERROR: installed tokenizers is outside the supported range; do not downgrade 0.23.3.")
        return 1

    failures = []
    for package in optional_needed:
        version = installed_version(package)
        if version:
            print(f"  {package:<23}: {version} [enabled-source requirement: OK]")
        else:
            print(f"  {package:<23}: MISSING [required by an enabled source]")
            failures.append(package)

    if not args.skip_baseline_check:
        try:
            baseline = check_baseline_tokenizer(tok_path)
            print(f"  V1 32K baseline        : OK vocab={baseline['vocab_size']}, "
                  f"{baseline['special_token']} ID {baseline['special_token_id']} single-token")
        except Exception as e:
            print(f"  V1 32K baseline        : FAILED ({e})")
            failures.append("baseline_tokenizer")

    print("  Heavyweight packages   : not installed, not imported, not required")
    if failures:
        print("ERROR: missing enabled requirements: " + ", ".join(failures))
        print("Install optional packages explicitly only when needed; default launchers do not auto-install them.")
        return 1
    print("REQUIREMENTS CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
