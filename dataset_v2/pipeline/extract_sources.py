#!/usr/bin/env python3
"""Streaming source extraction for Persian Dataset V2.

Outputs one JSONL document stream per source under
``<workdir>/extracted/<PROFILE>/<source_id>.jsonl``.  Each record retains:
source_id, doc_id, title, text, profile, license, domain and a metadata object
with per-document provenance/attribution.  No synthetic documents are created;
short natural sentences stay one document each and unrelated works are NOT
stitched together.

Supported extractors (all pure Python / stdlib unless noted):
  mediawiki_xml_bz2, txt_lines, csv, tsv, jsonl, jsonl_gz,
  tatoeba_detailed (Persian-only `pes_sentences_detailed.tsv.bz2`, or the real
  all-language `sentences_detailed.tar.bz2` -> `sentences_detailed.csv` member),
  zip_txt_tree (commit-pinned GitHub tarball), parsinlu_suite,
  csv_from_7z (py7zr optional), parquet (pyarrow optional).

Wikisource pages are accepted only through an exact configurable licence
template allowlist. Tatoeba rows use MySQL OUTFILE tab-separated escapes;
contributor usernames stay in provenance metadata and never enter training text.
"""
from __future__ import annotations

import argparse
import bz2
import csv
import fnmatch
import gzip
import html
import io
import json
import os
import re
import sys
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    atomic_write_json, config_policy, config_source_metadata, configure_logging, load_config,
    now_utc, require_free_space, sha256_file, stable_hash,
)

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

# Template names are only *candidates* for classification; acceptance uses an
# exact normalised allowlist from configs/source_policy.json.  There is no broad
# "free licence" prefix regex: NC/ND, generic Copyright/مجوز and unknown
# licence-like templates must be rejected with an exact reason.
RE_TEMPLATE_NAME = re.compile(r"\{\{\s*([^{}|}]+?)\s*(?:\||\}\})")
RE_DISAMBIG = re.compile(r"\{\{\s*(?:ابهام\u200c?زدایی|disambig)", re.IGNORECASE)

TATOEBA_DETAILED_FIELDS = (
    "id", "lang", "text", "username", "date_added", "date_last_modified",
)
TATOEBA_ALL_LANGUAGE_MEMBER = "sentences_detailed.csv"


def _normalize_template_name(name: str) -> str:
    n = name.strip()
    if n.lower().startswith("subst:"):
        n = n[6:]
    n = n.lower().replace("\u200c", "-").replace("_", "-").replace(" ", "-")
    return re.sub(r"-+", "-", n).strip("-")


def _iter_template_names(raw: str) -> Iterator[str]:
    for match in RE_TEMPLATE_NAME.finditer(raw):
        name = (match.group(1) or "").strip()
        if name:
            yield name


def classify_wikisource_license(raw: str, policy: dict) -> tuple[str, dict]:
    """Return ("accepted"/"rejected"/"no_template", details).

    Conservative order: ignore non-licence container templates, identify
    licence-like template names, reject any NC/ND/incompatible/generic marker,
    then accept only an exact allowlist match.  Unknown licence-like templates
    are rejected with ``unverified_template``.  This is intentionally not
    relaxed when the corpus is small.
    """
    cfg = policy.get("license_allowlist", {})
    allow = {_normalize_template_name(t["name"]): t
             for t in cfg.get("templates", [])}
    ignored = {_normalize_template_name(x)
               for x in cfg.get("ignored_template_names", [])}
    candidate_patterns = [re.compile(p, re.IGNORECASE)
                          for p in cfg.get("candidate_name_patterns", [])]
    reject_patterns = [(re.compile(r["pattern"], re.IGNORECASE), r["reason"])
                       for r in cfg.get("reject_patterns", [])]

    license_like: list[tuple[str, str]] = []
    for original in _iter_template_names(raw):
        norm = _normalize_template_name(original)
        if norm in ignored:
            continue
        if candidate_patterns and not any(p.search(norm) for p in candidate_patterns):
            continue
        license_like.append((original, norm))

    if not license_like:
        return "no_template", {
            "template": "", "normalized_template": "none",
            "reason": "no_verified_license_template", "license_id": "",
        }
    for original, norm in license_like:
        for rx, reason in reject_patterns:
            if rx.search(norm):
                return "rejected", {
                    "template": original, "normalized_template": norm,
                    "reason": reason, "license_id": "",
                }
    for original, norm in license_like:
        if norm in allow:
            return "accepted", {
                "template": original, "normalized_template": norm,
                "reason": "exact_allowlist_match",
                "license_id": allow[norm]["license"],
                "allowlist_version": cfg.get("version", "unknown"),
            }
    original, norm = license_like[0]
    return "rejected", {
        "template": original, "normalized_template": norm,
        "reason": "unverified_template", "license_id": "",
    }


def parse_mysql_outfile_fields(line: str) -> list[str | None]:
    """Parse one Tatoeba tab-separated MySQL OUTFILE record.

    Tatoeba's weekly export is written by MySQL ``SELECT ... INTO OUTFILE``:
    fields end at a tab, records at a line feed; ``\\\\``, ``\\t`` and ``\\n``
    are backslash escapes, ``\\0`` is NUL and ``\\N`` is SQL NULL.  Any other
    escape is rejected rather than silently mis-parsed.  This is NOT Python's
    :mod:`csv` quoting format.
    """
    fields: list[str | None] = []
    buf: list[str] = []
    null_field = False
    field_start = True
    i = 0
    escapes = {"\\": "\\", "t": "\t", "n": "\n", "0": "\x00"}
    while i < len(line):
        c = line[i]
        if null_field and c != "\t":
            raise ValueError("\\N must occupy a whole field")
        if c == "\\":
            if i + 1 >= len(line):
                raise ValueError("unterminated MySQL OUTFILE escape")
            nxt = line[i + 1]
            if nxt == "N":
                if not field_start:
                    raise ValueError("\\N must occupy a whole field")
                null_field = True
                field_start = False
                i += 2
                continue
            if nxt not in escapes:
                raise ValueError(f"invalid MySQL OUTFILE escape \\{nxt}")
            buf.append(escapes[nxt])
            field_start = False
            i += 2
            continue
        if c == "\t":
            fields.append(None if null_field else "".join(buf))
            buf = []
            null_field = False
            field_start = True
            i += 1
            continue
        buf.append(c)
        field_start = False
        i += 1
    fields.append(None if null_field else "".join(buf))
    return fields

RE_POEM_HEADER = re.compile(r"^(?:رباعی|غزل|قصیده|مثنوی|قطعه|شماره|بیت)\b.{0,80}$")
RE_REF = re.compile(r"<ref[^>]{0,200}>.*?</ref>|<ref[^>]{0,200}/>", re.DOTALL | re.IGNORECASE)
RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
RE_WIKI_FILE = re.compile(r"\[\[(?:پرونده|تصویر|File|Image|رده|Category):[^\]]{0,500}\]\]", re.IGNORECASE)
RE_WIKI_LINK = re.compile(r"\[\[(?:[^\]|]{0,250}\|)?([^\]]{0,350})\]\]")
RE_EXTERNAL_LINK = re.compile(r"\[(?:https?://|www\.)\S+?\s+([^\]]{1,350})\]", re.IGNORECASE)
RE_HTML_TAG = re.compile(r"</?[A-Za-z][^<>]{0,300}?>")
RE_LANG_LINK = re.compile(r"\[\[[a-z]{2,3}(?:-[a-z]+)?:[^\]]{0,200}\]\]", re.IGNORECASE)
RE_HEADING = re.compile(r"^\s*={2,6}\s*(.*?)\s*={2,6}\s*$", re.MULTILINE)
RE_MAGIC = re.compile(r"__[A-Z]{3,12}__")
RE_LINE_NOISE = re.compile(r"^\s*[|*#:;=\-–—_]{2,}\s*$")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_text(elem: ET.Element, path: str) -> str:
    child = elem.find(path)
    return (child.text or "") if child is not None else ""


def _strip_templates(text: str) -> str:
    out, depth = [], 0
    i = 0
    while i < len(text):
        if text.startswith("{{", i):
            depth += 1
            i += 2
            continue
        if text.startswith("}}", i) and depth:
            depth -= 1
            i += 2
            if depth == 0 and out and not out[-1].endswith(("\n", " ")):
                out.append(" ")
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    if depth:
        out.append(" ")
    return "".join(out)


def wikitext_to_text(raw: str) -> str:
    """Conservative wikitext -> plain text.  Piped links keep ONLY the label."""
    text = RE_REF.sub(" ", raw)
    text = RE_HTML_COMMENT.sub(" ", text)
    text = RE_TABLE.sub(" ", text)
    text = RE_WIKI_FILE.sub(" ", text)
    text = RE_LANG_LINK.sub(" ", text)
    text = _strip_templates(text)
    text = RE_WIKI_LINK.sub(r"\1", text)
    text = RE_EXTERNAL_LINK.sub(r"\1", text)
    text = RE_HTML_TAG.sub(" ", text)
    text = html.unescape(text)
    text = RE_HEADING.sub(r"\1", text)
    text = RE_MAGIC.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*[*#:;]+\s*", "", line)
        line = re.sub(r"'''''|'''|''", "", line)
        line = line.strip()
        if RE_LINE_NOISE.match(line) or not line:
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class ExtractContext:
    source: dict
    options: dict
    raw_paths: list[Path]
    parts_dir: Path | None
    workdir: Path
    logger: Any
    policy: dict | None = None
    max_docs: int | None = None
    stats: dict = field(default_factory=lambda: {
        "raw_records": 0, "documents": 0, "characters": 0,
        "skipped_by_reason": {}, "license_template_rejections": {},
    })

    def skip(self, reason: str) -> None:
        d = self.stats["skipped_by_reason"]
        d[reason] = d.get(reason, 0) + 1

    def record_license_rejection(self, status: str, info: dict) -> str:
        """Return an exact skip key and retain the actual matched template."""
        template = info.get("normalized_template") or "none"
        reason = info.get("reason") or "unverified_template"
        key = f"license_template_{status}:{template}:{reason}"
        d = self.stats.setdefault("license_template_rejections", {})
        entry = d.setdefault(key, {
            "count": 0, "status": status,
            "template": info.get("template", ""),
            "normalized_template": template, "reason": reason,
        })
        entry["count"] += 1
        return key

    def record(self, raw_id: str, text: str, *, title: str = "",
               meta: dict | None = None) -> dict:
        text = (text or "").strip()
        if not text:
            self.skip("empty_text")
            return {}
        source_id = self.source["id"]
        m = dict(meta or {})
        m.setdefault("source_id", source_id)
        m.setdefault("profile", self.source["profile"])
        m.setdefault("license", self.source["license_content"])
        m.setdefault("domain", self.source["domain"])
        m.setdefault("natural_text_only", True)
        m.setdefault("raw_doc_id", raw_id)
        return {
            "source_id": source_id,
            "doc_id": f"{source_id}:{raw_id}",
            "title": (title or "").strip(),
            "text": text,
            "profile": self.source["profile"],
            "license": self.source["license_content"],
            "domain": self.source["domain"],
            "content_rights_verified": bool(self.source["content_rights_verified"]),
            "meta": m,
        }


def _lines_from_binary(fileobj) -> Iterator[str]:
    for raw in fileobj:
        yield raw.decode("utf-8", errors="replace").rstrip("\r\n")


def _open_text(path: Path) -> Iterator[str]:
    with open(path, "rb") as f:
        yield from _lines_from_binary(f)


def _iter_tar_files(path: Path) -> Iterator[tuple[str, Any]]:
    """Stream tar/tar.gz/tar.bz2 members; never materializes the archive."""
    with tarfile.open(path, mode="r|*") as tf:
        for member in tf:
            if not member.isfile():
                continue
            f = tf.extractfile(member)
            if f is not None:
                yield member.name, f


def _delimited_rows(lines: Iterator[str], options: dict, ctx: ExtractContext):
    delimiter = options.get("delimiter", "\t")
    columns = options.get("columns")
    reader = csv.reader(lines, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
    if options.get("has_header"):
        header = next(reader, None)
        if not columns:
            columns = header
    for lineno, row in enumerate(reader, 1):
        ctx.stats["raw_records"] += 1
        if not row:
            ctx.skip("empty_row")
            continue
        if columns and len(row) < len(columns):
            ctx.skip("short_row")
            continue
        d = {columns[i] if columns else str(i): row[i] for i in range(len(row))}
        yield lineno, d, row


# --------------------------------------------------------------------------- #
# Extractors
# --------------------------------------------------------------------------- #

def extract_mediawiki_xml_bz2(ctx: ExtractContext) -> Iterator[dict]:
    opts = ctx.options
    policy = ctx.policy or {}
    require_license = bool(opts.get("require_free_license_template", False))
    page_base = opts.get("page_base_url") or ctx.source.get("page_base_url")
    if not page_base:
        raise RuntimeError(
            f"MediaWiki source {ctx.source['id']} has no page_base_url. "
            "Refusing to assume https://fa.wikipedia.org/wiki/ for Wikibooks, "
            "Wikisource or Wikinews pages.")
    page_base = str(page_base).rstrip("/") + "/"
    for path in ctx.raw_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing MediaWiki dump: {path}")
        ctx.logger.info("streaming MediaWiki XML: %s (pages: %s)", path.name, page_base)
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as f:
            context = ET.iterparse(f, events=("start", "end"))
            root = None
            for event, elem in context:
                if event == "start" and root is None:
                    root = elem
                if event != "end" or _local(elem.tag) != "page":
                    continue
                ctx.stats["raw_records"] += 1
                title = _find_text(elem, "{*}title").strip().replace("_", " ")
                ns = _find_text(elem, "{*}ns").strip()
                page_id = _find_text(elem, "{*}id").strip()
                text_el = elem.find("{*}revision/{*}text")
                raw = (text_el.text or "") if text_el is not None else ""
                license_status = "not_required"
                license_info: dict = {}
                license_rejection = None
                if require_license:
                    license_status, license_info = classify_wikisource_license(raw, policy)
                    if license_status != "accepted":
                        license_rejection = ctx.record_license_rejection(
                            license_status, license_info)
                skip = None
                if ns and ns != str(opts.get("namespace", 0)):
                    skip = "non_article_namespace"
                elif elem.find("{*}redirect") is not None:
                    skip = "redirect"
                elif re.match(r"^(?:الگو|رده|پرونده|مدیا|ویکی\u200c?پدیا|کاربر|بحث|Template|Category|File):", title, re.IGNORECASE):
                    skip = "non_article_title"
                elif opts.get("exclude_disambiguation", True) and RE_DISAMBIG.search(raw):
                    skip = "disambiguation"
                elif license_rejection:
                    skip = license_rejection
                elif len(raw) < int(opts.get("min_raw_chars", 120)):
                    skip = "raw_too_short"
                if skip:
                    ctx.skip(skip)
                else:
                    body = wikitext_to_text(raw)
                    if len(body) < int(opts.get("min_raw_chars", 120)):
                        ctx.skip("cleaned_too_short")
                    else:
                        full = f"{title}\n\n{body}" if opts.get("include_title", True) else body
                        page_url = f"{page_base}{quote(title.replace(' ', '_'))}"
                        meta = {
                            "page_id": page_id, "namespace": ns or "0",
                            "raw_file": path.name,
                            "free_license_template": (license_status == "accepted"
                                                      if require_license else None),
                            "license_template": license_info.get("template", ""),
                            "license_template_normalized": license_info.get("normalized_template", ""),
                            "license_template_license_id": license_info.get("license_id", ""),
                            "license_verification_method": ("explicit_allowlist"
                                                           if require_license else "source_catalog"),
                            "license_allowlist_version": license_info.get("allowlist_version", ""),
                            "source_url": ctx.source.get("url", ""),
                            "dump_index_url": ctx.source.get("url", ""),
                            "page_base_url": page_base,
                            "page_url": page_url,
                            "attribution": ctx.source.get("copyright_holder", ""),
                        }
                        yield ctx.record(
                            f"{path.name}:{page_id or stable_hash(title, 'wiki')}",
                            full, title=title, meta=meta)
                elem.clear()
                if root is not None:
                    while len(root) and _local(root[0].tag) == "page":
                        del root[0]
    ctx.logger.info("MediaWiki pages skipped: %s", ctx.stats["skipped_by_reason"])

def extract_txt_lines(ctx: ExtractContext) -> Iterator[dict]:
    opts = ctx.options
    one_per_line = bool(opts.get("one_doc_per_line", True))
    block_chars = int(opts.get("block_chars", 2000))
    min_line = int(opts.get("min_line_chars", 10))
    for path in ctx.raw_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing text file: {path}")
        buf: list[str] = []
        buf_len = 0
        block_i = 0
        for lineno, line in enumerate(_open_text(path), 1):
            ctx.stats["raw_records"] += 1
            line = line.strip()
            if not line:
                continue
            if one_per_line:
                if len(line) < min_line:
                    ctx.skip("short_line")
                    continue
                yield ctx.record(f"{path.name}:{lineno}", line,
                                 meta={"raw_file": path.name, "line": lineno,
                                       "source_label": opts.get("source_label", "")})
            else:
                buf.append(line)
                buf_len += len(line) + 1
                if buf_len >= block_chars:
                    yield ctx.record(f"{path.name}:block{block_i}", "\n".join(buf),
                                     meta={"raw_file": path.name, "block": block_i})
                    buf, buf_len, block_i = [], 0, block_i + 1
        if buf:
            yield ctx.record(f"{path.name}:block{block_i}", "\n".join(buf),
                             meta={"raw_file": path.name, "block": block_i})


def extract_csv(ctx: ExtractContext) -> Iterator[dict]:
    opts = ctx.options
    for path in ctx.raw_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV: {path}")
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f, delimiter=opts.get("delimiter", ","))
            for lineno, row in enumerate(reader, 1):
                ctx.stats["raw_records"] += 1
                text = (row.get(opts["text_column"], "") or "").strip()
                if not text:
                    ctx.skip("empty_text_column")
                    continue
                title = (row.get(opts.get("title_column", ""), "") or "").strip()
                rid = (row.get(opts.get("id_column", ""), "") or str(lineno)).strip()
                meta = {k: row.get(k, "") for k in opts.get("meta_columns", [])}
                meta.update({"raw_file": path.name, "line": lineno})
                full = f"{title}\n\n{text}" if title else text
                yield ctx.record(f"{path.name}:{rid}", full, title=title, meta=meta)


def extract_tsv(ctx: ExtractContext) -> Iterator[dict]:
    opts = ctx.options
    for path in ctx.raw_paths:
        if ".tar." in path.name:
            sources = _iter_tar_files(path)
        else:
            def plain():
                with open(path, "rb") as f:
                    yield path.name, f
            sources = plain()
        for member_name, fileobj in sources:
            try:
                if not fnmatch.fnmatch(PurePosixPath(member_name).name,
                                       opts.get("member_pattern", "*")):
                    continue
                for lineno, d, row in _delimited_rows(_lines_from_binary(fileobj), opts, ctx):
                    text = str(d.get(opts.get("text_column", "text"), "")).strip()
                    if not text:
                        ctx.skip("empty_text_column")
                        continue
                    title = str(d.get(opts.get("title_column", ""), "") or "").strip()
                    rid = str(d.get(opts.get("id_column", "id"), "") or f"{lineno}").strip()
                    meta = {k: d.get(k, "") for k in opts.get("meta_columns", [])}
                    if ctx.source["id"] == "persian_poetry_ccbysa":
                        meta.update({"poet": d.get("poet", ""), "century": d.get("century", ""),
                                     "compilation_license": ctx.source.get("license_compilation"),
                                     "underlying_work_rights": "unverified; quarantine pending per-work review",
                                     "repository": ctx.source.get("url")})
                    meta.update({"raw_file": path.name, "member": member_name, "line": lineno})
                    yield ctx.record(f"{member_name}:{rid}", text, title=title, meta=meta)
            finally:
                fileobj.close()


def _tatoeba_columns(opts: dict) -> list[str]:
    columns = [("date_last_modified" if c == "date_modified" else c)
               for c in (opts.get("columns") or list(TATOEBA_DETAILED_FIELDS))]
    if len(columns) != 6 or set(columns) != set(TATOEBA_DETAILED_FIELDS):
        raise RuntimeError(
            "Tatoeba detailed exporter requires the six official columns: "
            + ", ".join(TATOEBA_DETAILED_FIELDS))
    return columns


def _tatoeba_language_code(opts: dict) -> str:
    code = opts.get("language_code") or opts.get("lang_code") or "pes"
    if code == "fas":
        raise RuntimeError(
            "Tatoeba's verified Persian export uses ISO 639-3 'pes', not 'fas'. "
            "Refusing to filter the obsolete 'fas' code.")
    if code != "pes":
        raise RuntimeError(f"Unsupported Tatoeba Persian language code {code!r}; expected 'pes'.")
    return code


def _tatoeba_record_from_line(line: str, lineno: int, ctx: ExtractContext,
                              opts: dict, path: Path, member_name: str) -> dict | None:
    ctx.stats["raw_records"] += 1
    line = line.rstrip("\r\n")
    if not line:
        ctx.skip("empty_row")
        return None
    try:
        fields = parse_mysql_outfile_fields(line)
    except ValueError:
        ctx.skip("invalid_mysql_outfile_escape")
        return None
    columns = _tatoeba_columns(opts)
    if len(fields) != len(columns):
        ctx.skip("tatoeba_field_count")
        return None
    row = {columns[i]: fields[i] for i in range(len(columns))}
    lang_code = _tatoeba_language_code(opts)
    lang = (row.get("lang") or "").strip()
    if lang != lang_code:
        ctx.skip("other_language")
        return None
    text = (row.get("text") or "").strip()
    if not text:
        ctx.skip("empty_text")
        return None
    sentence_id = (row.get("id") or str(lineno)).strip()
    username = (row.get("username") or "").strip() or "unknown"
    date_added = (row.get("date_added") or "").strip()
    date_last_modified = (row.get("date_last_modified") or "").strip()
    license_label = ctx.source.get("license_content", "CC BY 2.0 FR")
    sentence_url = f"https://tatoeba.org/sentences/show/{sentence_id}"
    meta = {
        "raw_file": path.name,
        "source_filename": path.name,
        "member": member_name,
        "line": lineno,
        "tatoeba_sentence_id": sentence_id,
        "tatoeba_language_code": lang,
        "tatoeba_username_for_attribution_only": username,
        "username_for_attribution_only": username,
        "tatoeba_date_added": date_added,
        "tatoeba_date_last_modified": date_last_modified,
        "attribution": (f"Tatoeba contributor {username}; sentence {sentence_id}; "
                        f"{license_label}; {sentence_url}; source file {path.name}"),
        "attribution_url": sentence_url,
        "attribution_required": True,
        "license_terms_url": "https://creativecommons.org/licenses/by/2.0/fr/",
        "terms_of_use_url": "https://tatoeba.org/en/terms_of_use",
        "attribution_policy": ("CC BY 2.0 FR author attribution required; contributor "
                               "usernames stay in provenance metadata and are never "
                               "written into training text. Some sentences may be CC0; "
                               "the detailed export does not mark them, so attribution "
                               "is conservatively retained for every row."),
        "attribution_in_training_text": False,
    }
    # text is ONLY the Persian sentence; no username, ID or licence boilerplate.
    return ctx.record(f"{path.name}:{sentence_id}", text, meta=meta)


def extract_tatoeba_detailed(ctx: ExtractContext) -> Iterator[dict]:
    """Stream the verified Persian-only detailed export, or the real all-language CSV archive.

    Preferred: ``pes_sentences_detailed.tsv.bz2`` from
    ``downloads.tatoeba.org/exports/per_language/pes/`` (bzip2 TSV, language
    code ``pes``).  Fallback: the all-language
    ``sentences_detailed.tar.bz2`` archive contains ONE tab-separated MySQL
    OUTFILE member named ``sentences_detailed.csv`` -- it is NOT a collection of
    imaginary per-language ``*fas*sentences*.tsv`` members.
    """
    opts = ctx.options
    lang_code = _tatoeba_language_code(opts)
    _tatoeba_columns(opts)
    per_language = [p for p in ctx.raw_paths
                    if p.name.endswith("_sentences_detailed.tsv.bz2")]
    fallback = [p for p in ctx.raw_paths
                if p.name.endswith("sentences_detailed.tar.bz2")]
    if per_language:
        chosen = per_language
        if fallback:
            ctx.logger.warning("preferring Persian-only %s; ignoring all-language fallback",
                               ", ".join(p.name for p in per_language))
    elif fallback:
        chosen = fallback
        ctx.logger.warning("using all-language fallback %s; filtering %s rows only",
                           ", ".join(p.name for p in fallback), lang_code)
    else:
        raise FileNotFoundError(
            "No verified Tatoeba detailed export found. Expected "
            "pes_sentences_detailed.tsv.bz2 or sentences_detailed.tar.bz2.")

    for path in chosen:
        if not path.exists():
            raise FileNotFoundError(f"Missing Tatoeba export: {path}")
        if path.name.endswith(".tsv.bz2"):
            ctx.logger.info("Tatoeba Persian-only bzip2 TSV: %s", path.name)
            with bz2.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
                for lineno, line in enumerate(f, 1):
                    rec = _tatoeba_record_from_line(
                        line, lineno, ctx, opts, path, path.name)
                    if rec:
                        yield rec
        elif path.name.endswith(".tar.bz2"):
            found = False
            for member_name, fileobj in _iter_tar_files(path):
                if PurePosixPath(member_name).name != TATOEBA_ALL_LANGUAGE_MEMBER:
                    continue
                found = True
                ctx.logger.info("Tatoeba all-language CSV member: %s", member_name)
                for lineno, line in enumerate(_lines_from_binary(fileobj), 1):
                    rec = _tatoeba_record_from_line(
                        line, lineno, ctx, opts, path, member_name)
                    if rec:
                        yield rec
                if hasattr(fileobj, "close"):
                    fileobj.close()
            if not found:
                ctx.skip("missing_sentences_detailed_csv_member")
                raise RuntimeError(
                    f"{path.name} does not contain {TATOEBA_ALL_LANGUAGE_MEMBER}. "
                    "Refusing to search for imaginary per-language TSV members "
                    "inside a CSV archive.")
        else:
            raise RuntimeError(f"Unsupported Tatoeba detailed export file: {path.name}")

def extract_jsonl(ctx: ExtractContext) -> Iterator[dict]:
    opts = ctx.options
    text_fields = opts.get("text_fields") or [opts.get("text_field", "text")]
    exclude_prefixes = tuple(opts.get("exclude_field_prefixes", ["query_", "judge_"]))
    for path in ctx.raw_paths:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                ctx.stats["raw_records"] += 1
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    ctx.skip("invalid_json")
                    continue
                if not isinstance(d, dict):
                    ctx.skip("non_object_json")
                    continue
                parts = []
                for field in text_fields:
                    if field.startswith(exclude_prefixes):
                        continue
                    val = d.get(field)
                    if isinstance(val, str) and val.strip():
                        parts.append(val.strip())
                text = "\n\n".join(parts)
                if not text:
                    ctx.skip("empty_text")
                    continue
                rid = str(d.get(opts.get("id_field", "id"), lineno))
                title = str(d.get(opts.get("title_field", "title"), "") or "")
                meta = {k: d.get(k) for k in opts.get("meta_fields", []) if k in d}
                meta.update({"raw_file": path.name, "line": lineno,
                             "excluded_synthetic_prefixes": list(exclude_prefixes)})
                yield ctx.record(f"{path.name}:{rid}", text, title=title, meta=meta)


def extract_parquet(ctx: ExtractContext) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Parquet extraction needs pyarrow. Install with: "
            "D:\\GPT\\venv\\Scripts\\python.exe -m pip install pyarrow") from e
    opts = ctx.options
    text_fields = opts.get("text_fields") or [opts.get("text_field", "text")]
    exclude_prefixes = tuple(opts.get("exclude_field_prefixes", ["query_", "judge_"]))
    for path in ctx.raw_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing Parquet file: {path}")
        pf = pq.ParquetFile(str(path))
        allowed = opts.get("allow_fields")
        if not allowed:
            allowed = [n for n in pf.schema_arrow.names
                       if not n.startswith(exclude_prefixes)]
        for batch in pf.iter_batches(batch_size=int(opts.get("batch_size", 1024)), columns=allowed):
            cols = batch.to_pydict()
            n = batch.num_rows
            for i in range(n):
                ctx.stats["raw_records"] += 1
                parts = []
                for field in text_fields:
                    if field in cols and cols[field][i]:
                        parts.append(str(cols[field][i]))
                text = "\n\n".join(parts).strip()
                if not text:
                    ctx.skip("empty_text")
                    continue
                row = {k: cols[k][i] for k in cols}
                rid = str(row.get(opts.get("id_field", "id"), f"{path.name}:{i}"))
                title = str(row.get(opts.get("title_field", "title"), "") or "")
                meta = {k: row.get(k) for k in opts.get("meta_fields", []) if k in row}
                meta.update({"raw_file": path.name, "row": i,
                             "excluded_synthetic_prefixes": list(exclude_prefixes)})
                yield ctx.record(f"{path.name}:{rid}", text, title=title, meta=meta)


def extract_zip_txt_tree(ctx: ExtractContext) -> Iterator[dict]:
    """Ganjoor-style commit-pinned tarball; one natural document per source file."""
    opts = ctx.options
    subdir = opts.get("subdir", "txt")
    allowed = set(opts.get("allowed_dirs", []))
    min_chars = int(opts.get("min_chars", 40))
    for path in ctx.raw_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing source tarball: {path}")
        for member_name, fileobj in _iter_tar_files(path):
            if not member_name.endswith(".txt"):
                continue
            parts = PurePosixPath(member_name).parts
            if subdir not in parts:
                continue
            idx = parts.index(subdir)
            book = parts[idx + 1] if len(parts) > idx + 1 else ""
            if allowed and book not in allowed:
                ctx.skip("book_not_in_public_domain_allowlist")
                continue
            ctx.stats["raw_records"] += 1
            lines = []
            for raw_line in _lines_from_binary(fileobj):
                ln = raw_line.strip()
                if ln and not RE_POEM_HEADER.match(ln):
                    lines.append(ln)
            text = "\n".join(lines).strip()
            if hasattr(fileobj, "close"):
                fileobj.close()
            if len(text) < min_chars:
                ctx.skip("short_source_file")
                continue
            yield ctx.record(f"{member_name}", text,
                             title=book,
                             meta={"raw_file": path.name, "member": member_name,
                                   "book": book,
                                   "attribution": "Ganjoor classical works (public domain); ganjoor-tex GPL-3.0 data files"})


def extract_parsinlu_suite(ctx: ExtractContext) -> Iterator[dict]:
    """Natural text fields from ParsiNLU; NO synthetic QA/option/label markers."""
    opts = ctx.options
    include = set(opts.get("include", ["reading_comprehension", "multiple_choice", "entailment"]))
    exclude_rc_passages = bool(opts.get("exclude_rc_passages", True))
    for path in ctx.raw_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing ParsiNLU tarball: {path}")
        for member_name, fileobj in _iter_tar_files(path):
            name = PurePosixPath(member_name).name
            full = member_name.lower()
            if "test_translation" in full and opts.get("exclude_translated_test", True):
                ctx.skip("machine_translated_test")
                continue
            lines = _lines_from_binary(fileobj)
            if "/reading_comprehension/" in full and "reading_comprehension" in include and name.endswith(".jsonl"):
                for i, line in enumerate(lines, 1):
                    ctx.stats["raw_records"] += 1
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        ctx.skip("invalid_json")
                        continue
                    if not exclude_rc_passages:
                        # Explicit opt-in only: RC passages are publisher news.
                        p = str(d.get("passage", "")).strip()
                        if p:
                            yield ctx.record(f"{member_name}:{i}:passage", p,
                                             meta={"subsource": "rc_passage_opt_in", "member": member_name})
                    q = str(d.get("question", "")).strip()
                    if q:
                        yield ctx.record(f"{member_name}:{i}:question", q,
                                         meta={"subsource": "rc_question", "member": member_name})
                    answers = d.get("answers") or []
                    if answers and isinstance(answers[0], (list, tuple)) and len(answers[0]) > 1:
                        ans = str(answers[0][1]).strip()
                        if ans:
                            yield ctx.record(f"{member_name}:{i}:answer", ans,
                                             meta={"subsource": "rc_answer", "member": member_name})
            elif "/multiple-choice/" in full and "multiple_choice" in include and name.endswith(".jsonl"):
                for i, line in enumerate(lines, 1):
                    ctx.stats["raw_records"] += 1
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        ctx.skip("invalid_json")
                        continue
                    q = str(d.get("question", "")).strip()
                    if q:
                        yield ctx.record(f"{member_name}:{i}:question", q,
                                         meta={"subsource": "mc_question", "member": member_name})
                    for j, cand in enumerate(d.get("candidates") or []):
                        c = str(cand).strip()
                        if c:
                            yield ctx.record(f"{member_name}:{i}:option{j}", c,
                                             meta={"subsource": "mc_option", "member": member_name})
            elif "/entailment/" in full and "entailment" in include and name.endswith(".tsv"):
                for i, row in enumerate(csv.reader(lines, delimiter="\t"), 1):
                    ctx.stats["raw_records"] += 1
                    if len(row) < 2:
                        ctx.skip("short_row")
                        continue
                    if "<sep>" in row[0] or len(row) == 2:
                        pair = row[0]
                        if "<sep>" not in pair:
                            continue
                        prem, hyp = pair.split("<sep>", 1)
                    else:
                        prem, hyp = row[0], row[1]
                    for j, sent in enumerate((prem.strip(), hyp.strip())):
                        if sent:
                            yield ctx.record(f"{member_name}:{i}:{j}", sent,
                                             meta={"subsource": "entailment_sentence", "member": member_name})
            if hasattr(fileobj, "close"):
                fileobj.close()


def extract_csv_from_7z(ctx: ExtractContext) -> Iterator[dict]:
    """Concatenate split 7z parts, extract with py7zr, stream the inner CSV."""
    try:
        import py7zr  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "7z extraction needs py7zr. Install with: "
            "D:\\GPT\\venv\\Scripts\\python.exe -m pip install py7zr") from e
    opts = ctx.options
    page_base = opts.get("page_base_url") or ctx.source.get("page_base_url")
    if not page_base:
        raise RuntimeError(
            f"7z mirror source {ctx.source['id']} has no page_base_url; "
            "refusing to assume a Wikipedia domain.")
    page_base = str(page_base).rstrip("/") + "/"
    parts_dir = ctx.parts_dir
    if parts_dir is None or not parts_dir.exists():
        raise FileNotFoundError(f"Missing split 7z parts directory: {parts_dir}")
    parts = sorted(parts_dir.glob(opts.get("filename_pattern", "*.7z.*")))
    if not parts:
        raise FileNotFoundError(f"No 7z parts matched in {parts_dir}")
    joined = ctx.workdir / "tmp" / f"{ctx.source['id']}.7z"
    joined.parent.mkdir(parents=True, exist_ok=True)
    ctx.logger.info("joining %d 7z parts into %s", len(parts), joined)
    with open(joined, "wb") as out:
        for p in parts:
            with open(p, "rb") as inp:
                while True:
                    chunk = inp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
    extract_dir = ctx.workdir / "tmp" / f"{ctx.source['id']}_7z"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(str(joined), mode="r") as z:
        z.extractall(path=str(extract_dir))
    inner = extract_dir / opts.get("inner_member", "fawiki.csv")
    if not inner.exists():
        found = list(extract_dir.rglob(opts.get("inner_member", "fawiki.csv")))
        if not found:
            raise FileNotFoundError(f"{opts.get('inner_member')} not found in 7z archive")
        inner = found[0]
    with open(inner, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=opts.get("delimiter", ","))
        for lineno, row in enumerate(reader, 1):
            ctx.stats["raw_records"] += 1
            text = (row.get(opts["text_column"], "") or "").strip()
            if not text:
                ctx.skip("empty_text_column")
                continue
            title = (row.get(opts.get("title_column", ""), "") or "").strip()
            rid = (row.get(opts.get("id_column", ""), "") or str(lineno)).strip()
            full = f"{title}\n\n{text}" if title else text
            yield ctx.record(f"{inner.name}:{rid}", full, title=title,
                             meta={"raw_file": inner.name, "line": lineno,
                                   "attribution": "Wikipedia contributors, CC BY-SA 4.0",
                                   "page_base_url": page_base,
                                   "page_url": f"{page_base}{quote(title.replace(' ', '_'))}"})


EXTRACTORS = {
    "mediawiki_xml_bz2": extract_mediawiki_xml_bz2,
    "txt_lines": extract_txt_lines,
    "csv": extract_csv,
    "tsv": extract_tsv,
    "jsonl": extract_jsonl,
    "jsonl_gz": extract_jsonl,
    "tatoeba_detailed": extract_tatoeba_detailed,
    "parquet": extract_parquet,
    "zip_txt_tree": extract_zip_txt_tree,
    "parsinlu_suite": extract_parsinlu_suite,
    "csv_from_7z": extract_csv_from_7z,
}


def source_raw_paths(source: dict, raw_root: Path, skip_missing: bool, logger) -> tuple[list[Path], Path | None, str | None]:
    paths, parts_dir, extractor = [], None, None
    missing_required = False
    files = source.get("files", [])
    if source["id"] == "tatoeba_fa_cc_by":
        # The two files are alternatives, never simultaneous inputs.
        files = [next((f for f in files if not f.get("fallback_only") and
                       (raw_root / source["id"] / f["filename"]).exists()), None)
                 or next(f for f in files if f.get("fallback_only"))]
    for f in files:
        ex = f["extractor"]
        extractor = extractor or ex
        if ex != extractor:
            raise RuntimeError(f"Source {source['id']} mixes extractors {extractor}/{ex}")
        if f.get("format") == "github_api_7z":
            candidate = raw_root / source["id"] / "parts"
            if candidate.exists():
                parts_dir = candidate
            elif f.get("required", True):
                if skip_missing:
                    missing_required = True
                else:
                    raise FileNotFoundError(f"Missing 7z parts directory: {candidate}")
        else:
            p = raw_root / source["id"] / f["filename"]
            if not p.exists():
                if skip_missing or not f.get("required", True):
                    logger.warning("missing raw file %s; required=%s", p, f.get("required", True))
                    if f.get("required", True):
                        missing_required = True
                    continue
                raise FileNotFoundError(f"Missing raw file for {source['id']}: {p}. Run download_sources.py first.")
            paths.append(p)
    if missing_required:
        return [], None, extractor  # never publish an incomplete multi-file source
    return paths, parts_dir, extractor


def input_identity(source, paths, parts_dir, extractor, options, policy):
    files = list(paths)
    if parts_dir:
        files += sorted(p for p in parts_dir.rglob("*") if p.is_file())
    return {
        "raw_inputs": [{"filename": str(p), "size": p.stat().st_size,
                        "sha256": sha256_file(p)} for p in files],
        "source": source, "extractor": extractor, "options": options,
        "policy": policy,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream V2 raw sources into per-source JSONL documents")
    ap.add_argument("--config", default=None)
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--extracted", default=None, help="override extracted output directory")
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--only-sources", nargs="*", default=None)
    ap.add_argument("--force", action="store_true", help="re-extract completed sources")
    ap.add_argument("--skip-missing", action="store_true",
                    help="log and skip missing files instead of failing (useful for optional sources)")
    ap.add_argument("--max-docs", type=int, default=None, help="cap documents per source (small tests only)")
    ap.add_argument("--min-free-gb", type=float, default=None,
                    help="override configured disk-space safety threshold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    meta_path = Path(args.metadata) if args.metadata else config_source_metadata(cfg)
    workdir = Path(args.workdir) if args.workdir else Path(cfg["workdir"])
    extracted_root = Path(args.extracted) if args.extracted else workdir / "extracted"
    raw_root = workdir / "raw"
    logger = configure_logging(workdir, "dataset_v2.extract")
    min_free = args.min_free_gb if args.min_free_gb is not None else float(cfg.get("min_free_gb", 20))
    require_free_space(workdir, min_free, logger)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    policy_path = config_policy(cfg)
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    profiles = args.profiles or cfg.get("default_profiles", ["A"])
    for p in profiles:
        if p not in policy["profiles"]:
            raise ValueError(f"Unknown profile {p!r}")
    manifest_path = workdir / "extract_manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {"version": "2.1.0", "sources": {}})
    manifest["version"] = "2.1.0"
    manifest["license_allowlist_version"] = policy.get("license_allowlist", {}).get("version")

    selected = [s for s in meta["sources"]
                if s["profile"] in profiles and s.get("download")
                and s["decision"] in policy["profiles"][s["profile"]]["allowed_decisions"]
                and (not args.only_sources or s["id"] in args.only_sources)]
    logger.info("selected %d sources for extraction, profiles=%s", len(selected), profiles)

    for s in selected:
        out_dir = extracted_root / s["profile"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{s['id']}.jsonl"
        paths, parts_dir, extractor = source_raw_paths(s, raw_root, args.skip_missing, logger)
        if not paths and parts_dir is None:
            manifest["sources"][s["id"]] = {"status": "missing_raw_files", "updated_at_utc": now_utc()}
            logger.warning("[skip] %s: all raw files missing", s["id"])
            continue
        if extractor not in EXTRACTORS:
            raise RuntimeError(f"No extractor named {extractor!r} for source {s['id']}")
        options: dict = {}
        for f in s.get("files", []):
            if s["id"] == "tatoeba_fa_cc_by" and f["filename"] != paths[0].name:
                continue
            options.update(f.get("options", {}))
        identity = input_identity(s, paths, parts_dir, extractor, options, policy)
        prior = manifest["sources"].get(s["id"])
        if (not args.force and args.max_docs is None and prior and
                prior.get("status") == "complete" and prior.get("input_identity") == identity
                and out_path.exists() and sha256_file(out_path) == prior.get("sha256")):
            logger.info("[skip] %s already extracted and inputs verified", s["id"])
            continue
        out_path.unlink(missing_ok=True)  # stale outputs must not be buildable
        ctx = ExtractContext(source=s, options=options, raw_paths=paths,
                             parts_dir=parts_dir, workdir=workdir, logger=logger,
                             policy=policy, max_docs=args.max_docs)
        tmp = out_path.with_name(out_path.name + ".part")
        docs = chars = 0
        logger.info("[extract] %s -> %s", s["id"], out_path)
        partial = False
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as out:
                gen = EXTRACTORS[extractor](ctx)
                try:
                    for rec in gen:
                        if not rec:
                            continue
                        text = rec["text"]
                        out.write(json.dumps(rec, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":")) + "\n")
                        docs += 1
                        chars += len(text)
                        if docs % int(cfg.get("log_every_docs", 10000)) == 0:
                            logger.info("  %s: %s docs", s["id"], f"{docs:,}")
                        if args.max_docs and docs >= args.max_docs:
                            partial = True  # even at EOF, conservative until iterator exhausted
                            break
                finally:
                    gen.close()
                out.flush()
                os.fsync(out.fileno())
        except BaseException:
            manifest["sources"][s["id"]] = {"status": "failed", "reason": "interrupted_or_extraction_error",
                                             "input_identity": identity}
            atomic_write_json(manifest_path, manifest)
            raise
        if not partial:
            os.replace(tmp, out_path)
        else:
            # A previous complete file is stale; do not expose it to the builder.
            out_path.unlink(missing_ok=True)
        checksum = sha256_file(tmp if partial else out_path)
        snapshot = None
        dl_manifest = workdir / "download_manifest.json"
        if dl_manifest.exists():
            downloaded = json.loads(dl_manifest.read_text(encoding="utf-8"))
            snapshot = [r for r in downloaded.get("sources", {}).get(s["id"], {}).get("files", [])
                        if any(r.get("filename") == p.name and r.get("sha256") == sha256_file(p)
                               for p in paths)]
        rec = {
            "download_snapshot": snapshot,
            "status": "partial" if partial else "complete",
            "reason": "max_docs" if partial else None,
            "profile": s["profile"], "decision": s["decision"],
            "license_content": s["license_content"],
            "output": str(tmp if partial else out_path),
            "documents": docs, "characters": chars, "sha256": checksum,
            "raw_files": [str(p) for p in paths], "input_identity": identity,
            "parts_dir": str(parts_dir) if parts_dir else None,
            "extractor": extractor, "options": options,
            "skipped_by_reason": ctx.stats["skipped_by_reason"],
            "license_template_rejections": ctx.stats.get("license_template_rejections", {}),
            "license_allowlist_version": policy.get("license_allowlist", {}).get("version"),
            "raw_records": ctx.stats["raw_records"],
            "truncated_by_max_docs": partial,
            "extracted_at_utc": now_utc(),
        }
        manifest["sources"][s["id"]] = rec
        tmp_manifest = manifest_path.with_name(manifest_path.name + ".part")
        tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
        os.replace(tmp_manifest, manifest_path)
        logger.info("  %s: %s docs / %s chars / sha256=%s", s["id"], f"{docs:,}",
                    f"{chars:,}", checksum[:12])

    manifest["generated_at_utc"] = now_utc()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    logger.info("extract manifest: %s", manifest_path)
    print(f"Extraction complete: {len(manifest['sources'])} source entries.")


if __name__ == "__main__":
    main()
