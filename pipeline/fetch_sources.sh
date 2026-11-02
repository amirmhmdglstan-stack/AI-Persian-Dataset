#!/usr/bin/env bash
# fetch_sources.sh — download every upstream corpus used by this dataset.
#
# Everything here is public on GitHub. Nothing is scraped from the live web, no
# access control is bypassed, and no personal data is collected. Each source's
# license and origin are recorded in sources.json.
#
# Usage:  WORKDIR=/tmp/work bash pipeline/fetch_sources.sh
#         (default WORKDIR: ./work — outside the repository, sources are large)
set -euo pipefail

WORKDIR="${WORKDIR:-./work}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

clone() {  # clone <url> <dirname>
  local url="$1" dir="$2"
  if [ -d "$dir" ]; then echo "[skip] $dir already present"; return; fi
  echo "[clone] $url"
  git clone --depth 1 --quiet "$url" "$dir"
}

# 1. Persian Wikipedia, cleaned article dump (865k articles, 2021-02-01 snapshot)
#    Content: Wikipedia, CC BY-SA 4.0.  The repository ships a split 7z archive.
if [ ! -f fawiki.csv ]; then
  clone https://github.com/hsmirzaie/fawiki.git fawiki
  cat fawiki/fawiki.7z.00* > fawiki.7z
  echo "[7z] extracting fawiki.csv (~1.1 GB, needs py7zr or 7z)"
  if command -v 7z >/dev/null 2>&1; then 7z x -y fawiki.7z >/dev/null
  else python3 -c "import py7zr;py7zr.SevenZipFile('fawiki.7z').extractall('.')"; fi
fi

# 2. Classical Persian literature (Ganjoor), plain text (public-domain works)
clone https://github.com/ganjoor/ganjoor-tex.git ganjoor-tex

# 3. Chronological Persian poetry corpus (CC BY-SA 4.0)
clone https://github.com/aghasemi/ChronologicalPersianPoetryDataset.git ChronologicalPersianPoetryDataset

# 4. ParsiNLU: reading comprehension, multiple-choice, textual entailment
clone https://github.com/persiannlp/parsinlu.git parsinlu

# 5. PeQa: Persian chat/dialogue pairs (MIT)
clone https://github.com/ZahraArshia/PeQa.git PeQa

# 6. Persian news corpus, 13.5k articles (title + summary) from 5 agencies
clone https://github.com/milad-4274/persian_news.git persian_news

# 7. pn-summary: Iranian news articles with human-written summaries
clone https://github.com/hooshvare/pn-summary.git pn-summary

echo
echo "All sources are in: $(pwd)"
echo "Next: python3 pipeline/extract_sources.py --workdir $(pwd) --out $(pwd)/extracted"
