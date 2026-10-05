#!/usr/bin/env bash
# docs/report.md -> docs/report.pdf (A4) with pandoc + headless Chrome. Usage: scripts/build_report.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
command -v google-chrome >/dev/null && CHROME=google-chrome
tmp=$(mktemp -d)
pandoc docs/report.md -s --metadata title="VORA technical report" --css report.css -o "$tmp/report.html"
cp docs/report.css "$tmp/"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/docs/report.pdf" "file://$tmp/report.html" 2>/dev/null
rm -rf "$tmp"
echo "docs/report.pdf: $(pdfinfo docs/report.pdf | awk '/^Pages/ {print $2}') pages"
