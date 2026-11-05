#!/usr/bin/env bash
# Rebuild the IEEE manuscript end to end: assets, PDF, number check.
#
#   ./paper/ieee/build.sh
#
# Reads only committed result files under outputs/ and experiments/, and
# writes only inside paper/ieee/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PPMI_PYTHON:-$HOME/envs/ppmi/bin/python}"
TEX="${TECTONIC:-$HOME/envs/tex/bin/tectonic}"

echo "==> figures, tables and derived numbers"
"$PY" "$HERE/make_assets.py"

echo "==> compiling main.tex"
if command -v "$TEX" >/dev/null 2>&1; then
    (cd "$HERE" && "$TEX" -X compile main.tex)
elif command -v latexmk >/dev/null 2>&1; then
    (cd "$HERE" && latexmk -pdf -interaction=nonstopmode main.tex)
else
    echo "no tectonic or latexmk found; upload paper/ieee/ to Overleaf instead" >&2
    exit 1
fi

echo "==> verifying every number in the text against its source file"
"$PY" "$HERE/verify_numbers.py"

echo
echo "built $HERE/main.pdf"
