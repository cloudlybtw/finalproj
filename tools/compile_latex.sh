#!/bin/sh
# Compile a source file with project-local caches and unchanged IEEEtran settings.
set -eu
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ "$#" -lt 1 ]; then
    echo 'Usage: sh tools/compile_latex.sh report/report.tex [output-directory]' >&2
    exit 2
fi
INPUT_DIR=$(CDPATH= cd -- "$(dirname -- "$1")" && pwd)
INPUT_NAME=$(basename -- "$1")
OUT_DIR=${2:-"$INPUT_DIR"}
mkdir -p "$OUT_DIR"
OUT_DIR=$(CDPATH= cd -- "$OUT_DIR" && pwd)
if [ -x "$PROJECT_ROOT/tools/tectonic" ]; then
    LATEX_ENGINE="$PROJECT_ROOT/tools/tectonic"
elif command -v tectonic >/dev/null 2>&1; then
    LATEX_ENGINE=$(command -v tectonic)
else
    echo 'Install Tectonic 0.17.0 or run: python3 tools/setup_tectonic.py' >&2
    exit 1
fi
export TECTONIC_CACHE_DIR="$PROJECT_ROOT/tools/tectonic-cache"
cd "$INPUT_DIR"
exec "$LATEX_ENGINE" --keep-logs --keep-intermediates --outdir "$OUT_DIR" "$INPUT_NAME"
