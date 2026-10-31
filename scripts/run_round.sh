#!/bin/bash
# One data round for a task: convert the crawl, download the selected audio, extract features.
# Every step is resumable: re-run the same command after an interruption.
#
#   CRAWL_DIR=/path/to/out_answers scripts/run_round.sh read_aloud data/alfa data/alfa_r1 cache/alfa_r1
#
# args: TASK  POOL_DIR (all answers + audio/)  ROUND_DIR (selected answers)  CACHE_DIR
# Before running: scripts/sample_alfa.py POOL_DIR ... writes POOL_DIR/selected.csv, and ROUND_DIR/labels.csv
# is that selection with audio_path pointing into POOL_DIR (see docs/data_pipeline.md).
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=$1 POOL=$2 ROUND=$3 CACHE=$4
: "${CRAWL_DIR:?set CRAWL_DIR to the crawl output folder (contains <task>/<question_id>/...)}"
.venv/bin/python scripts/prepare_alfa.py "$CRAWL_DIR" "$POOL" --task "$TASK" \
  --download 1000000 --selected "$ROUND/labels.csv" --workers 8
.venv/bin/python -m pte_scorer extract "$ROUND" --out "${CACHE}_features.parquet" --cache "$CACHE"
echo "ROUND_DONE $TASK"
