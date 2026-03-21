#!/usr/bin/env bash
set -euo pipefail

BASE="/Users/yadnesh/cybba/cybba_segment_automation"
BACKEND="$BASE/backend"
CFG="$BACKEND/config/config.yml"
DB="$BACKEND/data/runs.db"
TRAIN_GZ="$BASE/Data/Input/Raw_segments/taxonomy_training_prepared.csv.gz"

cd "$BACKEND"

echo "[1/2] Collect new generated segments into training file..."
python3 src/collect_generated_training_rows.py \
  --config "$CFG" \
  --db "$DB" \
  --out "$TRAIN_GZ"

echo "[2/2] Train taxonomy model on updated training file..."
python3 -m src.train_taxonomy_model \
  --config "$CFG" \
  --train-file "taxonomy_training_prepared.csv.gz"

echo "[OK] Nightly training done."
