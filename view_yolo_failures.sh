#!/bin/bash
# Visualize frame-by-frame YOLO failures in FiftyOne.
# Usage: bash view_yolo_failures.sh [train|val]

SET=${1:-val}

case "$SET" in
  train|val)
    ;;
  *)
    echo "Usage: $0 [train|val]"
    exit 1
    ;;
esac

FAILURES_DIR="failures_yolo_${SET}"
GT_LABELS_DIR="./data/01_model_input/yolo_train_val/labels/${SET}"

echo "Visualizing $SET failures from $FAILURES_DIR..."
uv run python view_yolo_val_failures.py \
  --failures-dir "$FAILURES_DIR" \
  --gt-labels-dir "$GT_LABELS_DIR"
