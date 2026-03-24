#!/bin/bash
# Visualize sequential failures for a given set in FiftyOne.
# Usage: bash view_seq_failures.sh [train|val|test]

SET=${1:-val}

case "$SET" in
  train)
    DATA_DIR="./data/01_model_input/sequential_train_val/train"
    FAILURES_DIR="failures_train"
    LABELS_DIR="predictions_labels_train"
    ;;
  val)
    DATA_DIR="./data/01_model_input/sequential_train_val/val"
    FAILURES_DIR="failures_val_seq"
    LABELS_DIR="predictions_labels_val"
    ;;
  test)
    DATA_DIR="./data/test/sequential_test/test"
    FAILURES_DIR="failures_test"
    LABELS_DIR="predictions_labels_test"
    ;;
  *)
    echo "Usage: $0 [train|val|test]"
    exit 1
    ;;
esac

echo "Visualizing $SET failures from $FAILURES_DIR (first 15 frames per sequence)..."
uv run python view_failures.py \
  --failures-dir "$FAILURES_DIR" \
  --data-dir "$DATA_DIR" \
  --labels-dir "$LABELS_DIR" \
  --max-frames 15
