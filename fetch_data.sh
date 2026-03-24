# #!/bin/bash
SSL_CERT_FILE=/Users/mateo/pyronear/vision/train/pyro-train/.venv/lib/python3.12/site-packages/certifi/cacert.pem \
uv run dvc get https://github.com/pyronear/pyro-dataset \
  data/processed/sequential_train_val/train \
  --rev v2.1.0 \
  --out ./data/01_model_input/sequential_train_val/train \

SSL_CERT_FILE=/Users/mateo/pyronear/vision/train/pyro-train/.venv/lib/python3.12/site-packages/certifi/cacert.pem \
uv run dvc get https://github.com/pyronear/pyro-dataset \
  data/processed/sequential_train_val/val \
  --rev v2.1.0 \
  --out ./data/01_model_input/sequential_train_val/val \


SSL_CERT_FILE=/Users/mateo/pyronear/vision/train/pyro-train/.venv/lib/python3.12/site-packages/certifi/cacert.pem \
uv run dvc get https://github.com/pyronear/pyro-dataset \
  data/processed/sequential_test \
  --rev v2.1.0 \
  --out ./data/test/sequential_test \


SSL_CERT_FILE=/Users/mateo/pyronear/vision/train/pyro-train/.venv/lib/python3.12/site-packages/certifi/cacert.pem \
uv run dvc get https://github.com/pyronear/pyro-dataset \
  data/processed/yolo_train_val \
  --rev v2.1.0 \
  --out ./data/01_model_input/yolo_train_val \

