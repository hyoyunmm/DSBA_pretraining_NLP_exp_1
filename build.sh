#!/usr/bin/env bash
# chmod +x build.sh
# ./build.sh

#python - <<'PY'
#import torch
#print("cuda available:", torch.cuda.is_available())
#print("device count:", torch.cuda.device_count())
#print("current device:", torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
#print("name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
#PY


set -e

IMAGE_NAME="hyoyoon_pretrain_nlp_1"
TAG="v1"

docker build -t ${IMAGE_NAME}:${TAG} .
echo "Built image: ${IMAGE_NAME}:${TAG}"
