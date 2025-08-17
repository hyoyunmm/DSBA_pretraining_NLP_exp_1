#!/usr/bin/env bash
# chmod +x run.sh -> ./run.sh

# docker run -it \
#  --name hyoyoon-pretrain-nlp-exp1 \
#  hyoyoon/pretrain-nlp:v1 \
#  bash

set -e

IMAGE_NAME="hyoyoon/pretrain-nlp"
TAG="v1"
CONTAINER_NAME="hyoyoon-pretrain-nlp-exp1"
HOST_PORT=8888
PROJECT_DIR="/home/jaeheekim/for_pretrain/hyoyoon/exp_1" #"$(dirname $(pwd))"

docker run -it \
  --name ${CONTAINER_NAME} \
  --gpus "device=0" \
  -e CUDA_VISIBLE_DEVICES=0 \
  --shm-size=16g \
  -p ${HOST_PORT}:8888 \
  -v "${PROJECT_DIR}":/workspace/exp_1 \
  -w /workspace/exp_1 \
  ${IMAGE_NAME}:${TAG} bash
# -lc "
#  set -e
#  source /opt/conda/etc/profile.d/conda.sh
#  conda env list | awk '{print \$1}' | grep -qx ${ENV_NAME} || conda create -n ${ENV_NAME} python=3.10 -y
   
#  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
