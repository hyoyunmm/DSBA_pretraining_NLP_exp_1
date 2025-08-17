#!/usr/bin/env bash
# (맥터미널)
# chmod +x scripts/run_all.sh && bash scripts/run_all.sh
set -euo pipefail

# 프로젝트 루트로 이동
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"

mkdir -p logs configs
[ -f logs/status.tsv ] || echo -e "run_id\tstatus" > logs/status.tsv
[ -f src/__init__.py ] || touch src/__init__.py

export TOKENIZERS_PARALLELISM=false
PY=python

# [모델 | pooling | max_len | batch | dropout | weight_decay | grad_clip | seed]
EXPS=(
  "bert-base-uncased|cls|128|16|0.1|0.0|0.0|42"
  "bert-base-uncased|mean|128|16|0.1|0.0|0.0|42"
  "bert-base-uncased|cls|256|16|0.1|0.0|0.0|42"
  "bert-base-uncased|cls|128|16|0.3|0.0|0.0|42"
  "bert-base-uncased|cls|128|32|0.1|0.01|1.0|42"

  "answerdotai/ModernBERT-base|cls|128|16|0.1|0.0|0.0|42"
  "answerdotai/ModernBERT-base|mean|128|16|0.1|0.0|0.0|42"
  "answerdotai/ModernBERT-base|cls|256|16|0.1|0.0|0.0|42"
  "answerdotai/ModernBERT-base|cls|128|16|0.3|0.0|0.0|42"
  "answerdotai/ModernBERT-base|cls|128|32|0.1|0.01|1.0|42"
)

EPOCHS=5
LR=5e-5
NUM_WORKERS=2
LOG_EVERY=50
STATUS="logs/status.tsv"

for exp in "${EXPS[@]}"; do
  IFS='|' read -r MODEL POOL L BS DROP WD GC SEED <<< "$exp"
  mslug="${MODEL//\//-}"
  run_id="m=${mslug}_p=${POOL}_L=${L}_bs=${BS}_drop=${DROP}_wd=${WD}_gc=${GC}_seed=${SEED}"
  cfg="configs/${run_id}.yaml"

  cat > "${cfg}" <<EOF
seed: ${SEED}
data:
  model_name: ${MODEL}
  max_len: ${L}
  seed: ${SEED}
  train_frac: 0.8
  val_frac: 0.1
  test_frac: 0.1
  batch_size: ${BS}
  num_workers: ${NUM_WORKERS}
model:
  model_name: ${MODEL}
  dropout: ${DROP}
  pooling: ${POOL}
  num_labels: 2
train_config:
  epochs: ${EPOCHS}
  lr: ${LR}
  weight_decay: ${WD}
  grad_clip: ${GC}
  device: auto
  optimizer: adam
  scheduler: constant
  log_every: ${LOG_EVERY}
  output_dir: outputs/${run_id}
  deterministic: false
logging:
  use_wandb: true
  project: nlp-sent-cls
  run_name: ${run_id}
EOF

  echo "=== RUN ${run_id} ==="
  if ${PY} -m src.main --config "${cfg}" 2>&1 | tee -a "logs/${run_id}.log"; then
    echo -e "${run_id}\tOK"   >> "${STATUS}"
  else
    echo -e "${run_id}\tFAIL" >> "${STATUS}"
  fi
done

echo "== done =="
echo "status -> ${STATUS}"
