#!/usr/bin/env bash
# Один эксперимент целиком: обучение (лучший чекпойнт по eval_loss внутри) ->
# генеративная оценка лучшего адаптера на замороженном тесте -> строка в results.csv.
#
# Лучший чекпойнт выбирается ВНУТРИ обучения по eval_loss (load_best_model_at_end).
# Генеративная оценка по полям вынесена в отдельный scripts/evaluate_model.py — его можно
# запускать и независимо (на любом чекпойнте/выборке).
#
# Использование:
#   CUDA_VISIBLE_DEVICES=1 bash run_experiment.sh E0      # E0..E3 -> Tier 0, E4..E5 -> Tier 1
set -euo pipefail
IFS=$'\n\t'

usage() { echo "usage: [CUDA_VISIBLE_DEVICES=N] bash run_experiment.sh <E0|E1|E2|E3|E4|E5>"; }
[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { usage; exit 0; }

EXP=${1:?"$(usage)"}
case "$EXP" in
  E0|E1|E2|E3) TIER=tier0 ;;
  E4|E5)       TIER=tier0 ;;
  *) echo "Неизвестный эксперимент: $EXP"; usage; exit 1 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONFIG="./configs/${TIER}_${EXP}.yaml"
ADAPTER="./adapters/${TIER}_${EXP}"
TESTCSV="./data/test_source.csv"

[ -f "$CONFIG" ]  || { echo "Нет конфига: $CONFIG"; exit 1; }
[ -f "$TESTCSV" ] || { echo "Нет теста: $TESTCSV"; exit 1; }

mkdir -p adapters predictions experiments logs

log() { printf '>>> [%s] %s | %s\n' "$EXP" "$(date '+%H:%M:%S')" "$*"; }
trap 'echo "!!! [$EXP] ОШИБКА на строке $LINENO"; exit 1' ERR

log "конфиг=$CONFIG  адаптер=$ADAPTER  GPU=${CUDA_VISIBLE_DEVICES:-all}"

log "ОБУЧЕНИЕ (лучший чекпойнт по eval_loss внутри)"
python scripts/train_qlora_cuda.py --config "$CONFIG"

log "ГЕНЕРАТИВНАЯ ОЦЕНКА лучшего адаптера на замороженном тесте ($TESTCSV)"
python scripts/evaluate_model.py --config "$CONFIG" --label "$EXP" --input "$TESTCSV"

log "ГОТОВО. Адаптер: $ADAPTER ; результаты: experiments/results.csv ; TensorBoard: tensorboard --logdir $ADAPTER/runs"
