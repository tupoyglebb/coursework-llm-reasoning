#!/usr/bin/env bash
# CUDA: один эксперимент целиком — обучение -> инференс на замороженном тесте -> метрики -> сводка.
# Использование:  bash run_experiment.sh E1     (E0..E3 -> Tier 0, E4..E5 -> Tier 1)
set -euo pipefail

EXP=${1:?"usage: bash run_experiment.sh <E0|E1|E2|E3|E4|E5>"}
case "$EXP" in
  E0|E1|E2|E3) TIER=tier0 ;;
  E4|E5)       TIER=tier1 ;;
  *) echo "Неизвестный эксперимент: $EXP"; exit 1 ;;
esac

CONFIG="./configs/${TIER}_${EXP}.yaml"
ADAPTER="./adapters/${TIER}_${EXP}"
TESTCSV="./data/test_source.csv"
PRED="./predictions/test_pred_${EXP}.csv"
MET="./predictions/metrics_${EXP}.json"

[ -f "$CONFIG" ]  || { echo "Нет конфига: $CONFIG"; exit 1; }
[ -f "$TESTCSV" ] || { echo "Нет теста: $TESTCSV (скопируй data_mlx/test_source.csv в ./data/)"; exit 1; }

echo ">>> [$EXP] ОБУЧЕНИЕ ($CONFIG)"
python scripts/train_qlora_cuda.py --config "$CONFIG"

echo ">>> [$EXP] ИНФЕРЕНС на замороженном тесте ($TESTCSV)"
python scripts/infer_cuda.py --input "$TESTCSV" --output "$PRED" \
  --adapter-path "$ADAPTER" --prompt-style compact --max-tokens 900

echo ">>> [$EXP] МЕТРИКИ"
python scripts/evaluate_predictions.py --predictions "$PRED" --out-json "$MET"

echo ">>> [$EXP] СВОДКА"
python scripts/aggregate_metrics.py --metrics "$MET" --label "$EXP"

echo ">>> [$EXP] ГОТОВО. Адаптер: $ADAPTER ; метрики: $MET"
