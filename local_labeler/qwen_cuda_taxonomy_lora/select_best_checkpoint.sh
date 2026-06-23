#!/usr/bin/env bash
# CUDA: выбор лучшего чекпойнта по VALIDATION, затем метрика победителя на TEST.
# На NVIDIA чекпойнты — это папки <adapter>/checkpoint-XXX/ (готовый peft-адаптер), плюс финальный <adapter>.
# Использование:  bash select_best_checkpoint.sh E1 [N_LAST] [VALID_LIMIT]
set -euo pipefail

EXP=${1:?"usage: bash select_best_checkpoint.sh <E0..E5> [N_LAST=3] [VALID_LIMIT=0]"}
N_LAST=${2:-3}
VALID_LIMIT=${3:-0}
case "$EXP" in
  E0|E1|E2|E3) TIER=tier0 ;;
  E4|E5)       TIER=tier1 ;;
  *) echo "Неизвестный эксперимент: $EXP"; exit 1 ;;
esac

ADAPTER_DIR="./adapters/${TIER}_${EXP}"
VALID="./data/valid_source.csv"
TEST="./data/test_source.csv"
[ -d "$ADAPTER_DIR" ] || { echo "Нет адаптеров: $ADAPTER_DIR (сначала: bash run_experiment.sh $EXP)"; exit 1; }
[ -f "$VALID" ] || { echo "Нет valid: $VALID"; exit 1; }

# Кандидаты: последние N checkpoint-* + финальный (корень adapter_dir).
CANDIDATES=()
while IFS= read -r line; do
  [ -n "$line" ] && CANDIDATES+=("$line")
done < <(ls -d "$ADAPTER_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -n "$N_LAST")
CANDIDATES+=("$ADAPTER_DIR")

RESULTS="./predictions/ckpt_select_${EXP}.csv"
rm -f "$RESULTS"
LIMIT_ARG=""
if [ "${VALID_LIMIT:-0}" -gt 0 ] 2>/dev/null; then LIMIT_ARG="--limit $VALID_LIMIT"; fi

echo "Эксперимент $EXP. Кандидаты (оценка на valid, лимит строк = $VALID_LIMIT; 0 = все):"
printf '  %s\n' "${CANDIDATES[@]}"
echo ""

for CKPT in "${CANDIDATES[@]}"; do
  [ -f "$CKPT/adapter_config.json" ] || { echo "пропуск (нет адаптера): $CKPT"; continue; }
  TAG=$(basename "$CKPT"); [ "$TAG" = "$(basename "$ADAPTER_DIR")" ] && TAG="final"
  LABEL="${EXP}@${TAG}"; SAFE="${LABEL//@/_}"
  echo ">>> [$LABEL] инференс на valid"
  python scripts/infer_cuda.py --input "$VALID" --output "./predictions/valid_pred_${SAFE}.csv" \
    --adapter-path "$CKPT" --prompt-style compact --max-tokens 900 $LIMIT_ARG
  python scripts/evaluate_predictions.py --predictions "./predictions/valid_pred_${SAFE}.csv" \
    --out-json "./predictions/valid_metrics_${SAFE}.json"
  python scripts/aggregate_metrics.py --metrics "./predictions/valid_metrics_${SAFE}.json" \
    --label "$LABEL" --results-csv "$RESULTS"
done

# Победитель = max mean_f1_all на valid.
python3 - "$RESULTS" "$ADAPTER_DIR" <<'PY'
import csv, os, sys
results, adapter_dir = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open(results)) if r.get("mean_f1_all") not in (None, "")]
best = max(rows, key=lambda r: float(r["mean_f1_all"]))
tag = best["label"].split("@", 1)[1] if "@" in best["label"] else "final"
path = adapter_dir if tag == "final" else os.path.join(adapter_dir, tag)
print("\n=== ЛУЧШИЙ ЧЕКПОЙНТ ПО VALID:", best["label"], "===")
print(f"  mean_f1_all={best['mean_f1_all']}  collapsed={best['mean_f1_collapsed']}  good={best['mean_f1_good']}")
print(f"  путь: {path}")
open(os.path.join(adapter_dir, "BEST_CHECKPOINT.txt"), "w").write(path + "\n")
PY

WINNER=$(cat "$ADAPTER_DIR/BEST_CHECKPOINT.txt")
echo ""
echo ">>> [$EXP-best] финальная оценка победителя на TEST ($TEST)"
python scripts/infer_cuda.py --input "$TEST" --output "./predictions/test_pred_${EXP}_best.csv" \
  --adapter-path "$WINNER" --prompt-style compact --max-tokens 900
python scripts/evaluate_predictions.py --predictions "./predictions/test_pred_${EXP}_best.csv" \
  --out-json "./predictions/metrics_${EXP}_best.json"
python scripts/aggregate_metrics.py --metrics "./predictions/metrics_${EXP}_best.json" --label "${EXP}-best"

echo ""
echo "Готово. Победитель: $WINNER"
echo "  - сводка кандидатов (valid): $RESULTS"
echo "  - финальное число (test): метка '${EXP}-best' в experiments/results.csv"
