#!/usr/bin/env bash
# Выбор лучшего чекпойнта по VALIDATION (без прогона всех 12), затем метрика победителя на TEST.
# Отбор по valid -> финальное число на test (без утечки теста в выбор модели).
#
# Использование:
#   bash select_best_checkpoint.sh E1 [N_LAST] [VALID_LIMIT]
#     E1          — эксперимент (E0..E5)
#     N_LAST      — сколько последних нумерованных чекпойнтов проверить (по умолчанию 3)
#     VALID_LIMIT — оценивать на первых K строках valid для скорости (0 = все 376; по умолчанию 0)
#
# Зависит от: scripts/infer_mlx.py, scripts/evaluate_predictions.py, scripts/aggregate_metrics.py
set -euo pipefail
source .venv/bin/activate

EXP=${1:?"usage: bash select_best_checkpoint.sh <E0..E5> [N_LAST=3] [VALID_LIMIT=0]"}
N_LAST=${2:-3}
VALID_LIMIT=${3:-0}
case "$EXP" in
  E0|E1|E2|E3) TIER=tier0 ;;
  E4|E5)       TIER=tier1 ;;
  *) echo "Неизвестный эксперимент: $EXP"; exit 1 ;;
esac

ADAPTER_DIR="./adapters/${TIER}_${EXP}"
VALID="./data_mlx/valid_source.csv"
TEST="./data_mlx/test_source.csv"
[ -d "$ADAPTER_DIR" ] || { echo "Нет адаптеров: $ADAPTER_DIR (сначала обучи: bash run_experiment.sh $EXP)"; exit 1; }
[ -f "$VALID" ] || { echo "Нет valid: $VALID"; exit 1; }
[ -f "$ADAPTER_DIR/adapter_config.json" ] || { echo "Нет adapter_config.json в $ADAPTER_DIR"; exit 1; }

# Кандидаты: последние N нумерованных чекпойнтов + финальный adapters.safetensors.
CANDIDATES=()
while IFS= read -r line; do
  [ -n "$line" ] && CANDIDATES+=("$line")
done < <(ls "$ADAPTER_DIR"/[0-9]*_adapters.safetensors 2>/dev/null | sort | tail -n "$N_LAST")
CANDIDATES+=("$ADAPTER_DIR/adapters.safetensors")

TMP="$ADAPTER_DIR/_ckpt_eval"
mkdir -p "$TMP"
cp "$ADAPTER_DIR/adapter_config.json" "$TMP/adapter_config.json"
RESULTS="./predictions/ckpt_select_${EXP}.csv"
rm -f "$RESULTS"
LIMIT_ARG=""
if [ "${VALID_LIMIT:-0}" -gt 0 ] 2>/dev/null; then LIMIT_ARG="--limit $VALID_LIMIT"; fi

echo "Эксперимент $EXP. Кандидаты (оценка на valid, лимит строк = $VALID_LIMIT; 0 = все):"
printf '  %s\n' "${CANDIDATES[@]}"
echo ""

for CKPT in "${CANDIDATES[@]}"; do
  [ -f "$CKPT" ] || continue
  base=$(basename "$CKPT")
  if [ "$base" = "adapters.safetensors" ]; then TAG="final"; else TAG="${base%_adapters.safetensors}"; fi
  LABEL="${EXP}@${TAG}"
  SAFE="${LABEL//@/_}"

  rm -f "$TMP/adapters.safetensors"
  cp "$CKPT" "$TMP/adapters.safetensors"

  echo ">>> [$LABEL] инференс на valid"
  python scripts/infer_mlx.py --input "$VALID" --output "./predictions/valid_pred_${SAFE}.csv" \
    --model ./models/qwen3-8b-4bit --adapter-path "$TMP" \
    --prompt-style compact --max-tokens 900 $LIMIT_ARG
  python scripts/evaluate_predictions.py --predictions "./predictions/valid_pred_${SAFE}.csv" \
    --out-json "./predictions/valid_metrics_${SAFE}.json"
  python scripts/aggregate_metrics.py --metrics "./predictions/valid_metrics_${SAFE}.json" \
    --label "$LABEL" --results-csv "$RESULTS"
done

# Победитель = max mean_f1_all по valid. Пишем путь чекпойнта в файл, печатаем сводку.
python3 - "$RESULTS" "$ADAPTER_DIR" <<'PY'
import csv, os, sys
results, adapter_dir = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open(results)) if r.get("mean_f1_all") not in (None, "")]
best = max(rows, key=lambda r: float(r["mean_f1_all"]))
tag = best["label"].split("@", 1)[1] if "@" in best["label"] else "final"
ckpt = os.path.join(adapter_dir, "adapters.safetensors" if tag == "final" else f"{tag}_adapters.safetensors")
print("\n=== ЛУЧШИЙ ЧЕКПОЙНТ ПО VALID:", best["label"], "===")
print(f"  mean_f1_all={best['mean_f1_all']}  collapsed={best['mean_f1_collapsed']}  good={best['mean_f1_good']}")
print(f"  файл: {ckpt}")
open(os.path.join(adapter_dir, "BEST_CHECKPOINT.txt"), "w").write(ckpt + "\n")
PY

WINNER=$(cat "$ADAPTER_DIR/BEST_CHECKPOINT.txt")

# Финальная метрика победителя на TEST (пишется в основной experiments/results.csv).
echo ""
echo ">>> [$EXP-best] финальная оценка победителя на TEST ($TEST)"
rm -f "$TMP/adapters.safetensors"
cp "$WINNER" "$TMP/adapters.safetensors"
python scripts/infer_mlx.py --input "$TEST" --output "./predictions/test_pred_${EXP}_best.csv" \
  --model ./models/qwen3-8b-4bit --adapter-path "$TMP" \
  --prompt-style compact --max-tokens 900
python scripts/evaluate_predictions.py --predictions "./predictions/test_pred_${EXP}_best.csv" \
  --out-json "./predictions/metrics_${EXP}_best.json"
python scripts/aggregate_metrics.py --metrics "./predictions/metrics_${EXP}_best.json" --label "${EXP}-best"

rm -rf "$TMP"
echo ""
echo "Готово. Победитель: $WINNER"
echo "  - сводка кандидатов (valid): $RESULTS"
echo "  - финальное число (test) для строки таблицы: метка '${EXP}-best' в experiments/results.csv"
echo "Чтобы зафиксировать победителя как основной адаптер:"
echo "  cp '$WINNER' '$ADAPTER_DIR/adapters.safetensors'"
