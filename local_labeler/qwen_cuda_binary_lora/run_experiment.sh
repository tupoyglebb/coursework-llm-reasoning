#!/usr/bin/env bash
# Один эксперимент целиком: обучение (лучший чекпойнт по eval_loss) →
# генеративная оценка на каноническом тесте data_binary/test_source.csv → строка в results.csv.
#
# Использование:
#   CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh E0
#   CUDA_VISIBLE_DEVICES=0 bash run_experiment.sh no_data__E3
#
# Поддерживаемые имена: E0 E1 E2 E3 E4 E5 E6 E7 E8 no_data__E3
set -euo pipefail
IFS=$'\n\t'

usage() {
    echo "usage: [CUDA_VISIBLE_DEVICES=N] bash run_experiment.sh <EXP>"
    echo "       EXP: E0 E1 E2 E3 E4 E5 E6 E7 E8 no_data__E3"
}
[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { usage; exit 0; }

EXP=${1:?"$(usage)"}

# Определяем имя конфига и метку для results.csv
case "$EXP" in
  E0|E1|E2|E3|E4|E5|E6|E7|E8)
    CONFIG="./configs/tier0_${EXP}.yaml"
    LABEL="$EXP"
    ;;
  no_data__E3)
    CONFIG="./configs/no_data__tier0_E3.yaml"
    LABEL="no_data__E3"
    ;;
  *)
    echo "Неизвестный эксперимент: $EXP"; usage; exit 1 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Canonical test set — всегда неизменённый тест из data_binary
TESTCSV="./data_binary/test_source.csv"

[ -f "$CONFIG"  ] || { echo "Нет конфига: $CONFIG";  exit 1; }
[ -f "$TESTCSV" ] || { echo "Нет теста: $TESTCSV (запустите prepare_data_cuda.py)"; exit 1; }

mkdir -p adapters predictions experiments logs

log() { printf '>>> [%s] %s | %s\n' "$LABEL" "$(date '+%H:%M:%S')" "$*"; }
trap 'echo "!!! [$LABEL] ОШИБКА на строке $LINENO"; exit 1' ERR

log "конфиг=$CONFIG  GPU=${CUDA_VISIBLE_DEVICES:-all}"

log "ОБУЧЕНИЕ (лучший чекпойнт по eval_loss)"
python scripts/train_qlora_cuda.py --config "$CONFIG"

log "ГЕНЕРАТИВНАЯ ОЦЕНКА на $TESTCSV"
python scripts/evaluate_model.py --config "$CONFIG" --label "$LABEL" --input "$TESTCSV"

log "ГОТОВО. Результаты: experiments/results.csv"
