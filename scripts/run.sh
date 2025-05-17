#!/usr/bin/env bash
BASE_CFG=configs/default_config_train.yaml
BASE_DATA=/data/GitHub/first_year_exam/data
REPS=(1 2 3)

#
# 1) Corridas por sub‐carpeta ×3 réplicas
#
for rep in "${REPS[@]}"; do
  for DS in data_uncorrected data_corrected_w_annotation data_corrected_wo_annotation; do
    RUNID="${DS}_r${rep}"
    echo "=== Run: ${RUNID} ==="
    python main.py \
      -c "${BASE_CFG}" \
      --run-id "${RUNID}" \
      --data-dir-override "${BASE_DATA}/${DS}" \
      --train-file-override "train.csv" \
      --dev-file-override   "validation.csv" \
      --test-file-override  "test.csv"
  done
done

#
# 2) Corridas por sufijo dentro de data_uncorrected ×3 réplicas
#
SUFFIXES=(l nl lb lm)
for rep in "${REPS[@]}"; do
  for suf in "${SUFFIXES[@]}"; do
    RUNID="uncorrected_${suf}_r${rep}"
    echo "=== Run: ${RUNID} ==="
    python main.py \
      -c "${BASE_CFG}" \
      --run-id "${RUNID}" \
      --data-dir-override "${BASE_DATA}/data_uncorrected" \
      --train-file-override "train_${suf}.csv" \
      --dev-file-override   "validation_${suf}.csv" \
      --test-file-override  "test_${suf}.csv"
  done
done
