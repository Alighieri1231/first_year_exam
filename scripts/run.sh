#!/usr/bin/env bash
BASE_CFG=configs/default_config_train.yaml
BASE_DATA=/data/GitHub/first_year_exam/data
REPS=(1 2 3)

CMD="python -m src.train_lightning_seg"

#
# 1) Corridas por sub‐carpeta ×3 réplicas
#
for rep in "${REPS[@]}"; do
  for DS in data_uncorrected data_corrected_w_annotation data_corrected_wo_annotation; do
    TRAIN_CSV="${BASE_DATA}/${DS}/train.csv"
    DEV_CSV ="${BASE_DATA}/${DS}/validation.csv"
    TEST_CSV ="${BASE_DATA}/${DS}/test.csv"
    RUNID="${DS}_r${rep}"
    echo "=== Run: ${RUNID} ==="
    $CMD \
      -c "${BASE_CFG}" \
      --run-id "${RUNID}" \
      --train-file-override "${TRAIN_CSV}" \
      --dev-file-override   "${DEV_CSV}" \
      --test-file-override  "${TEST_CSV}"
  done
done

#
# 2) Corridas por sufijo dentro de data_uncorrected ×3 réplicas
#
SUFFIXES=(l nl lb lm)
for rep in "${REPS[@]}"; do
  for suf in "${SUFFIXES[@]}"; do
    TRAIN_CSV="${BASE_DATA}/data_uncorrected/train_${suf}.csv"
    DEV_CSV  ="${BASE_DATA}/data_uncorrected/validation_${suf}.csv"
    TEST_CSV ="${BASE_DATA}/data_uncorrected/test_${suf}.csv"
    RUNID="uncorrected_${suf}_r${rep}"
    echo "=== Run: ${RUNID} ==="
    $CMD \
      -c "${BASE_CFG}" \
      --run-id "${RUNID}" \
      --train-file-override "${TRAIN_CSV}" \
      --dev-file-override   "${DEV_CSV}" \
      --test-file-override  "${TEST_CSV}"
  done
done
