#!/usr/bin/env bash
BASE_CFG=configs/assgan_config_train.yaml
BASE_DATA=/data/GitHub/first_year_exam/data
REPS=(1 2 3)
DS_TYPES=(data_uncorrected data_corrected_w_annotation data_corrected_wo_annotation)
SUFFIXES=(l nl lb lm)
CMD="python -m src.train_lightning_assgan"

# 1) Data variants × replicas
for rep in "${REPS[@]}"; do
  for DS in "${DS_TYPES[@]}"; do
    RUNID="${DS}_r${rep}"
    echo "=== Run: ${RUNID} ==="
    ${CMD} -c "${BASE_CFG}" \
      --run-id "${RUNID}" \
      --data-dir-override "${BASE_DATA}/${DS}" \
      --train-file-override "${BASE_DATA}/${DS}/train.csv" \
      --dev-file-override   "${BASE_DATA}/${DS}/validation.csv" \
      --test-file-override  "${BASE_DATA}/${DS}/test.csv"
  done
done

# 2) Sufijos sobre data_uncorrected × replicas
for rep in "${REPS[@]}"; do
  for suf in "${SUFFIXES[@]}"; do
    RUNID="uncorrected_${suf}_r${rep}"
    echo "=== Run: ${RUNID} ==="
    ${CMD} -c "${BASE_CFG}" \
      --run-id "${RUNID}" \
      --data-dir-override "${BASE_DATA}" \
      --train-file-override "${BASE_DATA}/train_${suf}.csv" \
      --dev-file-override   "${BASE_DATA}/validation_${suf}.csv" \
      --test-file-override  "${BASE_DATA}/test_${suf}.csv"
  done
done
