#!/usr/bin/env bash
BASE_CFG=configs/assgan_config_train.yaml
BASE_DATA=/scratch/bcastane_lab/eochoaal/first_year_exam/data
REPS=(1 2)
DS_TYPES=(data_uncorrected data_corrected_w_annotation data_corrected_wo_annotation)
SUFFIXES=(l nl lb lm)
GAMMAS=(0.2 0.5)
CMD="python -m src.train_lightning_assgan"

# 1) Data variants × replicas × gamma_thresh
for rep in "${REPS[@]}"; do
  for DS in "${DS_TYPES[@]}"; do
    for gt in "${GAMMAS[@]}"; do
      RUNID="${DS}_r${rep}_gt${gt}"
      echo "=== Run: ${RUNID} ==="
      ${CMD} -c "${BASE_CFG}" \
        --run-id "${RUNID}" \
        --data-dir-override "${BASE_DATA}/${DS}" \
        --train-file-override "${BASE_DATA}/${DS}/train.csv" \
        --dev-file-override   "${BASE_DATA}/${DS}/validation.csv" \
        --test-file-override  "${BASE_DATA}/${DS}/test.csv" \
        --gamma-thresh-override "${gt}"
    done
  done
done

# 2) Sufijos sobre data_uncorrected × replicas × gamma_thresh
for rep in "${REPS[@]}"; do
  for suf in "${SUFFIXES[@]}"; do
    for gt in "${GAMMAS[@]}"; do
      RUNID="uncorrected_${suf}_r${rep}_gt${gt}"
      echo "=== Run: ${RUNID} ==="
      ${CMD} -c "${BASE_CFG}" \
        --run-id "${RUNID}" \
        --data-dir-override "${BASE_DATA}" \
        --train-file-override "${BASE_DATA}/train_${suf}.csv" \
        --dev-file-override   "${BASE_DATA}/validation_${suf}.csv" \
        --test-file-override  "${BASE_DATA}/test_${suf}.csv" \
        --gamma-thresh-override "${gt}"
    done
  done
done
