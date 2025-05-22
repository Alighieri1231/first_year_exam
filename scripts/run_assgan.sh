#!/usr/bin/env bash
BASE_CFG=configs/assgan_config_train.yaml
BASE_DATA=/scratch/bcastane_lab/eochoaal/first_year_exam/data
REPS=(1)
SUFFIXES=(lm lb)
GAMMAS=(0.2 0.5 0.8)
CMD="python -m src.train_lightning_assgan"


# 2) Sufijos sobre data_uncorrected × replicas × gamma_thresh
for rep in "${REPS[@]}"; do
  for suf in "${SUFFIXES[@]}"; do
    for gt in "${GAMMAS[@]}"; do
      RUNID="uncorrected_${suf}_r${rep}_gt${gt}"
      echo "=== Run: ${RUNID} ==="
      ${CMD} -c "${BASE_CFG}" \
        --run-id "${RUNID}" \
        --train-file-override "${BASE_DATA}/train_${suf}.csv" \
        --dev-file-override   "${BASE_DATA}/validation_${suf}.csv" \
        --test-file-override  "${BASE_DATA}/test_${suf}.csv" \
        --gamma-thresh-override "${gt}" \
        ${extra_args}
    done
  done
done
