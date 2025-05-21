#!/usr/bin/env bash
BASE_CFG=configs/default_config_train.yaml
BASE_DATA=/data/GitHub/first_year_exam/data
REPS=(1 2 3)
ARCHS=(pspnet manet unetplusplus unet deeplabv3plus)
#ARCHS=(unetplusplus)
CMD="python -m src.train_lightning_seg"


#1) Carpetas × réplicas × archs

# for rep in "${REPS[@]}"; do
#   for DS in data_uncorrected data_corrected_w_annotation data_corrected_wo_annotation; do
#     for arch in "${ARCHS[@]}"; do
#       TRAIN_CSV="${BASE_DATA}/${DS}/train.csv"
#       DEV_CSV="${BASE_DATA}/${DS}/validation.csv"
#       TEST_CSV="${BASE_DATA}/${DS}/test.csv"
#       RUNID="${DS}_r${rep}_${arch}"
#       echo "=== Run: ${RUNID} ==="
#       $CMD \
#         -c "${BASE_CFG}" \
#         --run-id "${RUNID}" \
#         --arch-override "${arch}" \
#         --train-file-override "${TRAIN_CSV}" \
#         --dev-file-override   "${DEV_CSV}" \
#         --test-file-override  "${TEST_CSV}"
#     done
#   done
# done

#
# 2) Sufijos dentro de data_uncorrected × réplicas × archs
##
SUFFIXES=(l nl lb lm)
#SUFFIXES=(l nl)
SUFFIXES=(l lm)

for rep in "${REPS[@]}"; do
  for suf in "${SUFFIXES[@]}"; do
   # Determinar si hay que añadir --binary-split False
    extra_args=""
    if [[ "$suf" == "lm" || "$suf" == "lb" ]]; then
      extra_args="--binary-split False"
    fi

    for arch in "${ARCHS[@]}"; do
      TRAIN_CSV="${BASE_DATA}/train_${suf}.csv"
      DEV_CSV="${BASE_DATA}/validation_${suf}.csv"
      TEST_CSV="${BASE_DATA}/test_${suf}.csv"
      RUNID="uncorrected_${suf}_r${rep}_${arch}"
      echo "=== Run: ${RUNID} ==="
      $CMD \
        -c "${BASE_CFG}" \
        --run-id "${RUNID}" \
        --arch-override "${arch}" \
        --train-file-override "${TRAIN_CSV}" \
        --dev-file-override   "${DEV_CSV}" \
        --test-file-override  "${TEST_CSV}" \
         ${extra_args}
    done
  done
done



