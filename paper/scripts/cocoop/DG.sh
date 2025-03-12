#!/bin/bash

# custom config
DATA=data/datasets
TRAINER=CoCoOp
# TRAINER=CoOp

DATASET=$1
SHOTS=16

CFG=vit_b16_c16_ep10_batch1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet


for SEED in 1 2 3
do
    DIR=output/${DATASET}/${TRAINER}/shotsr_${SHOTS}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains photo cartoon art_painting \
        --target-domains sketch \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done