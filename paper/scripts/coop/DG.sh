#!/bin/bash

# custom config
DATA=data/datasets
TRAINER=DPR

# 按顺序依次放到后面去了
# bash xx.sh $1 $2 $3 $4 $5 $6
DATASET=$1
CFG=vit_b16  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

for SEED in 1 2 3
do
    DIR=output_sdg/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=1 python train.py \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --seed ${SEED} \
        --source-domains cartoon \
        --target-domains art_painting sketch photo \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done

