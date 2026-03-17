#!/bin/bash

# sleep 7h

MODEL_PATH=""
export PYTHONPATH=`pwd`:$PYTHONPATH

#R2R
CONFIG_PATH="config/vln_r2r.yaml"
SAVE_PATH="eval_log/navida_r2r"


#RxR
# CONFIG_PATH="config/vln_rxr.yaml"
# SAVE_PATH="eval_log/navida_rxr"

# gpus available
CHUNKS=8
gpus=(2 2 3 3 4 4 5 5)

for IDX in $(seq 0 $((CHUNKS-1))); do
    echo ${gpus[$IDX]}
    CUDA_VISIBLE_DEVICES=${gpus[$IDX]} python src/eval/eval.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --split-id $IDX \
    --forward-distance 25 \
    --turn-angle 15 \
    --resolution-ratio 0.5 \
    --max-action-history 200 \
    --model-path $MODEL_PATH \
    --num-generations 1 \
    --result-path $SAVE_PATH &
    
done

wait

python src/eval/analyze_results.py \
    --path $SAVE_PATH

