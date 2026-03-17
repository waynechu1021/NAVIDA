#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH

#R2R
CONFIG_PATH="config/vln_r2r.yaml"
SAVE_PATH="eval_log/navida_r2r"

#RxR
# CONFIG_PATH="config/vln_rxr.yaml"
# SAVE_PATH="eval_log/navida_rxr" 

CHUNKS=16 # which is also the number of simulators to launch during evaluation

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://127.0.0.1:8201/v1"

CUDA_VISIBLE_DEVICES=4 python src/eval/eval_vllm.py \
    --exp-config $CONFIG_PATH \
    --split-num $CHUNKS \
    --forward-distance 25 \
    --turn-angle 15 \
    --max-action-history 200 \
    --num-generations 1 \
    --result-path $SAVE_PATH \
    
python src/eval/analyze_results.py \
    --path $SAVE_PATH

