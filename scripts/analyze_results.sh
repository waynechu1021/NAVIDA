#!/bin/bash

SAVE_PATH=eval_log/navida_best_r2r

echo $SAVE_PATH
python src/eval/analyze_results.py \
    --path $SAVE_PATH
