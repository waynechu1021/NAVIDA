#!/bin/bash

# sleep 7h

MODEL_PATH="/nvme-ssd1/zwy/navida/result/Qwen2.5-VL_3B_sft_r2r_rxr_scalevln_and_r2r_dagger_double_and_rxr_dagger_main_task_with_idm_and_envdrop_traj_sum/ckpt-80000"
# MODEL_PATH="/nvme-ssd1/zwy/navid_ws/R1-V/src/r1-v/result/Qwen2.5-VL_3B_sft_r2r_rxr_and_dagger_double_with_idm_and_scalevln_and_envdrop_traj_summary"
# MODEL_PATH="/nvme-ssd1/zwy/navid_ws/R1-V/src/r1-v/result/Qwen2.5-VL_3B_best_post_train_on_rxr_dagger_main_task_with_idm"
export PYTHONPATH=`pwd`:$PYTHONPATH

#R2R
CONFIG_PATH="config/vln_r2r.yaml"
SAVE_PATH="eval_log/navida_best_r2r"


#RxR
# CONFIG_PATH="VLN_CE/vlnce_baselines/config/rxr_baselines/navid_rxr_test.yaml"
# SAVE_PATH="tmp/qwen2_vl_2b_on_rxr"

# gpus available
CHUNKS=8
gpus=(2 2 3 3 4 4 5 5)
# CHUNKS=1
# gpus=(2)

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

