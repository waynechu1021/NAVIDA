# sleep 3h

export PYTHONPATH="./:$PYTHONPATH"
export NCCL_P2P_LEVEL=NVL
CUDA_VISIBLE_DEVICES=2,3,4,5 deepspeed --master_port 25420 src/train/train.py \
    --deepspeed scripts/zero2.json \
    --dataset_name data/only_r2r_idm_vln_mix_data.jsonl \
    --model_name_or_path .cache/Qwen2.5-VL-3B-Instruct \
    --num_train_epochs 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --learning_rate 2.0e-5 \
    --logging_steps 5 \
    --eval_strategy no \
    --eval_steps 100 \
    --save_strategy no \
    --save_steps 12000 \
    --output_dir result/navida_qwen2_5_vl_3b_train_on_r2r_mix_data_wo_dagger \
    --report_to tensorboard \