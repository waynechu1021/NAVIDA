export NCCL_P2P_LEVEL=NVL

MODEL_PATH=""

CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL_PATH --task generate \
    --trust-remote-code  --limit-mm-per-prompt image=99999 \
    --mm_processor_kwargs '{"max_pixels": 501760}' \
    --max-model-len 32768 --max-num-batched-tokens 65536 \
    --port 8201 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \