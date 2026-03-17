export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python3 src/data/extract_frame.py \
    --dataset_name r2r rxr envdrop scalevln \
    --num_thread 16 \