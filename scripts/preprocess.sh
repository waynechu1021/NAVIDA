export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python3 src/data/preprocess.py \
    --dataset_name r2r rxr envdrop scalevln \