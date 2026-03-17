export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python3 src/data/prepare_training_data.py \
    --dataset_name r2r rxr envdrop scalevln \
    --task_type vln idm \
    --output_path data/navida_train_data