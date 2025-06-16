#!/bin/bash

set -x

nproc_per_node=4
project_name="Grammar_SFT"
experiment_name="SFT"
save_path="./checkpoints/sft/$project_name/$experiment_name"
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node ./_sft.py\
    data.train_files="./data/label/parquet/train.parquet" \
    data.val_files="./data/label/parquet/valid.parquet" \
    trainer.default_local_dir=$save_path\
    trainer.project_name=$project_name\
    trainer.experiment_name=$experiment_name $@