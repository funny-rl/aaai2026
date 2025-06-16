#!/bin/bash

set -x

python ./_infer.py\
    data.batch_size=32\
    model.load_param=True\
    model.load_param_path="./checkpoints/rl/Grammar_Generation/REASON_Unlabel_RL/global_step_70/model.pt"\
    data.output_path="./model_output/sft_pass@1.jsonl"\
    data.n_samples=1\
    data.path="./data/label/parquet/test.parquet" \
    rollout.temperature=0.0\
    rollout.top_p=1.0 \
    rollout.n=1\
    rollout.prompt_length=2500\
    rollout.response_length=1000\
