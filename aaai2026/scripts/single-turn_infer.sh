#!/bin/bash

set -x

python ./_infer.py\
    model.load_param=True\
    model.load_param_path="./models/sft.pt"\
    data.output_path="./model_output/sft_pass@1.jsonl"\
    data.n_samples=1\
    data.path="./data/label/parquet/test.parquet" \
    rollout.temperature=0.0\
    rollout.top_p=1.0 \
    rollout.n=1\
