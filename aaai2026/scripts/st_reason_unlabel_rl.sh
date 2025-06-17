#!/bin/bash

set -x

temperature=0.9
top_p=1.0
batch_size_per_gpu=2
dataset="unlabel"
exper_name="Reason_unlabel_2020_RL"
python ./_rl.py \
    custom_reward_function.path="./reward_model/unlabel.py" \
    trainer.project_name="Grammar_RL" \
    trainer.experiment_name=${exper_name} \
    data.path="./data/${dataset}" \
    data.train_files="./data/${dataset}/parquet/train.parquet" \
    data.val_files="./data/${dataset}/parquet/valid.parquet" \
    data.max_prompt_length=2350 \
    data.max_response_length=700 \
    data.train_batch_size=8 \
    data.val_batch_size=32 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${batch_size_per_gpu}\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=13300\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${batch_size_per_gpu}\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${batch_size_per_gpu}\
    actor_rollout_ref.rollout.temperature=${temperature}\
    actor_rollout_ref.rollout.top_p=${top_p}\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5\
    actor_rollout_ref.rollout.n=5\
    actor_rollout_ref.rollout.max_num_batched_tokens=4096\
    actor_rollout_ref.rollout.max_num_seqs=1024\
    trainer.save_freq=1\
    trainer.test_freq=10 \
    trainer.total_epochs=5 \
    actor_rollout_ref.model.load_param=True \
    actor_rollout_ref.model.load_param_path="./models/reason_sft.pt" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4\
    trainer.n_gpus_per_node=4 \
    trainer.logger=['console','wandb'] \
    trainer.default_local_dir="./checkpoints/rl/Grammar_Generation/${exper_name}" \