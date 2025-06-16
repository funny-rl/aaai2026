# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import os
import ast
import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


@hydra.main(config_path="config", config_name="infer", version_base=None)
def main(config):
    run_generation(config)

def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )
    ray.get(main_task.remote(config))

@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    
    from prompts.infer_prompt import infer_dataset
    
    infer_dataset(
        model_name = config.model.path,
        data_path = os.path.dirname(os.path.dirname(config.data.path)),
    )
    

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"
    
    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    reasoning_list = [[] for _ in range(config.data.n_samples)]
    grammar_list = [[] for _ in range(config.data.n_samples)]

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

        # START TO GENERATE FOR n_samples TIMES
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        for n_sample in range(config.data.n_samples):
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)
            grammar_texts = []
            reasoning_texts = []
            for i in range(len(output)): # len(output) -> batch_size x n
                data_item = output[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                grammar_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                
                think_end_index = grammar_str.find("</think>")
                if think_end_index != -1:
                    reasoning = grammar_str[:think_end_index + len("</think>")]
                    grammar_str = grammar_str[think_end_index + len("</think>"):].lstrip() 
                    
                try:
                    grammar_texts.append(ast.literal_eval(grammar_str))
                    reasoning_texts.append(reasoning)
                    print(f"Response {i}: {reasoning}")
                    print(f"Response {i}: {grammar_str}")
                except:
                    grammar_texts.append(grammar_str)
                    reasoning_texts.append("Error")
                    print(f"[Error] Unable to parse grammar: {grammar_str}")
                
                    
            grammar_list[n_sample].extend(grammar_texts)
            reasoning_list[n_sample].extend(reasoning_texts)

    # convert grammar_list from (n_samples, n_data) to (n_data, n_sampels)
    grammar_list = np.array(grammar_list, dtype=object)
    grammar_list = np.transpose(grammar_list, axes=(1, 0)).tolist()
    
    reasoning_list = np.array(reasoning_list, dtype=object)
    reasoning_list = np.transpose(reasoning_list, axes=(1, 0)).tolist()

    # add to the data frame
    dataset["grammar"] = grammar_list
    dataset["reasoning"] = reasoning_list
    
    columns_to_keep = ['name', 'description', 'grammar', 'reasoning']
    existing_columns_to_keep = [col for col in columns_to_keep if col in dataset.columns]
    dataset = dataset[existing_columns_to_keep]

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_json(config.data.output_path, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()