from prompts.base_instruction import get_prompt_func

def rl_dataset(
        model_name: str,
        data_path: str,
    ):
    import os
    import json
    from datasets import load_dataset
    
    jsonl_path = os.path.join(data_path, 'jsonl')
    parquet_path = os.path.join(data_path, 'parquet')
    
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{jsonl_path}/train.jsonl",
            "valid": f"{jsonl_path}/valid.jsonl"
        }
    )
    
    train_dataset = dataset['train'].shuffle(seed=42)
    valid_dataset = dataset["valid"].shuffle(seed=42)
    
    def make_map_fn(split):
        def process_fn(example, idx):
            chat_prompt = get_prompt_func(model_name, example["description"])
            del example["description"]
            data = {
                "data_source": jsonl_path,
                "prompt": chat_prompt,
                "reward_model" : {
                    "style": "rule",
                    "ground_truth": example
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    valid_dataset = valid_dataset.map(function=make_map_fn('valid'), with_indices=True)
    
    train_parquet_file = os.path.join(parquet_path, 'train.parquet')
    valid_parquet_file = os.path.join(parquet_path, 'valid.parquet')
    if os.path.exists(train_parquet_file):
        os.remove(train_parquet_file)
        train_dataset.to_parquet(train_parquet_file)
    if os.path.exists(valid_parquet_file):
        os.remove(valid_parquet_file)
        valid_dataset.to_parquet(valid_parquet_file)