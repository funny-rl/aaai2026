from prompts.base_instruction import get_prompt_func

def sft_dataset(
        model_name: str,
        data_path: str,
        use_CoT: bool,
) -> None:
    """Load and process the SFT dataset for training."""
    import os
    import json
    from datasets import load_dataset, DatasetDict 
    
    jsonl_path = os.path.join(data_path, 'jsonl')
    parquet_path = os.path.join(data_path, 'parquet')

    dataset: DatasetDict = load_dataset(
        "json",
        data_files={
            "train": f"{jsonl_path}/train.jsonl",
            "valid": f"{jsonl_path}/valid.jsonl",
        }
    )

    train_dataset = dataset['train']
    valid_dataset = dataset["valid"]

    def make_map_fn(split):
        def process_fn(example, idx):
            chat_prompt = get_prompt_func(model_name, example["description"])
            
            if use_CoT:
                answer = f"{example['reason']}</think>{json.dumps(example['grammar'], ensure_ascii=False)}"
            else:
                answer = f"Alright, based on the explanation above, the appropriate grammar for the given last <Specification> is as follows. </think> {json.dumps(example['grammar'], ensure_ascii=False)}"

            data = {
                "prompt": chat_prompt,
                "answer": answer,
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    valid_dataset = valid_dataset.map(function=make_map_fn('valid'), with_indices=True)
    train_dataset.to_parquet(os.path.join(parquet_path, 'train.parquet'))
    valid_dataset.to_parquet(os.path.join(parquet_path, 'valid.parquet'))