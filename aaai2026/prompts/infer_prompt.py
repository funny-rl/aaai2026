from prompts.base_instruction import get_prompt_func

def infer_dataset(
    model_name: str,
    data_path: str,
)-> None:
    import os 
    from datasets import load_dataset

    jsonl_path = os.path.join(data_path, 'jsonl')
    parquet_path = os.path.join(data_path, 'parquet')

    dataset = load_dataset(
        "json",
        data_files={
            "test": f"{jsonl_path}/test.jsonl",
        }
    )
    
    test_dataset = dataset['test']
    
    def make_map_fn(split:str):
        def process_fn(example:dict[str, str], idx: int):
            chat_prompt = get_prompt_func(model_name, example["description"])
            data = {
                "prompt": chat_prompt,
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    test_dataset.to_parquet(os.path.join(parquet_path, 'test.parquet'))