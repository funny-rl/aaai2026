import random 
import pandas as pd

def jsonl_to_parquet(jsonl_file):
    df = pd.read_json(jsonl_file, lines=True)
    df = df[:10]
    #df["grammar"] = df["grammar"].apply(lambda x: x.replace("productions","production") if random.random() < 0.1 else x)
    df.to_parquet("data.parquet", index=False)
    df.to_json("data.json", orient="records", lines=True)
if __name__ == "__main__":
    input_file = "test.jsonl"
    jsonl_to_parquet(input_file)
        
    
    
    
    