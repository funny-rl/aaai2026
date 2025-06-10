import os 
import time
import pandas as pd
import json
#from rm1.utils import get_testcases
from rm2.utils import get_testcases



def reward_function(data: dict) -> float:
    """
    calculate the reward score and return the time taken to process the data.

    Args:
        data (dict): grammar ex.{productions: [""], constraints: [""]}
        
    Returns:
        float: the time taken to process the data
    """
    start_time = time.time()
    get_testcases(
        data = data,
        k=5, # 2*k+1 
        timeout = 10,
    )
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    import pandas as pd
    data_file = os.path.join("./data/data.parquet")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} does not exist. Please run the data preparation script first.")
    
    df = pd.read_parquet(data_file)
    
    mean_time = 0.0
    correct_grammar = 0
    updated_data = []

    for index, data in df.iterrows():
        try:
            time1 = reward_function(data)
            mean_time += time1
            correct_grammar += 1
        except Exception as e:
            print(e)
            pass
    print(round(mean_time, 4), f"correct_grammar: {correct_grammar}") # len(df) is the number of rows in the dataframe
    
    
