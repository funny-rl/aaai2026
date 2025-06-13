import ast 
from reward_model.utils.utils import extract_solution
from reward_model.utils.get_testcases import get_testcases



def compute_score(
  data_source,
  solution_str,
  ground_truth,
  extra_info=None,
):
    """
    Reward function doesn't utilize the ground truth grammar.
    """
    solution = extract_solution(solution_str=solution_str)
    total_reward = 0.0
    try:
        grammar = ast.literal_eval(solution)
        validity, effectiveness = get_testcases(
            data=grammar,
            name=ground_truth["name"],
            num_testcase=5,
            n_testcode=10,
            timeout=10,
        )
        R = (validity + effectiveness) / 2.0
        total_reward += R
        print(f"{ground_truth['name']} - Validity: {validity}, Effectiveness: {effectiveness}, Reward: {total_reward}")
        
    except Exception as e:
        pass
    
    return total_reward