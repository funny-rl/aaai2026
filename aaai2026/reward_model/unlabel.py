import ast 
import time 
from reward_model.utils.utils import extract_solution
from reward_model.utils.get_testcases import get_testcases, get_efficiency_score

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
    total_reward = -0.5
    num_testcase = 10 # Number of testcases to generate: num_testcase x2 = total_testcase
    num_solution = 20
    tc_timeout = 10
    start_time = time.time()
    if solution is None:
        print(f"[Format Error] Invalid solution format")
        
    else:
        total_reward = -0.1
        try:
            
            grammar = ast.literal_eval(solution)
            print("[Grammar] ", grammar)
            testcases = get_testcases(
                data=grammar,
                num_testcase=num_testcase,
                timeout=tc_timeout,
            )
            total_reward = 0.0
            try:
                (
                    validity, 
                    effectiveness, 
                    n_correct_solution, 
                    n_incorrect_solution
                ) = get_efficiency_score(
                    name=ground_truth["name"],
                    testcases=testcases,
                    n_testcode=num_solution,
                )
                R = (validity + effectiveness) / 2.0
                total_reward = R
                print(
                    f"{ground_truth['name']} - Validity: {validity} | Effectiveness: {effectiveness} | Correct Solutions: {n_correct_solution} | Incorrect Solutions: {n_incorrect_solution}"
                )

            except Exception as e:
                print(f"[Solution Error] {type(e).__name__}: {e}")
        except Exception as e:
            print(f"[TC Generation Error] Unsatisfy Well-Formedness {type(e).__name__}: {e}")

    end_time = time.time()
    print(f"Total Reward {total_reward} | Time taken: {end_time - start_time:.3f} seconds")
    print("=" * 100)
    return total_reward