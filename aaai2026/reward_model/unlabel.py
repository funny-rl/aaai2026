import ast 
import time 
from reward_model.utils.utils import extract_solution
from reward_model.utils.get_testcases import get_testcases, get_efficiency_score, calculate_validity

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
    if extra_info["split"] == "train":
        total_reward = -0.5
        num_testcase = 10 # Number of testcases to generate: num_testcase x2 = total_testcase
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
        print(f"Name {ground_truth['name']} | Total Reward {total_reward} | Time taken: {end_time - start_time:.3f} seconds")
        print("=" * 100)
    elif extra_info["split"] == "valid":
        try:
            ground_truth = ground_truth["grammar"]
            total_reward = 0.0
            grammar = ast.literal_eval(solution)
            if set(grammar.keys()) != {"productions", "constraints"}:
                raise ValueError("Invalid grammar format")
            if grammar == ground_truth:
                total_reward += 1.0
                raise StopIteration
            testcases = get_testcases(
                data = grammar,
                num_testcase=10,
                timeout = 10,
            )
            validity_score = calculate_validity(
                testcases = testcases, 
                gt_grammar = ground_truth,
            )
            
            gt_testcases = get_testcases(
                data = ground_truth,
                num_testcase=10,
                timeout = 10,
            )
            generality_score = calculate_validity(
                testcases = gt_testcases,
                gt_grammar = grammar,
            )
            total_reward += (validity_score * generality_score)
        except Exception as e:
            pass
    else:
        raise ValueError(f"Invalid split: {extra_info['split']}. Expected 'train' or 'valid'.")
    return total_reward