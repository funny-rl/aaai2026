import re
import os
import time
import random 
import tempfile
import itertools
import py_compile
import subprocess
from pathlib import Path
from itertools import chain
import numpy as np
from typing import IO, Optional, TypedDict

from pathos.multiprocessing import ProcessingPool as Pool

# SOLUTIONS_DIR = os.getenv("SOLUTIONS_DIR")
# if SOLUTIONS_DIR is None:
#     raise ValueError("Environment variable SOLUTIONS_DIR is not set")
# CORRECT_SOLUTIONS_DIR = os.getenv("CORRECT_SOLUTIONS_DIR")
# if CORRECT_SOLUTIONS_DIR is None:
#     raise ValueError("Environment variable CORRECT_SOLUTIONS_DIR is not set")
# INCORRECT_SOLUTIONS_DIR = os.getenv("INCORRECT_SOLUTIONS_DIR")
# if INCORRECT_SOLUTIONS_DIR is None:
#     raise ValueError("Environment variable INCORRECT_SOLUTIONS_DIR is not set")

SOLUTIONS_DIR = "reward_model/utils/data/solutions"
CORRECT_SOLUTIONS_DIR = f"{SOLUTIONS_DIR}/solutions"
INCORRECT_SOLUTIONS_DIR = f"{SOLUTIONS_DIR}/incorrect_solutions"

def extract_solution(solution_str: str) -> str | None:
    think_token_occurrences = re.findall(r'</think>', solution_str)
    if len(think_token_occurrences) != 1:
        return None
    match = re.search(r'</think>(.+)', solution_str)
    if match and match.group(1).strip():
        return match.group(1)
    return None

class ExecutionResultPerTestcase(TypedDict):
    """ExecutionResultPerTestcase is a dictionary that contains the results of
    execution for a single testcase.
    """

    correct_outputs: list[Optional[str]]
    incorrect_outputs: list[Optional[str]]
    correct_solutions: list[str]
    incorrect_solutions: list[str]

def get_mode(xs: list[str]) -> tuple[str, int]:
    groupby_iterable = itertools.groupby(sorted(xs))
    groups = [(k, len(list(v))) for k, v in groupby_iterable]
    groups = sorted(groups, key=lambda e: e[1], reverse=True)
    mode, num_of_mode = groups[0]
    return mode, num_of_mode

def summarize_and_score(results: list[ExecutionResultPerTestcase]) -> float:
    if not results:
        return 0.0

    total_incorrect_flags = [True] * len(results[0]["incorrect_outputs"])

    for r in results:
        correct_outputs = [o for o in r["correct_outputs"] if o is not None]
        if not correct_outputs:
            return 0.0
        answer, _ = get_mode(correct_outputs)
        if any(o != answer for o in r["correct_outputs"]):
            return 0.0
        for i, o in enumerate(r["incorrect_outputs"]):
            total_incorrect_flags[i] &= (o == answer)

    return 1.0 - sum(total_incorrect_flags) / len(total_incorrect_flags)

def precompile_solution(solution_path: Path) -> bool:
    try:
        py_compile.compile(str(solution_path), doraise=True)
        return True
    except py_compile.PyCompileError as e:
        #print(f"[CompileError] {solution_path.name}: {e}")
        return False
    

def is_invalid_terminal(s: str) -> bool:
    return bool(
        any(
            re.search(pattern, s)
            for pattern in [
                r"\[[^\]]+\]\+",
                r"\\d\+",
                r"\\w\+",
                r"\\s\+",
                r"\\b",
                r"\\S",
                r"\\W",
            ]
        )
    )

def sample_solutions(
    name: str,
    num_sample: int = 20,
    correctness: str = "correct",
) -> tuple[list[Path], int]:
    """
    return only the Compilable solutions.
    if num_sample is larger than the number of compilable solutions, 
    it will return only compilable solutions.
    """
    if correctness == "correct":
        solutions_dir = Path(CORRECT_SOLUTIONS_DIR) / name
    elif correctness == "incorrect":
        solutions_dir = Path(INCORRECT_SOLUTIONS_DIR) / name
    else:
        raise ValueError("Correctness must be either 'correct' or 'incorrect'")
    solutions = list(solutions_dir.glob("*.py"))
    if len(solutions) < 1:
        raise ValueError(f"No solutions found for {name} in {correctness} directory")
    total_sample = min(num_sample, len(solutions))
    
    tried_solutions = set()
    n_sample:int = total_sample
    complied_solutions = []
    
    sol_set = set(solutions)
    with Pool() as compile_pool:
        while len(complied_solutions) < total_sample:
            
            untried_solutions = list(sol_set - tried_solutions)
            if len(untried_solutions) < 1:
                break
            if len(untried_solutions) < n_sample:
                n_solution = len(untried_solutions)
            else:
                n_solution = n_sample
            sampled_solution = random.sample(untried_solutions, n_solution)
                
            compiled_flags = list(compile_pool.map(precompile_solution, sampled_solution))
            tried_solutions.update(sampled_solution)
            for idx, sol in enumerate(sampled_solution):
                if compiled_flags[idx]:
                    complied_solutions.append(sol)
                    n_sample -= 1
    if len(complied_solutions) < 1:
        raise ValueError(f"No compilable solutions found for {name} in {correctness} directory")
    
    return complied_solutions, len(complied_solutions)

def get_stdout(
    python_file: Path, 
    testcase_str: str, 
    timeout: int
) -> Optional[str]:
    try:
        process = subprocess.run(
            ["python", str(python_file)],
            capture_output=True,
            input=testcase_str,
            timeout=timeout,
            text=True,
            check=True,
        )
        if process.returncode != 0:
            return None
    except Exception as e:
        return None

    return " ".join(process.stdout.split()).lower()

def test_pairs(
    args: tuple[str, Path, int]
):
    testcase, solution, timeout = args 
    return get_stdout(solution, testcase, timeout)

def efficiency_score(
    name: str,
    n_sample: int,
    testcases: list[str],
    timeout: int,
):
    correct_solutions, n_correct_solution = sample_solutions(
        name = name,
        num_sample = n_sample, 
        correctness = "correct"
    )
    incorrect_solutions, n_incorrect_solution = sample_solutions(
        name = name,
        num_sample = n_sample,
        correctness = "incorrect"
    )
    num_testcase = len(testcases)
    all_solutions = list(chain(correct_solutions, incorrect_solutions))
    tc_sol_pairs = list(itertools.product(testcases, all_solutions))
    all_args = [(*pair, timeout) for pair in tc_sol_pairs]
    
    with Pool() as test_tc:
        test_result = test_tc.map(
            test_pairs,
            all_args,
        )
        test_result = list(test_result)
    
    assert len(test_result) == num_testcase * len(all_solutions), \
        f"Expected {num_testcase * len(all_solutions)} results, got {len(test_result)}"
        
    n_solution = n_correct_solution + n_incorrect_solution
    testcase_outputs = [test_result[i:i + n_solution] for i in range(0, len(test_result), n_solution)]
    
    assert len(testcase_outputs) == num_testcase, \
        f"Length of correct outputs and incorrect outputs must match: {len(testcase_outputs)} != {num_testcase}"

    results = []
    for testcase_output in testcase_outputs:
        tc_correct_output = testcase_output[:n_correct_solution]
        tc_incorrect_output = testcase_output[n_correct_solution:]
        
        assert len(tc_correct_output) == n_correct_solution, \
            f"Length of correct outputs must match: {len(tc_correct_output)} != {n_correct_solution}"
        assert len(tc_incorrect_output) == n_incorrect_solution, \
            f"Length of incorrect outputs must match: {len(tc_incorrect_output)} != {n_incorrect_solution}"
        filtered_correct_outputs = [o for o in tc_correct_output if o is not None]
        
        if not filtered_correct_outputs:
            print("[Empty Error] No valid correct outputs")
            return 0, 0, n_correct_solution, n_incorrect_solution
        if len(set(filtered_correct_outputs)) > 1:
            print("[Mismatch Error] Multiple correct outputs found")
            return 0, 0, n_correct_solution, n_incorrect_solution

        result_per_testcase = ExecutionResultPerTestcase(
            correct_outputs=tc_correct_output,
            incorrect_outputs=tc_incorrect_output,
            correct_solutions=[str(e.name) for e in correct_solutions],
            incorrect_solutions=[str(e.name) for e in incorrect_solutions],
        )
        results.append(result_per_testcase)
    
    effectiveness = summarize_and_score(results)
    return 1.0, effectiveness, n_correct_solution, n_incorrect_solution