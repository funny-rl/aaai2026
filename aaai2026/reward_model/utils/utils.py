import re
import os
import random 
import tempfile
import itertools
import py_compile
import subprocess
from pathlib import Path
from itertools import chain
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
    match = re.search(r'</think>(.*)', solution_str)
    return match.group(1)

class ExecutionResultPerTestcase(TypedDict):
    """ExecutionResultPerTestcase is a dictionary that contains the results of
    execution for a single testcase.
    """

    correct_outputs: list[Optional[str]]
    incorrect_outputs: list[Optional[str]]
    correct_solutions: list[str]
    incorrect_solutions: list[str]

def run_testcase(
    temp_file: IO[bytes],
    solutions: list[Path],
    timeout: int,
) -> list[Optional[str]]:
    position = temp_file.tell()
    temp_file.seek(0)
    outputs = [
        get_stdout(solution, temp_file, timeout) for solution in solutions
    ]
    temp_file.seek(position)
    return outputs

def get_stdout(
    python_file: Path, stdin: IO[bytes], timeout: int
) -> Optional[str]:
    stream_position = stdin.tell()
    try:
        process = subprocess.run(
            ["python", str(python_file)],
            capture_output=True,
            stdin=stdin,
            timeout=timeout,
            text=True,
            check=True,
        )
        stdin.seek(stream_position)
        if process.returncode != 0:
            return None
    except Exception as e:
        return None

    return " ".join(process.stdout.split()).lower()

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
) -> list[Path]:
    
    if correctness == "correct":
        solutions_dir = Path(CORRECT_SOLUTIONS_DIR) / name
    elif correctness == "incorrect":
        solutions_dir = Path(INCORRECT_SOLUTIONS_DIR) / name
    else:
        raise ValueError("Correctness must be either 'correct' or 'incorrect'")
    solutions = list(solutions_dir.glob("*.py"))
    if len(solutions) < 1:
        raise ValueError(f"No solutions found for {name} in {correctness} directory")
    num_sample = min(num_sample, len(solutions))
    return random.sample(solutions, num_sample)

def test_testcase(args: tuple[str, list[Path], list[Path], int]) -> dict[str, list[str]]:
    
    testcase, correct_solutions, incorrect_solutions, timeout = args 
    
    temp_file = tempfile.TemporaryFile("w+b")
    temp_file.write(testcase.encode("utf-8"))
    temp_file.flush()

    correct_outputs = run_testcase(temp_file, correct_solutions, timeout)
    incorrect_outputs = run_testcase(temp_file, incorrect_solutions, timeout)
    temp_file.close()

    filtered_correct_outputs = [o for o in correct_outputs if o is not None]
    if not filtered_correct_outputs:
        return f"[Empty Error] No valid correct outputs"
    if len(set(filtered_correct_outputs)) > 1:
        return f"[Mismatch Error] Correct outputs"

    result_per_testcase = ExecutionResultPerTestcase(
        correct_outputs=correct_outputs,
        incorrect_outputs=incorrect_outputs,
        correct_solutions=[str(e.name) for e in correct_solutions],
        incorrect_solutions=[str(e.name) for e in incorrect_solutions],
    )
    return result_per_testcase

def efficiency_score(
    name: str,
    n_sample: int,
    testcases: list[str],
    timeout: int,
):
    correct_solutions = sample_solutions(
        name = name,
        num_sample = n_sample,
        correctness = "correct"
    )
    incorrect_solutions = sample_solutions(
        name = name,
        num_sample = n_sample,
        correctness = "incorrect"
    )
    all_solutions = list(chain(correct_solutions, incorrect_solutions))
    
    with Pool() as compile_pool:
        compiled_flags = compile_pool.uimap(precompile_solution, all_solutions)
    
    if not all([f for f in compiled_flags]):
        print(f"[Abort] Compilation failed for some solutions in problem {name}")
        return 0, 0
    
    args: list[tuple[str, list[Path], list[Path]]] = [
        (
            tc,
            correct_solutions,
            incorrect_solutions,
            timeout
        )
        for tc in testcases
    ]
    
    with Pool() as test_tc:
        test_result = test_tc.uimap(
            test_testcase,
            args,
        )
    results = [r for r in test_result]
    for r in results:
        if "Error" in r:
            return 0, 0
    effectiveness = summarize_and_score(results)
    return 1, effectiveness