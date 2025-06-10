import re
import os 
import tempfile
import random 
import itertools
from itertools import chain
from pathlib import Path
from typing import Any, Optional
import time
import py_compile
import timeout_decorator
from timeout_decorator.timeout_decorator import TimeoutError 
from rm2.grammar.counting_context_free_grammar import CountingContextFreeGrammar as Ccfg
from pathos.multiprocessing import ProcessingPool as Pool
import subprocess
import jsonlines
from typing import List, Dict, IO, TypedDict

SOLUTIONS_DIR = "./data/solutions"
PUBLIC_TESTCASE_DIR = "./data/testcase/code-contest/public/test.jsonl"
CORRECT_SOLUTIONS_DIR = f"{SOLUTIONS_DIR}/solutions"
INCORRECT_SOLUTIONS_DIR = f"{SOLUTIONS_DIR}/incorrect_solutions"

public_dataset = jsonlines.open(PUBLIC_TESTCASE_DIR, 'r')
with jsonlines.open(PUBLIC_TESTCASE_DIR, 'r') as public_dataset:
    timeout_dict: Dict[str, int] = {
        data["name"]: max(
            1,
            int(
                data["time_limit"]["seconds"] + data["time_limit"]["nanos"] / 1e9
            )
        )
        for data in public_dataset
    }

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
    
def get_testcase(
    ccfg: Ccfg,
    timeout: int,
    min_degree: int,
    k: int,
) -> list[tuple[str, int]]:
    """
    throw: Error
    """
    
    @timeout_decorator.timeout(timeout) 
    def _generate(degree: int) -> str:
        return ccfg.generate(degree=degree)  
    if min_degree == -1:
        assert k == 1
        val = _generate(-1)
        if is_invalid_terminal(val):
            raise ValueError(f"Invalid testcase generated: {val}")
        return [(val, -1)]
    
    degree = min_degree
    degrees = [degree] * k
    def _generate_parallel(degree: int) -> tuple[str, int]:
        while True:
            try:
                val = _generate(degree)
                if is_invalid_terminal(val):
                    raise ValueError(f"Invalid testcase generated: {val}")
                
                return (val, degree)
            except TimeoutError as e:
                if degree >= 2:
                    raise e
                degree += 1
    
    with Pool() as pool:
        testcases = pool.uimap(_generate_parallel, degrees)
    return testcases

def sample_solutions(
    name: str,
    num_samples: int = 10,
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
    num_samples = min(num_samples, len(solutions))
    return random.sample(solutions, num_samples)

class ExecutionResultPerTestcase(TypedDict):
    """ExecutionResultPerTestcase is a dictionary that contains the results of
    execution for a single testcase.
    """

    correct_outputs: list[Optional[str]]
    incorrect_outputs: list[Optional[str]]
    correct_solutions: list[str]
    incorrect_solutions: list[str]
    
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
        print(f"[RuntimeError] {python_file.name}: {e}")
        return None

    return " ".join(process.stdout.split()).lower()

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
        print(f"[CompileError] {solution_path.name}: {e}")
        return False

def efficiency_score(
    name: str,
    testcases: list[str],
    timeout: int,
):
    _k = 10
    correct_solutions = sample_solutions(
        name = name,
        num_samples = _k,
        correctness = "correct"
    )
    incorrect_solutions = sample_solutions(
        name = name,
        num_samples = _k,
        correctness = "incorrect"
    )

    
    all_solutions = list(chain(correct_solutions, incorrect_solutions))
    
    with Pool() as compile_pool:
        compiled_flags = compile_pool.uimap(precompile_solution, all_solutions)
    
    if not all([f for f in compiled_flags]):
        print(f"[Abort] Compilation failed for some solutions in problem {name}")
        return False, None
    
    results = []
    for idx, testcase in enumerate(testcases):
        temp_file = tempfile.TemporaryFile("w+b")
        temp_file.write(testcase.encode("utf-8"))
        temp_file.flush()

        correct_outputs = run_testcase(temp_file, correct_solutions, timeout)
        incorrect_outputs = run_testcase(temp_file, incorrect_solutions, timeout)
        temp_file.close()
        
        print(correct_outputs, incorrect_outputs)

        filtered_correct_outputs = [o for o in correct_outputs if o is not None]
        if not filtered_correct_outputs:
            print(f"[Empty] No valid correct outputs for {name}, testcase #{idx}")
            return False, None
        if len(set(filtered_correct_outputs)) > 1:
            print(f"[Mismatch] Correct outputs differ for {name}, testcase #{idx}")
            return False, None
        if any(i_out == filtered_correct_outputs[0] for i_out in incorrect_outputs if i_out is not None):
            print(f"[Conflict] Incorrect output matches correct for {name}, testcase #{idx}")
            return False, None

        result_per_testcase = ExecutionResultPerTestcase(
            correct_outputs=correct_outputs,
            incorrect_outputs=incorrect_outputs,
            correct_solutions=[str(e.name) for e in correct_solutions],
            incorrect_solutions=[str(e.name) for e in incorrect_solutions],
        )
        results.append(result_per_testcase)

    effectiveness = summarize_and_score(results)
    return True, effectiveness
    
        
def get_testcases(
        data: dict[str, str],
        k: int,
        timeout: int, 
    )-> Optional[tuple[list[str], list[int]]]:
    
    name = data["name"]
    grammar = data["grammar"]
    
    productions = grammar["productions"]
    constraints = grammar["constraints"]
    
    # Raise error if any regex production contains a '+' quantifier
    for prod in productions:
        if re.search(r"\[[^\]]+\]\+", prod) or re.search(r"\\[dws]\+", prod): # 
            raise ValueError(f"Invalid regex pattern with '+' found in production: {prod}")
        if re.search(r"\[[^\]]*\]\*", prod) or re.search(r"\\[dws]\*", prod):
            raise ValueError(f"Invalid regex pattern with '*' found in production: {prod}")
    
    ccfg = Ccfg(productions, constraints)
    testcases = []
    try:
        tuples = get_testcase(ccfg, timeout, -1, 1) # deterministic 
    except TimeoutError:
        tuples = []
        
    tuples += get_testcase(ccfg, timeout, 2, k)
    tuples += get_testcase(ccfg, timeout, 1, k)
    tuples += get_testcase(ccfg, timeout, 0, k)
    
    
    testcases: List[str] = [t[0] for t in tuples]
    degrees: List[int] = [t[1] for t in tuples] 
    
    timeout = 2 * timeout_dict[name] # x2 for safty margin
    
    validity, effectiveness = efficiency_score(
        name = name,
        testcases = testcases,
        timeout = timeout,
    )