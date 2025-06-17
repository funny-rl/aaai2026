import re
import os 
from typing import List
import timeout_decorator
from pathos.multiprocessing import ProcessingPool as Pool

from reward_model.utils.grammar.counting_context_free_grammar import CountingContextFreeGrammar as Ccfg

from reward_model.utils.utils import (
    is_invalid_terminal, 
    efficiency_score,
)

import warnings
warnings.filterwarnings("ignore")

# PUBLIC_TESTCASE_DIR = os.getenv("PUBLIC_TESTCASE_DIR")
# if PUBLIC_TESTCASE_DIR is None:
#     raise ValueError("Environment variable PUBLIC_TESTCASE_DIR is not set")

def get_testcase(
    ccfg: Ccfg,
    timeout: int,
    min_degree: int,
    num_testcase: int,
) -> list[tuple[str, int]]:
    @timeout_decorator.timeout(timeout) 
    def _generate(degree: int) -> str:
        return ccfg.generate(degree=degree)  
    if min_degree == -1:
        assert num_testcase == 1
        val = _generate(-1)
        if is_invalid_terminal(val):
            raise ValueError(f"Invalid testcase generated: {val}")
        return [(val, -1)]
    
    degree = min_degree
    degrees = [degree] * num_testcase
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


def get_testcases(
        data: dict[str, str],
        num_testcase: int,
        timeout: int, 
    ) ->  List[str]:
    
    productions = data["productions"]
    constraints = data["constraints"]
    
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

    tuples += get_testcase(ccfg, timeout, 2, num_testcase-1)
    tuples += get_testcase(ccfg, timeout, 1, num_testcase)
    testcases: List[str] = [t[0] for t in tuples]
    return testcases

def get_efficiency_score(
        name: str,
        testcases: List[str],
) -> tuple[float, float, int, int]:
    validity, effectiveness, n_correct_solution, n_incorrect_solution = efficiency_score(
        name = name,
        testcases = testcases,
    )
    return validity, effectiveness, n_correct_solution, n_incorrect_solution

from statistics import mean
from reward_model.utils.grammar.discriminator import discriminator
from reward_model.utils.grammar.counting_context_free_grammar import CountingContextFreeGrammar as Ccfg

def check_syntactic_validness(testcase: str, gt_grammar: dict) -> bool:
    @timeout_decorator.timeout(10)  
    def _check_syntactic_validness(testcase: str, grammar: dict) -> bool:
        d = discriminator()
        productions = grammar["productions"]
        constraints = grammar["constraints"]
        return d(productions, constraints, testcase)  
    try:
        return _check_syntactic_validness(testcase, gt_grammar)
    except Exception as e:  
        return False
def calculate_validity(testcases: list[str], gt_grammar: dict) -> float:
    parsable_cases = [check_syntactic_validness(testcase, gt_grammar) for testcase in testcases]
    validity = mean(parsable_cases)
    return validity