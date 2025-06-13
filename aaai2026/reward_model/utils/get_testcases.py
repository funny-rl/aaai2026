import re
import os 
import jsonlines
from typing import List
import timeout_decorator
from typing import Optional
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

PUBLIC_TESTCASE_DIR = "reward_model/utils/data/testcase/code-contest/public/test.jsonl"

public_dataset = jsonlines.open(PUBLIC_TESTCASE_DIR, 'r')
with jsonlines.open(PUBLIC_TESTCASE_DIR, 'r') as public_dataset:
    timeout_dict: dict[str, int] = {
        data["name"]: max(
            1,
            int(
                data["time_limit"]["seconds"] + data["time_limit"]["nanos"] / 1e9
            )
        )
        for data in public_dataset
    }

def get_testcase(
    ccfg: Ccfg,
    timeout: int,
    min_degree: int,
    num_testcase: int,
) -> list[tuple[str, int]]:
    """
    throw: Error
    """
    
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
        name: str,
        num_testcase: int,
        n_testcode: int,
        timeout: int, 
    )-> Optional[tuple[list[str], list[int]]]:
    
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
    
    timeout = 2 * timeout_dict[name]
    
    validity, effectiveness = efficiency_score(
        name = name,
        n_sample = n_testcode,
        testcases = testcases,
        timeout = timeout,
    )
    return validity, effectiveness

