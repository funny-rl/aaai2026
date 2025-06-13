
def basic_instruction(content, modelname):
    role_instruction = '''You are an expert grammar generator. Strictly generate the grammars using specification.'''

    output_instruction = '''Make sure the <Grammar> for the given last <Specification> which follows the general rules and examples provided. Output in a json format. Json format should be {"productions": [""], "constraints": [""]}.'''

    action_instruction = '''
The following are the examples for the generation of the grammar and constraint along with their reasons of how they are generated: <Specification> Constraints\n\n* -1000 ≤ a, b ≤ 1000\n\nInput\n\nTwo integers a and b separated by a single space are given in a line. </Specification> <Reason> "The grammar construction begins with the initial non-terminal <S>. It defines two variables, 'a' and 'b', which are separated by a space symbol denoted as <s>. The structure of the grammar is formulated as '<S>->a <s> b', representing the input format where 'a' and 'b' are two integers separated by a space." </Reason> <Grammar> "{\"productions\": [\"<S>->a <s> b\"], \"constraints\": [\"-1000<=a<=1000\", \"-1000<=b<=1000\"]}" </Grammar>
<Specification> "Constraints\n\n* 1 \\leq N \\leq 100\n* 1 \\leq A_i \\leq 1000\n\nInput\n\nInput is given from Standard Input in the following format:\n\n\nN\nA_1 A_2 \\ldots A_N" </Specification> <Reason> "The grammar begins with the start symbol <S>. Here, 'N' is used as a counter variable for the array elements, thus it is denoted as [N] in the grammar to reflect its role as a counter. If 'N' were not a counter, it would be represented simply as 'N'. All variables that serve as counters are similarly denoted with brackets. Consequently, the grammar is initiated with '<S>->[N] <n> <T_N>'. The non-terminal <T_i> is a counter-driven rule for [N], where 'A_i' represents each array element separated by a space symbol <s>. The grammar constructs are laid out as '<S>->[N] <n> <T_N>', '<T_i>-><T_i-1> <s> A_i', and '<T_1>->A_1' to systematically parse the sequence of array elements following the count [N]." </Reason> <Grammar> "{\"productions\": [\"<S>->[N] <n> <T_N>", "<T_i>-><T_i-1> <s> A_i\", \"<T_1>->A_1\"], \"constraints\": [\"1<=n<=100\", \"1<=A_i<=1000\"]}" </Grammar>
<Specification> "Ania has a large integer S. Its decimal representation has length n and doesn't contain any leading zeroes. Ania is allowed to change at most k digits of S. She wants to do it in such a way that S still won't contain any leading zeroes and it'll be minimal possible. What integer will Ania finish with?\n\nInput\n\nThe first line contains two integers n and k (1 \u2264 n \u2264 200 000, 0 \u2264 k \u2264 n) \u2014 the number of digits in the decimal representation of S and the maximum allowed number of changed digits.\n\nThe second line contains the integer S. It's guaranteed that S has exactly n digits and doesn't contain any leading zeroes." </Specification> <Reason> "The grammar begins with the starting non-terminal <S>. It inlcudes two variables n and k, but n  serves as the counter variable so we enclose in [n]. The grammar is initiated with <S>->[n] <s> k <n> <T_n>. Since it does not have any leading zeroes so, <T_i>->[1-9]{1} <B_i-1> is used reflect this and <B_i> is used to reflect all the other numbers including zero." </Reason> <Grammar> "{\"productions\": [\"<S>->[n] <s> k <n> <T_n>\", \"<T_i>->[1-9]{1} <B_i-1>\", \"<T_1>->[1-9]{1}\", \"<B_i>->[0-9]{1} <B_i-1>\", \"<B_1>->[0-9]{1}\"], \"constraints\": [\"1<=n<=200000\", \"0<=k<=n\"]}" </Grammar>
<Specification> "Input\n\nThe first line contains a single integer n (1 ≤ n ≤ 10^6) — the length of Dima's sequence.\n\nThe second line contains string of length n, consisting of characters \"(\" and \")\" only." </Specification> <Reason> "The grammar begins with the starting non-terminal <S>. It includes one variable, n, which is used both as a counter and to specify the length of a sequence. Consequently, the variable is represented as [n] to indicate its role as a length specifier for the sequence of characters, which consist only of '(' and ')'. This sequence is described using the regular expression [()]{n}, meaning a string of n characters, each of which is either '(' or ')'. Thus, the grammar is constructed to reflect this format: "<S>->[n] <n> [()]{n}"."</Reason> <Grammar> "{\"productions\": [\"<S>->[n] <n> [()]{n}\"], \"constraints\": [\"1<=n<=10^6\"]}" </Grammar>
<Specification> "Input\n\nThe first line of the input contains two integers n and m (1 ≤ n, m ≤ 100) — the number of floors in the house and the number of flats on each floor respectively.\n\nNext n lines describe the floors from top to bottom and contain 2·m characters each. If the i-th window of the given floor has lights on, then the i-th character of this line is '1', otherwise it is '0'." </Specification> <Reason> "The formation of the grammar starts with the start non-terminal <S>, there is two variables that is n and m and both of them serves as the counter variables, hence the variables are converted to [n] and [m]  which then makes <T_n> that is separated by the new line <n>.  If variable n and m was not the counter variable we will write as n and m only not [n] and [m]. We convert all the variables that serves as a counter variable also to "[variable]".  Hence,  the grammar starts with "<S>->[n] <n> [m] <n>  <T_N>". The <T_i>  is the counter non-terminal of the counter variable [n]  having  <L_2m> counter non-terminal of size 2m which is separated by the new line token <n>.  The <L_i> has only 0 and 1 of length one using the regular expression as  [01]{1}, so we represent the length in {}" </Reason> <Grammar> "{\"productions\": [\"<S>->[n] <s> [m] <n> <T_n>\", \"<T_i>-><T_i-1> <n> <L_2m>\",\"<T_1>-><L_2m>\", \"<L_i>-><L_i-1> <s> [01]{1}\"], \"constraints\": [\"1<=n,m<=100\"]}" </Grammar>
'''
    if "DeepSeek-R1-Distill" in modelname:
        return [
            {
                "role": "user", 
                "content":role_instruction+ action_instruction + output_instruction + f'<Specification> "{content}" </Specification>'
            }
        ]
    else:
        raise ValueError(f"Model {modelname} is not supported for basic instruction generation.")
    
def get_specification(description: str) -> str:
        constraints_start_token = '\nconstraints\n'
        input_start_token = '\ninput\n'
        end_token = '\noutput\n'
        constraints_start = description.lower().find(constraints_start_token)
        input_start = description.lower().find(input_start_token)
        end = description.lower().find(end_token)
        constraints_start = (
            len(description) if constraints_start < 0 else constraints_start)
        input_start = len(description) if input_start < 0 else input_start
        start = min(constraints_start, input_start)
        start = 0 if start == len(description) else start
        end = len(description) if end < 0 else end
        specification = description[start:end].strip()
        return specification

def replace_description(description: str) -> str:
        description_replacements = [
            ('\\cdots', '...'),
            ('\\dots', '...'),
            ('\\geq', '>='),
            ('\\ge', '>='),
            ('\\gt', '>'),
            ('\\ldots', '...'),
            ('\\leq', '<='),
            ('\\le', '<='),
            ('\\lt', '<'),
            ('\\neq', '!='),
            ('\\ne', '!='),
            ('\\times', '*'),
            ('\u2013', '--'),
            ('\u2014', '---'),
            ('\u2019', "'"),
            ('\u2026', '...'),
            ('\u2192', "->"),
            ('\u2208', "in"),
            ('\u2211', "sum"),
            ('\u2212', '-'),
            ('\u2260', "!="),
            ('\u2264', '<='),
            ('\u2265', '>='),
            ('\u2266', '<='),
            ('\u2295', "xor"),
            ('\u22c5', '*'),
            ('\u2308', ""),
            ('\u2309', ""),
        ]
        for old, new in description_replacements:
            description = description.replace(old, new)
        return description

def get_prompt_func(
    modelname: str,
    desc: str,
)-> list[dict[str, str]]:
    """
    Get the prompt function for generating basic instructions.
    Args:
        modelname (str): The name of the model.
        desc (str): The description to be processed.
    Returns:
        list[dict]: A list containing the prompt.
    """
    #spec = get_specification(replace_description(desc)) # get the specification from description
    spec = replace_description(desc) # get all description
    return basic_instruction(spec, modelname)