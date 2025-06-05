import os
import json
import subprocess
import ast

from ..constants import REPO_ROOT, TMP_ROOT
import re

DEFAULT_TRITON_BENCH_ROOT = os.path.join(REPO_ROOT, "data", "TritonBench", "data", "TritonBench_G_v1")



def extract_first_pytest_failure(stderr_string: str) -> str:
    """
    Extracts the content of the first pytest failure block from a stderr string.

    Args:
        stderr_string: The complete stderr output as a string.

    Returns:
        A string containing the first failure block, or an empty string if
        no failure blocks are found.
    """
    lines = stderr_string.splitlines()
    
    # Regex to match the pytest failure start line pattern
    # e.g., ___________________ test_correctness[...] ___________________
    failure_start_pattern = re.compile(r'^_{3,} test_.* _{3,}$')
    
    first_start_index = -1
    # Find the index of the first failure marker
    for i, line in enumerate(lines):
        if failure_start_pattern.match(line):
            first_start_index = i
            break # Found the first one

    if first_start_index == -1:
        # No failure markers found
        return ""

    next_start_index = -1
    # Find the index of the next failure marker *after* the first one
    for i in range(first_start_index + 1, len(lines)):
        if failure_start_pattern.match(lines[i]):
            next_start_index = i
            break # Found the start of the next one

    # Extract the lines for the first failure block
    if next_start_index != -1:
        extracted_lines = lines[first_start_index : next_start_index]
    else:
        # If no next failure marker is found, extract till the end
        extracted_lines = lines[first_start_index :]

    return "\n".join(extracted_lines)

def get_fname_difficulty_from_label(label):
    # triton_root = DEFAULT_TRITON_BENCH_ROOT
    triton_root = os.path.join(REPO_ROOT, "data", "TritonBench", "data", "TritonBench_G_comp_alpac_v1_fixed_with_difficulty.json")
    with open(triton_root, 'r') as f:
        data = json.load(f)
        for item in data:
            if item['output'] == label:
                return item['file'], item['difficulty']
    return None, None

def run_shell(command, cwd=None, env=None, timeout=None):
    """
    Run a shell command and return the output.
    """
    if cwd is None:
        cwd = REPO_ROOT
    if env is None:
        env = os.environ.copy()
    
    result = subprocess.run(command, shell=True, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    status = result.returncode == 0
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    return status, stdout, stderr


class TestFunctionRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name.startswith('test_'):
            return None  # Kill the function
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id.startswith('test_'):
                return None  # Kill expressions like test_foo()
            if isinstance(func, ast.Attribute) and func.attr.startswith('test_'):
                return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        # If the value being assigned is a call to test_ function, kill the entire assignment
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id.startswith('test_'):
                return None
            if isinstance(func, ast.Attribute) and func.attr.startswith('test_'):
                return None
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        # For augmented assignments like x += test_func()
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id.startswith('test_'):
                return None
            if isinstance(func, ast.Attribute) and func.attr.startswith('test_'):
                return None
        return self.generic_visit(node)

    def visit_Module(self, node):
        # Manually rebuild body without None's
        node.body = [stmt for stmt in map(self.visit, node.body) if stmt is not None]
        return node

    def visit_ClassDef(self, node):
        node.body = [stmt for stmt in map(self.visit, node.body) if stmt is not None]
        return node

def strip_test_functions(source_code):
    tree = ast.parse(source_code)
    remover = TestFunctionRemover()
    tree = remover.visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

def process_code(code: str):
    if "```python" in code:
        code = code.split("```python")[-1].replace("<|im_end|>", "").replace("<|EOT|>", "")    
    try:
        code = strip_test_functions(code)
    except Exception as e:
        pass    
    return code
