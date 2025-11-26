# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import os
import json
import subprocess
import ast

from ..constants import REPO_ROOT, TMP_ROOT
import re

DEFAULT_TRITON_BENCH_ROOT = os.path.join(REPO_ROOT, "data", "TritonBench", "data", "TritonBench_G_v1")


def extract_collection_error(stderr_string: str) -> str:
    """
    Extracts the content of a pytest collection error block from a stderr string.

    This is designed for errors that happen during test discovery (e.g.,
    syntax errors), which are reported in an "ERRORS" block.

    Args:
        stderr_string: The complete stderr output as a string.

    Returns:
        A string containing the collection error block, or an empty string if
        the specific block is not found.
    """
    # Use a regular expression to find the content between the ERRORS
    # and "short test summary" markers.
    # re.DOTALL makes the '.' special character match any character, including newlines.
    pattern = re.compile(
        r"={10,}\s+ERRORS\s+={10,}(.*?)\n={10,}\s+short test summary info\s+={10,}",
        re.DOTALL
    )

    match = pattern.search(stderr_string)

    if match:
        # group(1) contains the text captured by (.*?)
        # .strip() removes leading/trailing whitespace and newlines.
        return match.group(1).strip()
    else:
        return ""

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

def extract_errors(stderr_string: str) -> str:
    """
    Extracts the primary error from a pytest stderr output, acting as an
    abstraction layer.

    It first checks for a fatal collection error. If none is found, it
    falls back to extracting the first runtime test failure.

    Args:
        stderr_string: The complete stderr output from a pytest run.

    Returns:
        A string containing the most relevant error block, or an empty string
        if no errors are found.
    """
    # Priority 1: Check for collection errors, as they are fatal and
    # prevent tests from running.
    collection_error = extract_collection_error(stderr_string)
    if collection_error:
        return collection_error

    # Priority 2: If no collection errors, check for standard runtime failures.
    runtime_failure = extract_first_pytest_failure(stderr_string)
    if runtime_failure:
        return runtime_failure
        
    # If neither type of error is found, return an empty string.
    return ""
    
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
