import os
import json
import subprocess
import ast

from ..constants import REPO_ROOT, TMP_ROOT

DEFAULT_TRITON_BENCH_ROOT = os.path.join(REPO_ROOT, "data", "TritonBench", "data", "TritonBench_G_v1")

def get_fname_difficulty_from_label(label):
    triton_root = DEFAULT_TRITON_BENCH_ROOT
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
