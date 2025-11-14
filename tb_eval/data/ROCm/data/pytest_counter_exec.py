#!/usr/bin/env python3
import ast
import os
import sys
import importlib.util
from typing import Dict, List, Any

def safely_evaluate_parametrize(decorator: ast.Call, module_globals: dict) -> int:
    """Safely evaluate parametrize arguments to get exact count."""
    if len(decorator.args) < 2:
        return 1
    
    try:
        # Compile and evaluate the second argument (parameter values)
        values_code = compile(ast.Expression(decorator.args[1]), '<parametrize>', 'eval')
        values = eval(values_code, module_globals)
        
        if isinstance(values, (list, tuple)):
            return len(values)
        else:
            return 1
    except:
        # Fall back to AST analysis if evaluation fails
        return count_parametrize_combinations_ast(decorator)

def count_parametrize_combinations_ast(decorator: ast.Call) -> int:
    """Fallback AST-based counting (from previous version)."""
    if len(decorator.args) < 2:
        return 1
    
    values_arg = decorator.args[1]
    
    if isinstance(values_arg, ast.List):
        return len(values_arg.elts)
    elif isinstance(values_arg, ast.Tuple):
        return len(values_arg.elts)
    elif isinstance(values_arg, ast.ListComp):
        return estimate_listcomp_size(values_arg)
    else:
        return estimate_complex_expression(values_arg)

def estimate_listcomp_size(listcomp: ast.ListComp) -> int:
    """Estimate the size of a list comprehension by analyzing its structure."""
    try:
        total_combinations = 1
        
        for generator in listcomp.generators:
            if isinstance(generator.iter, ast.Call):
                func_name = None
                if isinstance(generator.iter.func, ast.Name):
                    func_name = generator.iter.func.id
                
                if func_name == 'get_x_vals':
                    estimated_size = 10  # Adjust based on your actual function
                else:
                    estimated_size = 5
                
                total_combinations *= estimated_size
                
            elif isinstance(generator.iter, ast.List):
                total_combinations *= len(generator.iter.elts)
                
            elif isinstance(generator.iter, ast.Tuple):
                total_combinations *= len(generator.iter.elts)
                
            else:
                total_combinations *= 3
        
        return max(1, total_combinations)
    except:
        return 1

def estimate_complex_expression(expr: ast.AST) -> int:
    """Try to estimate the size of complex expressions."""
    if isinstance(expr, ast.Name):
        return 1
    elif isinstance(expr, ast.Call):
        return 5
    else:
        return 1

def load_module_safely(filepath: str) -> dict:
    """Load a Python module and return its globals, handling errors gracefully."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", filepath)
        if spec is None:
            return {}
        
        module = importlib.util.module_from_spec(spec)
        
        # Add the module's directory to sys.path temporarily
        module_dir = os.path.dirname(os.path.abspath(filepath))
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        
        try:
            spec.loader.exec_module(module)
            return module.__dict__
        finally:
            # Remove the directory from sys.path
            if module_dir in sys.path:
                sys.path.remove(module_dir)
                
    except Exception as e:
        print(f"Warning: Could not load module {filepath}: {e}")
        return {}

def extract_parametrize_info(node: ast.FunctionDef, module_globals: dict) -> int:
    """Extract parametrize information from a test function and return test case count."""
    total_combinations = 1
    
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call):
            # Handle pytest.mark.parametrize
            if (isinstance(decorator.func, ast.Attribute) and 
                isinstance(decorator.func.value, ast.Attribute) and
                isinstance(decorator.func.value.value, ast.Name) and
                decorator.func.value.value.id == 'pytest' and
                decorator.func.value.attr == 'mark' and
                decorator.func.attr == 'parametrize'):
                
                combinations = safely_evaluate_parametrize(decorator, module_globals)
                total_combinations *= combinations
                
            # Handle @parametrize (direct import)
            elif (isinstance(decorator.func, ast.Name) and 
                  decorator.func.id == 'parametrize'):
                
                combinations = safely_evaluate_parametrize(decorator, module_globals)
                total_combinations *= combinations
    
    return total_combinations if total_combinations > 1 else 0

def analyze_file(filepath: str) -> Dict[str, Any]:
    """Analyze a Python file for parametrized tests."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Try to load the module to get access to functions like get_x_vals()
        module_globals = load_module_safely(filepath)
        
        test_functions = []
        total_cases = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                case_count = extract_parametrize_info(node, module_globals)
                if case_count > 0:
                    test_functions.append({
                        'name': node.name,
                        'cases': case_count
                    })
                    total_cases += case_count
        
        return {
            'filepath': filepath,
            'test_functions': test_functions,
            'total_cases': total_cases,
            'error': None
        }
    
    except Exception as e:
        return {
            'filepath': filepath,
            'test_functions': [],
            'total_cases': 0,
            'error': str(e)
        }

def find_test_files(directory: str = '.') -> List[str]:
    """Find all Python test files in the directory."""
    test_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    return test_files

def main():
    if len(sys.argv) > 1:
        files_to_analyze = sys.argv[1:]
    else:
        files_to_analyze = find_test_files()
    
    if not files_to_analyze:
        print("No test files found.")
        return
    
    print(f"Analyzing {len(files_to_analyze)} file(s) for parametrized tests...\n")
    
    grand_total = 0
    files_with_parametrize = 0
    
    for filepath in files_to_analyze:
        result = analyze_file(filepath)
        
        if result['error']:
            print(f"❌ {result['filepath']}: Error - {result['error']}")
            continue
        
        if result['total_cases'] > 0:
            files_with_parametrize += 1
            grand_total += result['total_cases']
            
            print(f"📁 {result['filepath']}: {result['total_cases']} test cases")
            for func in result['test_functions']:
                print(f"  └── {func['name']}: {func['cases']} cases")
        else:
            print(f"📁 {result['filepath']}: No parametrized tests found")
    
    print(f"\n📊 Summary:")
    print(f"   Files analyzed: {len(files_to_analyze)}")
    print(f"   Files with parametrized tests: {files_with_parametrize}")
    print(f"   Total parametrized test cases: {grand_total}")

if __name__ == "__main__":
    main()
