#!/usr/bin/env python3
import ast
import os
import sys
from typing import Dict, List, Any

def extract_parametrize_info(node: ast.FunctionDef) -> int:
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
                
                combinations = count_parametrize_combinations(decorator)
                total_combinations *= combinations
                
            # Handle @parametrize (direct import)
            elif (isinstance(decorator.func, ast.Name) and 
                  decorator.func.id == 'parametrize'):
                
                combinations = count_parametrize_combinations(decorator)
                total_combinations *= combinations
    
    return total_combinations if total_combinations > 1 else 0

def count_parametrize_combinations(decorator: ast.Call) -> int:
    """Count the number of test combinations from a parametrize decorator."""
    if len(decorator.args) < 2:
        return 1
    
    # Second argument contains the parameter values
    values_arg = decorator.args[1]
    
    if isinstance(values_arg, ast.List):
        return len(values_arg.elts)
    elif isinstance(values_arg, ast.Tuple):
        return len(values_arg.elts)
    else:
        # Could be a variable or more complex expression
        # For safety, assume 1 combination
        return 1

def analyze_file(filepath: str) -> Dict[str, Any]:
    """Analyze a Python file for parametrized tests."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        test_functions = []
        total_cases = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                case_count = extract_parametrize_info(node)
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
        # Analyze specific files provided as arguments
        files_to_analyze = sys.argv[1:]
    else:
        # Find all test files in current directory
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
