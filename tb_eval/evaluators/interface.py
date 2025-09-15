# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import os, sys, json
import subprocess
from shutil import copyfile
from .base import BaseEvaluator
from ..helpers import get_temp_file, get_rocm_temp_file
from ..helpers.helper import run_shell, process_code, extract_errors, extract_json_from_stdout
from ..processors.llm import LLMOutputProcessor
from ..constants import REPO_ROOT, TMP_ROOT, TBG_DATA_ROOT, ROCm_DATA_ROOT, Names, NATIVE_PERF_GOLD_ROOT
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

class TestAllCloseEvaluatorTBG(BaseEvaluator):
    def __init__(self, ground_truth_root :str = TBG_DATA_ROOT, MODULE_DIR :str =_MODULE_DIR) -> None:
        super().__init__("CorrectnessWithAllCloseEvaluator")

        self.ground_truth_root = ground_truth_root
        self.MODULE_DIR = MODULE_DIR
        

        self.llm_output_processor = LLMOutputProcessor()

    def get_gen_fpath(self, root :str, fname :str) -> str:
        """
        Returns the path to the generated file based on the given filename.
        """
        tmp_gen_folder = os.path.join(root, "tmp", Names.GEN_FOLDER)
        os.makedirs(tmp_gen_folder, exist_ok=True)
        gen_file = get_temp_file(prefix=f'{fname}{Names.GEN_SUFFIX}')
        return os.path.join(tmp_gen_folder, gen_file)

    def get_ground_truth_fpath(self, fname:str ) -> str:
        """
        Returns the path to the ground truth file based on the given filename.
        """
        return os.path.join(self.ground_truth_root, fname)

    def get_tests_code(self, fname :str) -> list[str]:
        with open(fname, 'r') as f:
            lines = f.readlines()
            for iL, line in enumerate(lines):
                if line.strip() == self.tests_sep_line:
                    break
            test_code_lines = lines[iL+1:]
            assert len(test_code_lines) > 0, f"No test code found in {fname} with test line seperator {self.tests_sep_line}"
        return test_code_lines

    def format_gen_code(self, fname:str, code:str, tests:list[str]) -> str:
        code =  code + '\n\n' + self.tests_sep_line + '\n' + '\n' + '\n'.join(tests)
        with open(fname, 'w') as f:
            f.write(code)
        return code
    
    def _call_file(self, fpath :str, timeout :int =2*60) -> tuple[bool, str, str]:
        ## Just to a simple call to the generated code
        cmd = [f'python3 {fpath}']
        call_status, stdout, stderr = run_shell(cmd, timeout=timeout)
        return call_status, stdout, stderr

    def _check_match(self, gen_fpath :str, ref_fpath:str , atol :float=1e-3, rtol :float=1e-1, timeout :int =2*60) -> tuple[bool, str, str]:
        """
        Executes the generated file and checks its correctness against the reference file.
        """
        cmd = [f'python3 {self.MODULE_DIR}/TB_correctness.py --gen_file {gen_fpath} --ref_file {ref_fpath} --atol {atol} --rtol {rtol}']
        status, stdout, stderr = run_shell(cmd, timeout=timeout)
        with open(gen_fpath+".stdout", 'w') as f:
            f.write(stdout)

        with open(gen_fpath+".stderr", 'w') as f:
            f.write(stderr)
        return status, stdout, stderr

    def execute(self, code :str, log_root:str, exec_root:str, fname:str, atol :float=1e-3, rtol:float=1e-1, timeout :int =2*60, verbose :bool =False, custom_tests_path=None) -> tuple[bool, bool, str, str]:
        
        speedup = 0

        triton_file = self.get_ground_truth_fpath(fname) 

        gen_file = self.get_gen_fpath(log_root, fname)

        test_code_lines_procs = self.get_tests_code(triton_file)

        code = process_code(code)

        code = self.format_gen_code(gen_file, code, test_code_lines_procs)

        try:
            call_status, stdout, stderr = self._call_file(gen_file, timeout=timeout)

            if not call_status:
                return call_status, False, speedup, stdout, stderr

            ## copy the generated file and its perf file to exec_fpath
            exec_fpath = os.path.join(exec_root, fname)
            os.makedirs(exec_root, exist_ok=True)
            with open(exec_fpath, 'w') as f:
                f.write(code)
            
            perf_fname = f"{NATIVE_PERF_GOLD_ROOT}/{fname.replace('.py', '_perf.py')}"
            assert os.path.exists(perf_fname), f"Performance file {perf_fname} does not exist. Please check the ground truth data."

            exec_perf_fpath = os.path.join(exec_root, os.path.basename(perf_fname))
            copyfile(perf_fname, exec_perf_fpath)
            
            cmd_status, perf_stdout, perf_stderr = self._call_file(exec_perf_fpath, timeout=timeout)

            if not cmd_status:
                if verbose:
                    print(f"Performance file execution failed: {perf_stderr}")
                return call_status, False, speedup, perf_stdout, perf_stderr
            
            exec_status = "Failed to run benchmark for input tensor." not in perf_stdout #perf_stdout.split(",")[0].strip().lower() == str(True).lower()

            if not exec_status:
                if verbose:
                    print(f"Performance file execution failed: {perf_stderr}")
                return call_status, False, speedup, perf_stdout, perf_stderr

            perf_data = extract_json_from_stdout(perf_stdout)

            if not perf_data:
                if verbose:
                    print(f"Performance data extraction failed: {perf_stderr}")
                return call_status, exec_status, speedup, perf_stdout, perf_stderr

            speedup = perf_data[-1].get("speedup", 0)
            return call_status, exec_status, speedup, perf_stdout, perf_stderr

        except Exception as e:
            if verbose:
                print(f"File: {fname}, Execution error: {e}")
            return False, False, 0, None, str(e)

        except subprocess.TimeoutExpired:
            if verbose:
                print(f"File: {fname} timed out!")
            return False, False, 0, None, "Time out"


class TestAllCloseEvaluatorROCm(TestAllCloseEvaluatorTBG):
    def __init__(self, ground_truth_root :str = ROCm_DATA_ROOT, MODULE_DIR :str =_MODULE_DIR) -> None:
        super().__init__("CorrectnessWithAllCloseEvaluator")

        self.ground_truth_root = ground_truth_root
        self.MODULE_DIR = MODULE_DIR
        
        self.llm_output_processor = LLMOutputProcessor()
    
    def get_ground_truth_fpath(self, fname:str ) -> str:
        """
        Returns the path to the ground truth file based on the given filename.
        """
        return os.path.join(self.ground_truth_root, fname)
    
    

    def get_fpath(self, root :str, fname :str, suffix: str) -> str:
        """
        Returns the path to the generated file based on the given filename.
        """
        tmp_gen_folder = os.path.join(root, "tmp", Names.GEN_FOLDER)
        os.makedirs(tmp_gen_folder, exist_ok=True)
        file = get_rocm_temp_file(prefix=f'{fname}{suffix}')
        return os.path.join(tmp_gen_folder, file)
    
    def get_tests_code(self, fname :str) -> list[str]:
        with open(fname, 'r') as f:
            content = f.read()

        snippets = content.split(self.tests_sep_line) 
        assert len(snippets) == 2, f"No test code found in {fname} with test line seperator {self.tests_sep_line}"  
 
        test_code_lines_procs = snippets[1].strip()
        return test_code_lines_procs
    
    def format_gen_code(self, fname:str, code:str, tests:list[str]) -> str:
        code =  code + '\n\n' + '#'*146 + '\n\n' + tests
        with open(fname, 'w') as f:
            f.write(code)
        return code
    
    def _check_match(self, gen_fpath :str, ref_fpath:str , atol :float=1e-2, rtol :float=1e-2, timeout :int =60*60) -> tuple[bool, str, str]:
        """
        Executes the generated file and checks its correctness against the reference file.
        """
        cmd = [f'python3 {self.MODULE_DIR}/ROCm_correctness.py --gen_file {gen_fpath} --ref_file {ref_fpath} --atol {atol} --rtol {rtol} --global_timeout {timeout} --verbose']
        status, stdout, stderr = run_shell(cmd, timeout=None if timeout is None else timeout)
        with open(gen_fpath+".stdout", 'w') as f:
            f.write(stdout)

        stderr += extract_errors(stdout.split(Names.PYTEST_SEPARATOR)[0])
        with open(gen_fpath+".stderr", 'w') as f:
            f.write(stderr)
        return status, stdout, stderr
    
    def execute(self, code :str, log_root:str, exec_root:str, fname:str, atol :float=1e-3, rtol:float=1e-1, timeout :int =40*60, custom_tests_path=None, verbose :bool =False) -> tuple[bool, bool, str, str]:
        
        triton_file = self.get_ground_truth_fpath(fname) 

        gen_file = self.get_fpath(log_root, fname, Names.GEN_SUFFIX)

        ref_file = self.get_fpath(log_root, fname, Names.REF_SUFFIX)

        copyfile(triton_file,ref_file)
    
        test_code_lines_procs = self.get_tests_code(triton_file)

        if custom_tests_path is not None and "@triton.autotune" in code:
            custom_tests_file = os.path.join(custom_tests_path, fname)
            if os.path.exists(custom_tests_file):
                test_code_lines_procs = self.get_tests_code(custom_tests_file)
            else:
                if verbose:
                    print(f"Custom tests file {custom_tests_file} does not exist. Skipping custom tests.", file=sys.stderr)

        code = process_code(code)
        code = self.format_gen_code(gen_file, code, test_code_lines_procs)

        try:
            call_status, stdout, stderr = False, None, None 

            # Check for correctness
            match_status = False
            atol, rtol = 1e-2, 1e-2
            match_status, stdout, stderr = self._check_match(gen_file, ref_file, atol=atol, rtol=rtol, timeout=timeout)

            # Check if the generated code executed successfully
            if not match_status:
                if verbose:
                    print(f"Error in generated code: {stderr}")
                stderr += "Error in generate triton-kernel code :\n " + extract_errors(stdout.split(Names.PYTEST_SEPARATOR)[0])
                return call_status, False, stdout, stderr
            else:
                if verbose:
                    print(f"Success in generated code: {stdout}")
                call_status_str, exec_status_str, gen_stdout, gen_stderr = stdout.split(Names.PYTEST_SEPARATOR)[-1].split(Names.RET_SEPERATOR)
                gen_stderr += extract_errors(stdout.split(Names.PYTEST_SEPARATOR)[0])
                call_status = call_status_str.replace('\n', '').strip().lower() == str(True).lower()
                
                # Original logic for exec_status, ensure it depends on the boolean call_status
                # If call_status is False, exec_status should also be False.
                # If call_status is True, then exec_status depends on its own string value.
                if call_status:
                    exec_status = exec_status_str.strip().lower() == str(True).lower()
                else:
                    exec_status = False

                if exec_status:
                    ## The generated code executed successfully, save the file in exec folder
                    exec_fpath = os.path.join(exec_root, fname)
                    os.makedirs(exec_root, exist_ok=True)
                    with open(exec_fpath, 'w') as f:
                        f.write(code)

                
                return call_status, exec_status, gen_stdout, gen_stderr

        except Exception as e:
            if verbose:
                print(f"File: {fname}, Execution error: {e}")
            return False, False, None, str(e)

        except subprocess.TimeoutExpired:
            if verbose:
                print(f"File: {fname} timed out!")
            return False, False, None, "Time out"
get_evaluators = {
    "tbg": TestAllCloseEvaluatorTBG,
    "rocm": TestAllCloseEvaluatorROCm
}