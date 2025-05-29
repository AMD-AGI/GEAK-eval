import os
import shutil
import json
from .base import BasePerfEval
from ..helpers.helper import run_shell
from ..constants import TBG_PERF_GOLD_DATA_ROOT, ROCm_ROOT

class PerformanceEvalTBG(BasePerfEval):
    """
    Performance evaluation for the TBG model.
    This class inherits from BasePerfEval and implements the evaluate method.
    """
    ref_folder = TBG_PERF_GOLD_DATA_ROOT
    def __init__(self, name :str='PerformanceEvalTBG'):
        """
        Initialize the PerformanceEvalTBG instance.
        
        Args:
            name (str): The name of the performance evaluation instance.
        """
        super().__init__(name=name)

    def evaluate(self, exec_folder: str, gen_perf_folder: str=None, golden_metrics_folder:str=None) -> dict:
        """
        Evaluate the performance of the TBG model on the given data.
        
        Args:
            folder: Root location with kernels to evaluate.
        
        Returns:
            A dictionary containing the evaluation results.
        """
        
        ref_folder = self.ref_folder        
        print(f"Running performance analysis for {exec_folder}")
        assert os.path.exists(exec_folder), f"Execution folder {exec_folder} does not exist."

        gen_perf_folder = os.path.join(exec_folder, 'gen_perf') if gen_perf_folder is None else gen_perf_folder
        ## if gen_perf_folder exists, remove it
        if os.path.exists(gen_perf_folder):
            print(f"Removing existing performance folder: {gen_perf_folder}")
            shutil.rmtree(gen_perf_folder)
        os.makedirs( gen_perf_folder, exist_ok=True)

        exec_folder = os.path.abspath(exec_folder)
        gen_perf_folder = os.path.abspath(gen_perf_folder)

        curr_dir = os.path.dirname(os.path.abspath(__file__))

        print("Writing files to the performance folder...")
        cmd = [f'python3 {curr_dir}/run_bench/write_file.py --input_folder_path {exec_folder} --result_folder_path {gen_perf_folder}']
        if golden_metrics_folder:
            cmd[-1] += f' --golden_metrics_folder {golden_metrics_folder}'

        write_status, write_stdout, write_stderr = run_shell(cmd)
        print(f"Write status: {write_status}, stdout: {write_stdout}, stderr: {write_stderr}")
        perf_stdout = None

        if write_status:
            print("Files written successfully to the performance folder. Running them...")
            cmd = [f"python3 {curr_dir}/run_bench/multiprocess_gpu_run.py --root_dir {gen_perf_folder}"]
            mp_run_status, mp_run_stdout, mp_run_stderr = run_shell(cmd)

            if mp_run_status:
                print("Multiprocess GPU run completed successfully. Running performance analysis...")
                cmd = [f"python3 {curr_dir}/2_efficiency.py --gen_folder {gen_perf_folder} --ref_folder {ref_folder}"]
                perf_status, perf_stdout, perf_stderr = run_shell(cmd)
                
                if perf_status:
                    print(f"Performance analysis completed successfully for {exec_folder}.")
                    with open(os.path.join(exec_folder, 'performance_analysis.txt'), 'w') as f:
                        f.write(f"Performance analysis for {exec_folder}:\n")
                        f.write(perf_stdout)
                else:
                    assert False, f"Failed to run 2_efficiency.py: {perf_stderr}"
            else:
                assert False, f"Failed to run multiprocess_gpu_run.py: {mp_run_stderr}"

        else:
            assert False, f"Failed to write files: {write_stderr}"
        print(f"DONE with performance analysis for {exec_folder}")

        parser_perf_data = self.parse(gen_perf_folder)

        return parser_perf_data

    def parse(self, perf_data_path:str) -> dict:
        eff_fname = os.path.join(perf_data_path, 'efficiency.json')
        parsed_perf_data = {}
        if os.path.exists(eff_fname):
            with open(eff_fname, 'r') as f:
                perf_data = json.load(f)
            parsed_perf_data = perf_data
        return parsed_perf_data
    
class PerformanceEvalROCm(PerformanceEvalTBG):
    ref_folder = ROCm_ROOT
    pass


get_perf_evaluators = {
    'tbg': PerformanceEvalTBG,
    'rocm': PerformanceEvalROCm
}