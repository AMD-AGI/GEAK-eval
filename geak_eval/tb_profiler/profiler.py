import os, re
import tempfile
import subprocess

def get_temp_file(prefix='', suffix='.py'):
    """Generate a temporary file with a given prefix."""
    fname = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False).name.strip()    
    return fname

def get_temp_bash_file(prefix=''):
    """Generate a temporary bash file with a given prefix."""
    return get_temp_file(prefix=prefix, suffix='.sh')

class BaseProfiler: 
    def __init__(self, ):
        pass

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)

    def run(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses")

    @staticmethod
    def parse_profiler_content(profile_content):
        delimiter = "--------------------------------------------------------------------------------"
        
        parts = profile_content.split(delimiter)
        
        section_data = {}

        section_pattern = re.compile(r"^\s*(\d+)\..*$", re.MULTILINE)

        for part in parts:
            trimmed_part = part.strip()
            if not trimmed_part:
                continue
                
            match = section_pattern.search(trimmed_part)
            if match:
                section_number = match.group(1)
                full_section_content = delimiter + part
                section_data[section_number] = full_section_content
                
        return section_data

class ROCmBenchmarkProfiler(BaseProfiler):
    def __init__(self, ):
        self.code_bash_format = "python {gen_file} --kernel_call"

    def run(self, code_or_path, fname, py_folder, target_gpu="MI300", temp_root="tmp2", atol=1e-3, rtol=1e-1, timeout=6*60, verbose=False):

        if not code_or_path:
            if verbose:
                print(f"File: {fname}, No code provided for profiling.")
            return None, None, "No code provided", None
        
        tmp_gen_folder = os.path.join(temp_root, "gen")
        os.makedirs(tmp_gen_folder, exist_ok=True)
        fname_split = fname.split('.')[0]
        
        triton_root = py_folder
        triton_file = os.path.join(triton_root, fname)
        if code_or_path.endswith(".py") and  os.path.exists(code_or_path):
            if verbose:
                print(f"File: {fname}, Using code from path: {code_or_path}")
            gen_file = code_or_path
            gen_bash_file = gen_file.replace('.py', '.sh')
        else:            
            if verbose:
                print(f"File: {fname}, Using provided code directly.")
            code = code_or_path

            gen_file = get_temp_file(prefix=f'{fname}_gen_triton_code')
            gen_file = os.path.join(tmp_gen_folder, gen_file)
            
            gen_bash_file = get_temp_bash_file(prefix=f'{fname_split}_gen_triton_code')
            gen_bash_file = os.path.join(tmp_gen_folder, gen_bash_file)

            hash_line = "#"*146

            with open(triton_file, 'r') as f:
                lines = f.readlines()
                for iL, line in enumerate(lines):
                    if line.strip() == hash_line:
                        break
                test_code_lines = lines[iL+1:]
                test_code_lines_procs = test_code_lines

            # code = process_code(code)

            code =  code + '\n\n' + hash_line + '\n' + '\n' + '\n'.join(test_code_lines_procs)
            
            # code_bash = f"pytest {gen_file}::test_performance"

            with open(gen_file, 'w') as f:
                f.write(code)

            
        code_bash = self.code_bash_format.format(gen_file=gen_file)
        with open(gen_bash_file, 'w') as f:
            f.write(code_bash)
        try:
            ## Just to a simple call to the generated code
            cmd_result = f'rocprof-compute profile -n {fname_split}  -- /bin/bash {gen_bash_file}'
            print(f"File: {fname}, Running command: {cmd_result}")
            result_profile = subprocess.run([cmd_result], capture_output=True, text=True, timeout=None, shell=True)

            cmd_analyze = f'rocprof-compute analyze -p workloads/{fname_split}/{target_gpu}'
            print(f"File: {fname}, Command executed successfully, now analyzing the profile. with command: {cmd_analyze}")
            analyze_profile = subprocess.run([cmd_analyze], capture_output=True, text=True, timeout=None, shell=True)
            
            # abstract profiling info
            profile_status = result_profile.returncode == 0
            stdout_profile = result_profile.stdout
            stderr_profile = result_profile.stderr
        
        except Exception as e:
            if verbose:
                print(f"File: {fname}, Execution error: {e}")
            return None, None, str(e), None
        
        # Clean up the temporary file
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"File: {fname} timed out!")
            return None, None, "Time out", None
        finally:
            pass
        
        # Check if the generated code executed successfully
        if result_profile.returncode != 0:
            if verbose:
                print(f"Error in profiling kernel: {result_profile.stderr}")
        else:
            if verbose:
                print(f"Success in in profiling kernel")
        try:
            section_text = self.parse_profiler_content(analyze_profile.stdout)
            stdout_analyze = "\nBelow are some profiling info of this kernel generated by the tool of rocprof-compute on AMD MI300 gpu, you can reference these info to analyze and generate better kernel."
            stdout_analyze += "\n1.Overview:Briefly describe the kernel type along with its runtime and dispatch statistics, such as the main kernel name, invocation count, and average execution time."
            stdout_analyze += f"\n{section_text['0']}"
            stdout_analyze += "\n2.Hardware & Resources:Key hardware details including model, architecture, number of CUs, capacities of LDS/SMEM/registers, and maximum workgroup size."
            stdout_analyze += f"\n{section_text['1']}"
            stdout_analyze += "\n3.Performance Utilization & Bottlenecks:Core bottleneck indicators such as FLOPs utilization, active CUs, occupancy, and memory bandwidth/utilization."
            stdout_analyze += f"\n{section_text['2']}"
            stdout_analyze += "\n4.Instruction Mix & Memory Access:Distribution of arithmetic, memory, and branch instructions (e.g., MFMA/FMA/VALU/VMEM), cache hit rates (L1/L2), memory bandwidth, and conflict statistics."
            stdout_analyze += f"\n{section_text['10']}"
            stdout_analyze += f"\n{section_text['16']}"
            stdout_analyze += f"\n{section_text['17']}"
            stdout_analyze += "\n5.Threading & Allocation:Wavefront/workgroup counts, allocation of VGPRs/SGPRs/LDS, thread concurrency, and resource usage per thread or workgroup."
            stdout_analyze += f"\n{section_text['7']}"
        except Exception as e:
            return None, None, str(e), None
        return profile_status, stdout_profile, stderr_profile, stdout_analyze

class TritonBenchmarkProfiler(BaseProfiler):
    def __init__(self, ):
        super().__init__()
        self.code_bash_format = "python {gen_file}"

get_profilers = {
    "tbg": TritonBenchmarkProfiler,
    "rocm": ROCmBenchmarkProfiler,
}
