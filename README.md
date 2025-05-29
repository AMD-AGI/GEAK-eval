# Improved TritonBench evaluation framework

### Dependancy installation
- Install requirements as `pip install -r requirements.txt`

### Installation
Please install running the following command from the root folder:
- `pip install -e .`

Please note that installation does not automatically install dependancies. You must install the dependancies before installing the package.

### Running evaluation
You can run evaluations in the following two ways:
1. Command line run:
    - `tb_eval -f PATH_TO_FOLDER_OR_FILE -o NAME_OF_OUTPUT_FILE -ds tbg` for Tritonbench-G-v1
    - `tb_eval -f PATH_TO_FOLDER_OR_FILE -o NAME_OF_OUTPUT_FILE -ds rocm` for ROCm
2. From python script: the following is a bare minimum example, for a detail example please see `tb_eval/run.py`.
    - `from tb_eval.evaluators.interface import get_evaluators`
    - `evaluator = get_evaluators["tbg"]() # for TritonBenchG eval`
    - `evaluator = get_evaluators["rocm"]() # for ROCm eval`
    - `call_status, exec_status, stdout, stderr = evaluator(generated_code, log_root=PATH_TO_LOG, file_name="kernel.py", atol=1e-5, rtol=1e-2) # run evaluations`

### Issues with existing TritonBench evaluation framework
1. `1_exec_acc.py` file in TritonBench did not accurately compare the outputs of two Triton files.
1. The execution was purely done using subprocess call for both generated and ground truth files.
1. The seed consistancy is violated.
1. The outputs of the two Triton runs are compared using stdout string comparison, which is not always correct.
1. Around ground truth 150 files do not `print(result_gold)` line, hence the eval framework is essentially comapring the two null strings.
1. Some of the ground truth files (e.g. `context_attn_bloom.py`) does not even have `result_gold = test_*()` line at the end. So the call accuracy run using this file `0_call_acc.py` just blindly assumes that the call was success.
1. 7 kernel files (originally provided) run into `memory access faults`, we have fixed them.

### We have fixed these issues as follows:
1. Use `torch.allclose` to compare two runs (ground truth and generated).
1. Fix ground truth files to include `result_gold = test_*()`.
1. Ensure consistent seed across files.


We have also integrated performance measurement into the framework. Kernel evaluation flow is as follows:
1. Check if the kernel is callable: run the test function of the kernel.
2. If the kernel is callable then check if the kernel matches ground truthe by comparing outputs of the generated kernel on know tests.
3. If the generated kernel is correct: run the performance evaluation.

#### Help/support/contribute:
Please raise github issue or PR for any issues/help or contributions!

You can contribute in the following ways:
1. Add new kernels for evaluations: 
    - Add the dataset of new kernels under `tb_eval/data`.
    - Add the path of this new dataset in `tb_eval.constants`.
    - Add an evaluator interface for this new dataset in `tb_eval.evaluators.interface`.
    - Add an evaluator to be run by the interface in `tb_eval.evaluators`. The evaluator is a function that only runs python call and does not run if imported as a module. The `evaluator` (e.g. `TB_correctness.py`) is run by its `interface` (e.g. `interface.TestAllCloseEvaluatorTBG`).  
2. You can add new metrics for evaluator to work with in `tb_eval.metrics`.
3. You can add new performance eval metrics for your (or existing) dataset under `tb_eval.perf`.

#### Credits:
Our repo has found the following repos as helpful:
1. [TritonBench](https://github.com/thunlp/TritonBench/tree/main)
2. [ROCm AITER](https://github.com/ROCm/aiter)
3. [ROCm Triton](https://github.com/ROCm/triton)
