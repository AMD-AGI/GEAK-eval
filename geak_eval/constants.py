# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import os
import torch 

class Names:
    GEN_FOLDER = "gen"
    GEN_SUFFIX = "_gen_triton_code"
    REF_SUFFIX = "_ref_triton_code"
    RET_SEPERATOR = "*#*#"
    PYTEST_SEPARATOR = "&"*100
    GPU = torch.cuda.get_device_name(0).replace(" ", "_") if torch.cuda.is_available() else None

    PASS_NUM = 'pass_num'
    FILE_NAME = 'file_name'
    CALL_STATUS = 'call_status'
    EXEC_STATUS = 'exec_status'
    STDOUT = 'stdout'
    STDERR = 'stderr'
    DIFFICULTY = 'difficulty'

    PREDICT = 'predict'
    FILE = 'file'

    DIFFICULTY = 'difficulty'
    LABEL = 'label'

    SPEEDUP = 'speedup'

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
TMP_ROOT = "tmp2"
TBG_ROOT = os.path.join(REPO_ROOT, "data", "TritonBench")
TBG_DATA_ROOT=  os.path.join(TBG_ROOT, "data", "TritonBench_G_v1")
TBG_PERF_GOLD_ROOT = os.path.join(TBG_ROOT, "performance_metrics", "perf_G", f"{Names.GPU}_golden_metrics")
NATIVE_PERF_GOLD_ROOT = os.path.join(TBG_ROOT, "performance_metrics", "perf_G", "golden_metrics")
TBG_PERF_GOLD_DATA_ROOT = os.path.join(TBG_ROOT, "performance_metrics", "perf_G", f"{Names.GPU}_golden_results")
ROCm_ROOT = os.path.join(REPO_ROOT, "data", "ROCm")
ROCm_DATA_ROOT=  os.path.join(ROCm_ROOT, "data", "ROCm_v1")
ROCm_DATA_AUTOTUNE_ROOT=  os.path.join(ROCm_ROOT, "data", "ROCm_v1_autotune")
ROCM_PERF_GOLD_DATA_ROOT = os.path.join(ROCm_ROOT, "data", "performance", "golden_results") 