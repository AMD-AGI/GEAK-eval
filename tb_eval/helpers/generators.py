# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import os
from random import randint

def get_temp_file(prefix='temp_code'):
    # Generate a unique temporary file name
    temp_file_name = f'{prefix}_{randint(999, 999999)}.py'
    while os.path.exists(temp_file_name):
        temp_file_name.replace('.py', f'_{randint(999, 999999)}.py')
    return temp_file_name

def get_rocm_temp_file(prefix='temp_code'):
    # Generate a unique temporary file name for ROCm
    prefix = prefix.replace('.','_')
    temp_file_name = f'{prefix}_{randint(999, 999999)}.py'
    while os.path.exists(temp_file_name):
        temp_file_name.replace('.py', f'_{randint(999, 999999)}.py')
    return temp_file_name