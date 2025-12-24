# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import os
from random import randint

def get_temp_file(prefix='temp_code'):
    # Generate a unique temporary file name
    temp_file_name = f'{prefix.replace(".py", "")}_{randint(999, 999999)}.py'
    while os.path.exists(temp_file_name):
        temp_file_name = f'{temp_file_name.replace(".py", "")}_{randint(999, 999999)}.py'
    return temp_file_name

def get_rocm_temp_file(prefix='temp_code'):
    # Generate a unique temporary file name for ROCm
    temp_file_name = f'{prefix.replace(".py", "")}_{randint(999, 999999)}.py'
    while os.path.exists(temp_file_name):
        temp_file_name = f'{temp_file_name.replace(".py", "")}_{randint(999, 999999)}.py'
    return temp_file_name