# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import datetime

def get_time():
    # Get the current time in the format YYYY-MM-DD_HH-MM-SS
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
