# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import numpy as np
from .base import BaseMetric

class PassK(BaseMetric):
    """
    Pass@k metric.
    This metric checks if the correct answer is among the top k predictions.
    """

    def __init__(self,):
        super().__init__(name="Pass@k")

    def compute(self, n: int, c: int, k: int) -> float:
        """
        Compute the Pass@k metric.

        Args:
            n (int): The number of parallel instances.
            c (list): The number of correct instances.
            k (int): The number of instances to consider.

        Returns:
            float: The Pass@k score.
        """
        if n -c < k: return 1.0
        return 1 - np.prod(
            1 - k/ np.arange(
                n-c+1, n+1
            )
        )
