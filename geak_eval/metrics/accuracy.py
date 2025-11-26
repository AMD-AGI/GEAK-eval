# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
import numpy as np
from .base import BaseMetric

class Accuracy(BaseMetric):
    """
    Accuracy metric for evaluating model performance.
    Computes the ratio of correct predictions to total predictions.
    """

    def __init__(self):
        super().__init__("Accuracy")

    def compute(self, y_pred):
        """
        Compute the accuracy given true labels and predicted labels.

        Args:
            y_pred (np.ndarray): Predicted labels.

        Returns:
            float: Accuracy score.
        """
        return np.mean(y_pred)
