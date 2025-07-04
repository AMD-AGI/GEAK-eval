# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
class BaseMetric:
    """
    Base class for all metrics.
    """

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, **kwargs):
        """
        Compute the metric.
        """
        return self.compute(*args, **kwargs)

    def compute(self, *args, **kwargs):
        """
        Compute the metric.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __str__(self):
        return f"Metric(name={self.name})"
