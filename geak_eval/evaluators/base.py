# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

class BaseEvaluator:
    """
    Base class for all Evaluators.
    """

    def __init__(self, name: str, tests_sep_line: str = "#"*146):
        self.name = name
        self.tests_sep_line = tests_sep_line

    def __call__(self, *args, **kwargs):
        """
        Call the execute method with the given arguments.
        """
        return self.execute(*args, **kwargs)

    def execute(self, *args, **kwargs):
        """
        Execute the task with the given arguments.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __str__(self):
        """
        String representation of the Evaluator.
        """
        return f"Evaluator(name={self.name})"
