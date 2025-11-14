# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.

class BasePerfEval:
    """
    Base class for performance evaluation.
    """

    def __init__(self, name="BasePerfEval"):
        """
        Initialize the BasePerfEval class.

        Args:
            name (str): Name of the evaluation instance.
        """
        self.name = name

    def __call__(self, *args, **kwargs):
        """
        Call the evaluate method to perform evaluation.
        """
        return self.evaluate(*args, **kwargs)

    def evaluate(self):
        """
        Evaluate the model on the dataset.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def report(self):
        """
        Report the evaluation results.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __str__(self):
        """
        String representation of the evaluation instance.
        """
        return f"{self.__class__.__name__}({self.name})"
