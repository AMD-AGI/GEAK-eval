# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
class BaseProcessor:
    """
    Base class for all processors.
    This class should be inherited by all processors.
    """

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, **kwargs):
        """
        Call the process method with the given data.
        This method can be overridden by subclasses if needed.
        """
        return self.process(*args, **kwargs)

    def process(self, *args, **kwargs):
        return NotImplementedError("Subclasses must implement the __call__ method.")

    def __str__(self):
        """
        Return a string representation of the processor.
        This method can be overridden by subclasses if needed.
        """
        return f"{self.__class__.__name__} with config: {self.config}"
