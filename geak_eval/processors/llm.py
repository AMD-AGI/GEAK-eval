# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
from .base import BaseProcessor
from parse_llm_code import extract_code_blocks

class LLMOutputProcessor(BaseProcessor):
    """
    Processor for handling LLM outputs.
    This processor is designed to handle the specific structure of LLM outputs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name="LLMOutputProcessor", *args, **kwargs)

    def process(self, response: str) -> str:
        # Extract code blocks from the LLM response
        code = None
        if "```" not in response:
            return response
        code_blocks = extract_code_blocks(response)
        for _code in code_blocks.code_dict_list:
            if code is None:
                code = _code['context'] + "\n"
            else:
                code += _code['context'] + "\n"
        return code        
