"""
OpenAI API interface for LLMs
"""
import os
import logging
import time
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import openai

logger = logging.getLogger(__name__)

def get_time() -> str:
    """Get the current time in a formatted string"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

class OpenAILLM:
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(
        self,
    ):
        assert os.environ.get('OPENAI_API_KEY'), "API key must be provided either in config.yaml or as an environment variable 'OPENAI_API_KEY'"
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.random_seed = 42
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.api_key 
        }
        # Set up API client
        self.client = openai.OpenAI(
            api_key='dummy',
            base_url="https://llm-api.amd.com/openai/deployments/dvue-aoai-001-gpt-4.1",
            default_headers=self.headers
        )

        logger.info(f"Initialized OpenAI LLM with model 4.1")
    
    def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        response = self.client.chat.completions.create(**params)

        logger = logging.getLogger(__name__)
        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {response.choices[0].message.content}")
        return response.choices[0].message.content

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 8192,
        ## optimal temperature for coding:
        temperature: float = 0.3,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Call the LLM with the given parameters"""
        params = {
            'model': 'gpt-4.1',
            'messages': [
                            {'role': 'system', 'content': prompt["system"]},
                            {'role': 'user', 'content': prompt["user"]}
                        ],
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stop': stop or [],
            **kwargs
        }
        
        return self._call_api(params)

gpt41 = OpenAILLM()

system_prompt = """ You are an expert in Python programming and code generation. You are capable of writing efficient, clean, and well-documented code. You area of expertise includes but is not limited to:
- Python programming
- System programming
- Triton programming
- PyTorch programming
- Pytests

Your task is to write a new function named `kernel_call` that simply calls the kernel on tests cases given by the pytest parameterize decorator. You must use all the test cases provided in the parameterize decorator. The function must only call the kernel on those unit tests and it must not do anything else. The function must be named `kernel_call` and it must take no arguments. It must not return anything. It must not print anything. It must not have any side effects. It must not modify any global state. It must not use any external libraries or modules. It must not use any decorators and epsecially not the `@pytest.mark.parametrize` decorator. After this function you must add a line to call the `kernel_call` function e.g., `result = kernel_call()`. The function must be placed at the end of the file. 

If the code has a `main` function and a parser you must add boolean argument for `--kernel-call` to the parser and call the `kernel_call` function if the argument is provided. The `kernel_call` function must be called only if the `--kernel-call` argument is provided. If the `--kernel-call` argument is not provided, the `kernel_call` function must not be called.

If the code does not have a either a `main` function or a parser, you must add a `main` function that calls the `kernel_call` function and a parser that parses the `--kernel-call` argument. The `kernel_call` function must be called only if the `--kernel-call` argument is provided. If the `--kernel-call` argument is not provided, the `kernel_call` function must not be called.

You must note that `main` function must be callable only from `if __name__ == "__main__":` block. You must not add any other code to the file. You must not modify any other code in the file. You must not add any comments to the file. You must not add any docstrings to the file. You must not add any type hints to the file. You must not add any imports to the file. You must not add any global variables to the file. You must not add any classes to the file. You must not add any functions to the file. You must not add any methods to the file. You must not add any attributes to the file. You must not add any properties to the file.

You must generate the code in the following SEARCH and REPLACE diff format:

<<<<<< SEARCH
#code to be replaced from given code
======
#replacement code
>>>>>> REPLACE

e.g.

<<<<<<< SEARCH
def some_function():
    pass
======
def some_function():
    kernel_call()
>>>>>> REPLACE

"""

user_prompt = """ You are given a Triton code in Python below.

```python
{code}
```

Please provide a Python function named `kernel_call` if the SEARCH and REPLACE diff format. Only generate the desired code and DO NOT add any other text, comments, or explanations.

"""

def generate_prompt(code: str) -> Dict[str, str]:
    """Generate the prompt for the LLM"""
    return {
        "system": system_prompt,
        "user": user_prompt.format(code=code)
    }

def parse_diff(diff: str) -> Dict[str, str]:
    """Parse the diff format and return the search and replace parts"""
    search = diff.split('=====\n')[0].replace('<<<<<< SEARCH\n', '').strip()
    replace = diff.split('=====\n')[1].replace('>>>>>> REPLACE\n', '').strip()
    return {
        "search": search,
        "replace": replace
    }

def apply_diff(code: str, diff: Dict[str, str]) -> str:
    """Apply the search and replace diff to the code"""
    return code.replace(diff["search"], diff["replace"])

from glob import glob
root = "data/ROCm/data/ROCm_v1"
files = glob(f"{root}/*.py")
assert len(files) > 0, f"No files found in {root}/*.py"

print(f"Found {len(files)} files in {root}/*.py")
print("files:", files  )

target_root = "data/ROCm/data/ROCm_call"

for file in tqdm(files, desc="Processing files"):
    with open(file, 'r') as f:
        code = f.read()
    
    prompt = generate_prompt(code)
    
    try:
        response = gpt41(prompt)
        print(f"Processing {file}...")
        print(response)

        diff = parse_diff(response)
        modified_code = apply_diff(code, diff)

        # Write the response to a new file
        target_file = os.path.join(target_root, os.path.basename(file))
        with open(target_file, 'w') as f:
            f.write(modified_code)

        print(f"Modified code written to {target_file}")
    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue

# End of the file
