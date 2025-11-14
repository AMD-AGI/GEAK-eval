# Path to the Python file
file_path = 'test_nsa_kernel.py'

# Read the file content
with open(file_path, 'r', encoding='utf-8') as file:
    code_as_string = file.read()

# Create the dictionary
code_dict = {
    "label": code_as_string
}

# Print the dictionary
import json

with open('code_as_string.json', 'w', encoding='utf-8') as json_file:
    json.dump(code_dict, json_file, indent=2)
