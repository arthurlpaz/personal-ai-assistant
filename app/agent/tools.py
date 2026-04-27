import pandas as pd
from langchain.tools import tool


@tool
def calculator(expression: str) -> str:
    """Safe math calculator."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception:
        return "Error in calculation"


@tool
def read_file(file_path: str) -> str:
    """Reads a local file."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception:
        return "Error reading file"


@tool
def query_csv(file_path: str) -> str:
    """Returns first rows of a CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df.head().to_string()
    except Exception:
        return "Error reading CSV"
