import math
from datetime import datetime

import pandas as pd
from langchain.tools import tool


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a safe mathematical expression and return the result.
    Supports basic arithmetic, exponentiation (**), and common math functions
    (sqrt, sin, cos, log, etc.).
    Example: '2 ** 10', 'sqrt(144)', '(3 + 4) * 7'
    """
    try:
        safe_builtins = {
            k: getattr(math, k) for k in dir(math) if not k.startswith("_")
        }
        safe_builtins.update({"abs": abs, "round": round, "int": int, "float": float})
        result = eval(expression, {"__builtins__": {}}, safe_builtins)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


@tool
def datetime_info(query: str = "") -> str:
    """
    Returns current date and time information.
    Use this whenever the user asks about the current date, time, day of week,
    or anything time-sensitive.
    """
    now = datetime.now()
    return (
        f"Current datetime: {now.strftime('%A, %d %B %Y – %H:%M:%S')}\n"
        f"ISO format: {now.isoformat()}\n"
        f"Week number: {now.isocalendar().week}\n"
        f"Day of year: {now.timetuple().tm_yday}"
    )


@tool
def read_file(file_path: str) -> str:
    """
    Read a plain text file and return its contents.
    Use this when the user references a local file by path.
    Supports .txt, .md, .py, .json, .yaml and similar text formats.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 8000:
            content = content[:8000] + "\n\n[... file truncated at 8000 chars ...]"
        return content
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as exc:
        return f"Error reading file: {exc}"


@tool
def query_csv(file_path: str) -> str:
    """
    Load a CSV file and return a summary: shape, column names, dtypes,
    and the first 10 rows.  Use this when the user wants to explore or
    analyze tabular data.
    """
    try:
        df = pd.read_csv(file_path)
        summary = (
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            f"Columns: {list(df.columns)}\n"
            f"Dtypes:\n{df.dtypes.to_string()}\n\n"
            f"First 10 rows:\n{df.head(10).to_string(index=False)}"
        )
        return summary
    except FileNotFoundError:
        return f"CSV not found: {file_path}"
    except Exception as exc:
        return f"Error reading CSV: {exc}"


ALL_TOOLS = [calculator, datetime_info, read_file, query_csv]
