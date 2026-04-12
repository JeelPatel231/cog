from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, List

from pydantic import BaseModel
from core.tools import Tool, ToolResult
from core.tools.read import is_valid_absolute_path

@dataclass
class RunResult:
    stdout: str
    stderr: str
    return_code: int

class RunToolArgs(BaseModel):
    absolute_path: str
    arguments: List[str]

async def run(args: dict[str, Any] | None) -> ToolResult:
    run = RunToolArgs.model_validate(args)
    if not is_valid_absolute_path(run.absolute_path):
        raise ValueError("The path is not absolute, or the agent is not authorized to read it")
    
    path = Path(run.absolute_path)
    if path.is_dir():
        raise ValueError("The path does not point to a script")
    
    result = subprocess.run(
        [sys.executable, str(path), *run.arguments],
        capture_output=True,
        text=True,
        env={**os.environ, 'PYTHONPATH': str(path.parent)}
    )
    run_result = RunResult(stdout=result.stdout, stderr=result.stderr, return_code=result.returncode)
    return ToolResult(output=f"Return code: {run_result.return_code}\nSTDOUT:\n{run_result.stdout}\nSTDERR:\n{run_result.stderr}")

RunTool = Tool(
    name="run",
    description="Runs the python script at the given path. The path must be ABSOLUTE in format.",
    callback=run,
    args_json_schema=RunToolArgs.model_json_schema()
)
