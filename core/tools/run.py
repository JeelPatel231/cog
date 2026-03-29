import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
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
    
    process = await asyncio.create_subprocess_exec(
        str(path),
        *run.arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ} # TODO: load .env files from skills as well
    )
    stdout, stderr = await process.communicate()

    run_result = RunResult(
        stdout=stdout.decode(),
        stderr=stderr.decode(),
        return_code=process.returncode or 0
    )
    return ToolResult(output=f"Return code: {run_result.return_code}\nSTDOUT:\n{run_result.stdout}\nSTDERR:\n{run_result.stderr}")

RunTool = Tool(
    name="run",
    description="Runs the executable at the given path. The path must be ABSOLUTE in format.",
    callback=run,
    args_json_schema=RunToolArgs.model_json_schema()
)
