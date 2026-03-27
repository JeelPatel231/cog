from pathlib import Path
import subprocess

from pydantic import BaseModel

from core.skills import SkillRegistry
from core.tools import Tool, ToolResult


class LoadSkillInput(BaseModel):
    skill_name: str


def make_load_skill_tool(registry: SkillRegistry):
    """
    Returns a Tool that Claude can call to load the full SKILL.md of any
    registered skill. Claude may call this tool multiple times in one turn
    to load whichever skills it needs.
    """

    async def load_skill(input: LoadSkillInput) -> ToolResult:
        skill = registry.get_skill(input.skill_name)
        skill_md: Path = skill.skill_dir / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(
                f"SKILL.md not found for skill {input.skill_name!r} at {skill_md}"
            )
        return ToolResult(output=skill_md.read_text())

    return load_skill


def SkillLoaderTool(skill_registry: SkillRegistry) -> Tool[LoadSkillInput]:
    return Tool(
        name="load_skill",
        description=(
            "Load the full instructions for a named skill. "
            "Call this when you need to carry out a task covered by one of the available skills. "
            "You may call it multiple times to load several skills."
        ),
        callback=make_load_skill_tool(skill_registry),
    )

class RunSkillInput(BaseModel): 
    skill_name: str
    script_name: str
    args: list[str] = []

def SkillRunnerTool(skill_registry: SkillRegistry) -> Tool:
    async def run_skill(skill_args: RunSkillInput) -> ToolResult:

        print(f"Running skill with args: {skill_args}")

        skill_name = skill_args.skill_name
        script_name = skill_args.script_name
        args = skill_args.args

        skill = skill_registry.get_skill(skill_name)

        skill_script_dir = (skill.skill_dir / "scripts")

        if not skill_script_dir.exists() or not skill_script_dir.is_dir():
            raise FileNotFoundError(f"Scripts directory not found for skill {skill_name!r} at {skill_script_dir}")

        script_path = skill_script_dir / script_name

        if not script_path.exists() or not script_path.is_file():
            raise FileNotFoundError(f"Script {script_name!r} not found for skill {skill_name!r} at {script_path}")
        
        out = subprocess.check_output(['bash', str(script_path)] + args)  # This will raise an error if the script fails

        return ToolResult(output=f"Output of {script_name}:\n{out.decode()}")

    return Tool(
        name="run_skill",
        description=(
            "Execute a named skill that has been loaded. "
            "The input should include the 'skill_name', 'script_name', and any other parameters required by the skill."
        ),
        callback=run_skill,
    )