from pathlib import Path

from pydantic import BaseModel

from core.skills import SkillRegistry
from core.tools import Tool, ToolResult


class LoadSkillInput(BaseModel):
    skill_name: str


def make_load_skill_tool(registry: SkillRegistry) -> Tool[LoadSkillInput]:
    """
    Returns a Tool that Claude can call to load the full SKILL.md of any
    registered skill. Claude may call this tool multiple times in one turn
    to load whichever skills it needs.
    """

    async def load_skill(input: LoadSkillInput) -> ToolResult:
        skill = registry.get_skill(input.skill_name)
        skill_md: Path = skill.skill_dir / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found for skill {input.skill_name!r} at {skill_md}")
        return ToolResult(output=skill_md.read_text())

    return Tool(
        name="load_skill",
        description=(
            "Load the full instructions for a named skill. "
            "Call this when you need to carry out a task covered by one of the available skills. "
            "You may call it multiple times to load several skills."
        ),
        callback=load_skill,
    )
