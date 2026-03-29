from dataclasses import dataclass
from pathlib import Path

import frontmatter

from core.tools import Tool, ToolResult
from core.tools.utils.pydantic_adapter import PydanticToolArgs


# code copied/modified from https://www.jairtrejo.com/blog/2026/01/agent-skills

@dataclass
class Skill:
    name: str
    description: str
    location: str

    @classmethod
    def from_directory(cls, directory):
        location = str((Path(directory) / 'SKILL.md').resolve())
        with open(location, 'r') as f:
            post = frontmatter.load(f)

        name = post['name']
        description = post['description']
        assert isinstance(name, str), "name param in skill metadata must be a string."
        assert isinstance(description, str), "description param in skill metadata must be a string."

        return cls(
            name=name,
            description=description,
            location=location
        )

    def instructions(self):
        with open(self.location, 'r') as f:
            post = frontmatter.load(f)
        return post.content

import os
from typing import List

def load_skills() -> List[Skill]:
    return [
        Skill.from_directory(full_path)
        for skills_root in os.environ.get('SKILLS_PATH', '').split(':')
        for directory in Path(skills_root).iterdir()
        if (full_path := Path(skills_root) / directory).is_dir()
    ]

import xml.etree.ElementTree as ET

def available_skills() -> str:
    """Returns a skill description block to be added to the system prompt"""
    skills = load_skills()
    root = ET.Element('available_skills')
    for s in skills:
        skill_el = ET.SubElement(root, 'skill')
        ET.SubElement(skill_el, 'name').text = s.name
        ET.SubElement(skill_el, 'description').text = s.description
        ET.SubElement(skill_el, 'location').text = s.location
    ET.indent(root)
    return ET.tostring(root, encoding='unicode')

class SkillToolArgs(PydanticToolArgs):
    skill_name: str

async def skill(args: SkillToolArgs) -> ToolResult:
    for skills_root in os.environ.get('SKILLS_PATH', '').split(":"):
        skill_path = Path(skills_root) / args.skill_name
        if skill_path.exists():
            return ToolResult(output=Skill.from_directory(skill_path).instructions())
    
    raise ValueError(f"Skill {args.skill_name} not found in SKILLS_PATH.")

skill_instructions = f"""
Execute a skill within the main conversation

When users ask you to perform tasks, check if any of the available skills below
can help complete the task more effectively. Skills provide specialized
capabilities and domain knowledge.

How to use skills:
- Invoke skill using this tool with the skill name only (no arguments).
- When you invoke a skill, the skill's prompt will expand and provide
  detailed instructions on how to complete the task.

Structure of a skill:
  your-skill-name/
  ├── SKILL.md          # Required: instructions + metadata
  ├── scripts/          # Optional: executable code the agent runs
  ├── references/       # Optional: docs loaded only when needed
  └── assets/           # Optional: templates, images, fonts

Important:
- Only use skills listed in <available_skills> below.
- Do not invoke a skill that is already running.

{available_skills()}
"""

SkillTool = Tool(
    name="skill",
    description=skill_instructions,
    callback=skill,
)