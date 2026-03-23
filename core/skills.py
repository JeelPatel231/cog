from pathlib import Path

from pydantic import BaseModel


class Skill(BaseModel):
    name: str
    description: str
    skill_dir: Path  # directory containing SKILL.md

    model_config = {"arbitrary_types_allowed": True}


class SkillRegistry:
    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._skills: dict[str, Skill] = {s.name: s for s in (skills or [])}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def get_skill(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as error:
            raise ValueError(f"Skill not found: {name!r}. Available: {list(self._skills)}") from error
