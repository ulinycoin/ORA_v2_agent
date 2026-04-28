from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class SubagentTask(BaseModel):
    task: str
    context: str = ""


class OrchestratorPlan(BaseModel):
    thought: str
    summary_for_user: str
    obstacles: list[str] = []
    spawn_subagents: list[SubagentTask] = []
    done: bool = False
    final_report: str | None = None


class SubagentDecision(BaseModel):
    decision_rationale: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    action: str
    args: dict[str, Any] = {}
    final_answer: str | None = None
    new_skill_name: str | None = None
    new_skill_description: str | None = None
    new_skill_code: str | None = None
    why_no_existing_skill: str | None = None


class ClassifyResult(BaseModel):
    mode: str  # "agent" | "chat"
    reason: str


class SkillWriterResult(BaseModel):
    skill_name: str
    skill_description: str
    why_this_skill_is_needed: str
    why_existing_skills_do_not_fit: str
    input_schema: dict[str, str] = {}
    output_schema: dict[str, str] = {}
    edge_cases: list[str] = []
    skill_code: str
    implementation_notes: str = ""
