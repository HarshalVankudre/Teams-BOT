"""
Structured planning models for expert project recommendations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProjectSpec:
    """Normalized project specification inferred from the conversation."""

    project_type: str = ""
    construction_method: str = ""
    length_m: Optional[float] = None
    width_m: Optional[float] = None
    load_profile: str = ""
    ground_condition: str = ""
    space_constraints: str = ""
    drainage_requirements: str = ""
    delivery_preference: str = ""
    special_constraints: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)


@dataclass
class PlanningConstraints:
    """Planner policy defaults used for final machine allocation."""

    db_only: bool = True
    require_released_only: bool = True
    availability_first: bool = True
    allow_external_alternatives: bool = False
    language: str = "de"


@dataclass
class ProjectPhase:
    """Execution phase in the final project dossier."""

    name: str
    objective: str
    categories_needed: List[str] = field(default_factory=list)
    quality_check: str = ""
    dependency: str = ""


@dataclass
class MachineAllocation:
    """Resolved machine allocation for one required capability."""

    requested_category: str
    resolved_category: str
    phase_name: str
    purpose: str
    primary_machine: Dict[str, Any] = field(default_factory=dict)
    backup_machine: Dict[str, Any] = field(default_factory=dict)
    selection_reason: str = ""
    constraint_check: str = ""
    fallback_plan: str = ""
    is_alternative: bool = False


@dataclass
class ProjectPlan:
    """Complete structured project plan assembled by the expert planner."""

    query_text: str
    summary: str
    spec: ProjectSpec
    constraints: PlanningConstraints
    phases: List[ProjectPhase] = field(default_factory=list)
    allocations: List[MachineAllocation] = field(default_factory=list)
    unresolved_gaps: List[str] = field(default_factory=list)
    verification_notes: str = ""
    next_actions: List[str] = field(default_factory=list)
    decision_log: List[str] = field(default_factory=list)
    full_recommendation: str = ""  # Full AI expert recommendation text

    def to_dict(self) -> Dict[str, Any]:
        """Return plan as a plain serializable dictionary."""
        return asdict(self)
