from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class ProjectRecord(BaseModel):
    project_id: str
    project_name: str
    description: str
    department: Optional[str] = None
    cost: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class ClassificationResult(BaseModel):
    project_id: str
    eligible: bool
    confidence: float = Field(ge=0, le=1)
    rationale: str
    region: str = "US-IRS-Section-41"
    criteria: Optional[Dict[str, str]] = None  # None = Tier 1/2 result; synthesized downstream

class TraceStep(BaseModel):
    step_id: str
    timestamp: str
    model_name: str
    thought: str
    action: str
    observation: str
    confidence: float

class TraceEnvelope(BaseModel):
    user_id: str
    project_id: str
    steps: List[TraceStep]
    model_name: str
    region: str
    reviewer_id: Optional[str] = None
    legal_hold_flag: bool = False
    checksum_sha256: Optional[str] = None

class PipelineProjectInput(BaseModel):
    name: str
    description: str
    tech_domain: str
    fiscal_year: int

class PipelineExpenseRow(BaseModel):
    description: str
    amount: float
    type: str    # "wage", "contractor", "supplies", "cloud", etc.
    employee_role: Optional[str] = None

class PipelineRequest(BaseModel):
    project: PipelineProjectInput
    expenses: List[PipelineExpenseRow]

