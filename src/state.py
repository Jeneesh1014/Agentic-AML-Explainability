from typing import TypedDict, Annotated, Optional
import operator

class AgentState(TypedDict):
    transaction_data: dict
    is_suspicious: bool
    explanation_json: Optional[dict]
    validation_errors: Optional[str]
    revision_count: Annotated[int, operator.add]
    model_name: str  # Injected by benchmark runner to swap Teacher vs Student
