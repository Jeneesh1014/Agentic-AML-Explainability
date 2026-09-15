from pydantic import BaseModel, Field, ValidationError
from typing import List
from state import AgentState

# Define the strict structural schema expected by compliance teams
class BaFinComplianceSchema(BaseModel):
    """Deterministic Pydantic schema enforcing BaFin XAI compliance standards."""
    verdict: str = Field(..., pattern="^(SUSPICIOUS|CLEAN)$")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    rule_violated: str
    evidence_list: List[str]

def audit_explanation(state: AgentState) -> dict:
    """
    Node 3: Auditor.

    Deterministically validates the Explainer's output against the BaFin schema.
    If valid, clears validation errors.
    If invalid, returns exact parser error details and increments revision count.
    """
    print("--- AUDITOR NODE ---")
    explanation = state.get("explanation_json")
    
    if not explanation:
        print("Auditor Result: REJECTED (No explanation data found)")
        return {"validation_errors": "Missing explanation JSON completely.", "revision_count": 1}
    
    try:
        # Force strict validation against the Pydantic schema
        BaFinComplianceSchema(**explanation)
        print("Auditor Result: PASSED (Schema is 100% compliant)")
        return {"validation_errors": None, "revision_count": 0}
        
    except ValidationError as e:
        error_msg = str(e)
        print(f"Auditor Result: REJECTED\nDetails:\n{error_msg}")
        # Return error and increment the graph revision count
        return {"validation_errors": error_msg, "revision_count": 1}