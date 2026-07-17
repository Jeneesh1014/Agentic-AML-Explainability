import sys
import os
import json

# Ensure python can find our local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, END
from state import AgentState
from nodes.investigator import investigate
from nodes.explainer import explain_verdict
from nodes.auditor import audit_explanation

def route_after_auditor(state: AgentState):
    # If there are no validation errors, the explanation is BaFin compliant.
    if state.get("validation_errors") is None:
        return END
    
    # Hard stop to prevent infinite LLM hallucination loops
    if state.get("revision_count", 0) >= 3:
        print("MAX REVISIONS REACHED: Model failed to format JSON correctly after 3 attempts.")
        return END
        
    print(f"ROUTING: Validation failed. Looping back to Explainer (Attempt {state.get('revision_count')}).")
    return "explainer"

# 1. Initialize Graph
workflow = StateGraph(AgentState)

# 2. Add Nodes
workflow.add_node("investigator", investigate)
workflow.add_node("explainer", explain_verdict)
workflow.add_node("auditor", audit_explanation)

# 3. Define Edges
workflow.set_entry_point("investigator")
workflow.add_edge("investigator", "explainer")
workflow.add_edge("explainer", "auditor")
workflow.add_conditional_edges(
    "auditor",
    route_after_auditor,
    {
        END: END,
        "explainer": "explainer"
    }
)

# Compile the final application
app = workflow.compile()

if __name__ == "__main__":
    # Test the pipeline with a single mock transaction
    mock_tx = {
        "transaction_id": "TEST-999",
        "sender_account": "ACC-123",
        "receiver_account": "ACC-456",
        "amount": 9950.00,
        "date": "2026-07-17T12:00:00Z",
        "currency": "EUR"
    }
    
    initial_state = {
        "transaction_data": mock_tx,
        "is_suspicious": False,
        "explanation_json": None,
        "validation_errors": None,
        "revision_count": 0,
        "model_name": "qwen2.5:7b"
    }
    
    print("\n--- INITIATING PIPELINE TEST ---")
    final_state = app.invoke(initial_state)
    
    print("\n--- FINAL PIPELINE OUTPUT ---")
    print(json.dumps(final_state.get("explanation_json"), indent=2))