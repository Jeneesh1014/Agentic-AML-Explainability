import json
from langchain_ollama import ChatOllama
from state import AgentState

def explain_verdict(state: AgentState) -> dict:
    """
    Node 2: Explainer.

    Generates a structured JSON explanation complying with BaFin standards.
    Reads the target model from state to support dynamic model evaluation.
    Incorporates prior auditor validation errors to enable self-correction.
    """
    print("--- EXPLAINER NODE ---")
    tx = state["transaction_data"]
    is_suspicious = state["is_suspicious"]
    prev_errors = state.get("validation_errors")
    
    # Model is injected dynamically by the benchmark runner via state["model_name"]
    model = state["model_name"]
    llm = ChatOllama(model=model, temperature=0.1, format="json")
    
    # If the investigator flagged it clean, provide a standard empty compliance log
    if not is_suspicious:
        clean_json = {
            "verdict": "CLEAN",
            "confidence_score": 1.0,
            "rule_violated": "None",
            "evidence_list": []
        }
        return {"explanation_json": clean_json, "validation_errors": None}
    
    # Construct prompt with self-correction history if errors exist
    feedback_loop = ""
    if prev_errors:
        feedback_loop = f"\nCRITICAL: Your previous response was REJECTED by the Auditor for violating the schema constraints.\nParser Error: {prev_errors}\nFix your structural output accordingly."

    prompt = f"""
    You are a specialized Explainable AI (XAI) Compliance Officer working under BaFin regulations.
    Your task is to provide a structured, audit-ready explanation for a flagged suspicious transaction.
    
    Transaction context:
    {json.dumps(tx, indent=2)}
    {feedback_loop}
    
    You MUST output strictly a JSON object matching this schema blueprint:
    {{
        "verdict": "Must be 'SUSPICIOUS'",
        "confidence_score": "Float value between 0.0 and 1.0",
        "rule_violated": "Name of pattern detected (e.g., 'Smurfing' or 'Velocity Fraud')",
        "evidence_list": ["List of empirical strings showing why this conclusion was reached"]
    }}
    """
    
    response = llm.invoke(prompt)
    
    try:
        explanation_data = json.loads(response.content.strip())
        return {"explanation_json": explanation_data}
    except json.JSONDecodeError:
        # If raw parsing fails, pass back an empty object to let the Auditor catch it
        return {"explanation_json": {}, "validation_errors": "Raw JSON string parsing failed"}