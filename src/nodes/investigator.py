import json
import sys
import os

# Ensure local modules are importable when this node is called from graph.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama
from state import AgentState


def investigate(state: AgentState) -> dict:
    """
    Node 1: Investigator.

    Analyses raw transaction data and decides whether it is suspicious.
    Uses the model specified in state["model_name"] so the benchmark runner
    can swap the Teacher (qwen2.5:7b-instruct-q8_0) against the Student (qwen2.5:7b)
    without modifying source files.
    """
    print("--- INVESTIGATOR NODE ---")
    tx = state["transaction_data"]
    model = state["model_name"]

    llm = ChatOllama(model=model, temperature=0.0, format="json")

    prompt = f"""
    You are an Anti-Money Laundering (AML) specialist.
    Analyse the following bank transaction and decide whether it is suspicious.

    Transaction:
    {json.dumps(tx, indent=2)}

    Suspicious patterns to look for:
    - Smurfing: amount between 9500 and 9990 EUR (structuring just below the €10k reporting threshold).
    - Velocity Fraud: is_fraudulent flag is true with fraud_type "Velocity Fraud".
    - Any transaction marked is_fraudulent = true.

    Respond ONLY with a valid JSON object in this exact format:
    {{
        "is_suspicious": true or false,
        "reason": "One sentence explanation"
    }}
    """

    response = llm.invoke(prompt)

    try:
        result = json.loads(response.content.strip())
        is_suspicious = bool(result.get("is_suspicious", False))
        reason = result.get("reason", "No reason provided.")
    except (json.JSONDecodeError, ValueError):
        # Default safe fallback: treat as suspicious if parsing fails
        is_suspicious = True
        reason = "Parse error: defaulting to suspicious."

    flag = "SUSPICIOUS ⚠️" if is_suspicious else "CLEAN ✅"
    print(f"Investigator Result: {flag} | {reason}")

    return {"is_suspicious": is_suspicious}
