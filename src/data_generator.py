"""
data_generator.py

Generates synthetic AML transaction data using Llama-3.2 locally via Ollama.

Root cause of single-object outputs: Ollama's format="json" enforces a single
JSON *object* (not array). The fix is to generate ONE transaction per call and
loop to build the full dataset. This is more reliable and lets us precisely
control fraud type distribution.
"""
import json
import os
import random
from langchain_ollama import ChatOllama


# --- Configuration -------------------------------------------------------

TOTAL_TRANSACTIONS = 50   # Total to write across all batches
BATCH_SIZE = 10           # Transactions per LLM call loop
OUTPUT_PATH = "data/transactions.jsonl"

# Fraud distribution per batch of 10:  2 Smurfing, 1 Velocity, 7 Normal
FRAUD_SCHEDULE = (
    ["Smurfing"] * 2 + ["Velocity Fraud"] * 1 + ["None"] * 7
)

# -------------------------------------------------------------------------


def make_prompt(fraud_type: str, tx_index: int) -> str:
    """Build a prompt that generates ONE transaction of the requested fraud type."""
    amount_hint = ""
    fraudulent = fraud_type != "None"

    if fraud_type == "Smurfing":
        amount_hint = "The amount MUST be between 9500.00 and 9990.00 (structuring just below €10k)."
    elif fraud_type == "Velocity Fraud":
        amount_hint = "The amount can be any realistic value (1000–50000)."
    else:
        amount_hint = "The amount should be a realistic everyday transaction under 5000.00 (groceries, rent, salary, etc.)."

    return f"""Generate exactly ONE bank transaction as a JSON object.
fraud_type must be: "{fraud_type}"
is_fraudulent must be: {"true" if fraudulent else "false"}
{amount_hint}

Use this exact JSON structure:
{{
  "transaction_id": "TXN-{tx_index:04d}",
  "sender_account": "ACC-XXXX",
  "receiver_account": "ACC-XXXX",
  "amount": 0.00,
  "date": "YYYY-MM-DDTHH:MM:SSZ",
  "currency": "EUR",
  "is_fraudulent": {"true" if fraudulent else "false"},
  "fraud_type": "{fraud_type}"
}}

Replace XXXX with random 3-4 digit numbers. Replace date with a realistic 2024 date. Replace amount with a real number matching the fraud type rule above.
Output ONLY the JSON object. No explanations."""


def generate_one(llm: ChatOllama, fraud_type: str, tx_index: int) -> dict | None:
    """Call the LLM to generate a single transaction dict."""
    prompt = make_prompt(fraud_type, tx_index)
    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        required_keys = {"transaction_id", "sender_account", "receiver_account",
                         "amount", "date", "currency", "is_fraudulent", "fraud_type"}
        if not required_keys.issubset(data.keys()):
            return None
        # Override safety: enforce what we asked for
        data["fraud_type"] = fraud_type
        data["is_fraudulent"] = (fraud_type != "None")
        data["transaction_id"] = f"TXN-{tx_index:04d}"
        return data
    except (json.JSONDecodeError, AttributeError):
        return None


def generate_transactions():
    llm = ChatOllama(model="llama3.2", temperature=0.7, format="json")

    os.makedirs("data", exist_ok=True)
    num_batches = TOTAL_TRANSACTIONS // BATCH_SIZE

    tx_index = 1
    for batch_num in range(1, num_batches + 1):
        print(f"\n--- Generating Batch {batch_num}/{num_batches} ---")
        schedule = FRAUD_SCHEDULE.copy()
        random.shuffle(schedule)

        batch_rows = []
        for fraud_type in schedule:
            result = generate_one(llm, fraud_type, tx_index)
            if result:
                batch_rows.append(result)
            tx_index += 1

        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for tx in batch_rows:
                f.write(json.dumps(tx) + "\n")

        print(f"Appended {len(batch_rows)}/10 transactions to {OUTPUT_PATH}")

    print(f"\nDone. Dataset written to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_transactions()