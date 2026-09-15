"""
benchmark.py: Runs the full AML pipeline for every transaction
in data/transactions.jsonl against two models:

  Teacher : qwen2.5:7b-instruct-q8_0  (8-bit quantised, ~7.7 GB)
  Student : qwen2.5:7b                (4-bit quantised Q4_K_M, ~4.7 GB, Ollama default)

For each transaction the script records:
  - Whether the Auditor's Pydantic schema validation passed on the first try.
  - How many self-correction loops the Explainer needed.

Results are written to data/benchmark_results.csv.
"""

import json
import csv
import sys
import os

# Allow importing graph.py from the same src/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import app


MODELS = {
    "Teacher (Q8)": "qwen2.5:7b-instruct-q8_0",  # 8-bit, ~7.7 GB
    "Student (Q4)": "qwen2.5:7b",                # Ollama default = Q4_K_M, ~4.7 GB
}

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "transactions.jsonl")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "benchmark_results.csv")
MAX_TRANSACTIONS = 20  # 20 per model = 40 graph executions total


def load_transactions(path: str, limit: int) -> list[dict]:
    transactions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tx = json.loads(line)
                if isinstance(tx, dict):
                    transactions.append(tx)
            except json.JSONDecodeError:
                continue
            if len(transactions) >= limit:
                break
    return transactions


def run_evaluation():
    transactions = load_transactions(DATA_PATH, MAX_TRANSACTIONS)
    if not transactions:
        print("ERROR: No valid transactions found in data/transactions.jsonl")
        print("Run `python src/data_generator.py` first.")
        sys.exit(1)

    print(f"Loaded {len(transactions)} transactions for evaluation.\n")

    results = []

    for label, model_tag in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"  RUNNING BENCHMARK: {label}  ({model_tag})")
        print(f"{'=' * 60}")

        success_count = 0
        total_revisions = 0

        for tx in transactions:
            tx_id = tx.get("transaction_id", "UNKNOWN")
            print(f"  Processing {tx_id} ...", end=" ", flush=True)

            initial_state = {
                "transaction_data": tx,
                "is_suspicious": False,
                "explanation_json": None,
                "validation_errors": None,
                "revision_count": 0,
                "model_name": model_tag,
            }

            try:
                final_state = app.invoke(initial_state)
            except Exception as exc:
                print(f"PIPELINE ERROR: {exc}")
                results.append({
                    "transaction_id": tx_id,
                    "model_label": label,
                    "model_tag": model_tag,
                    "revisions_needed": -1,
                    "passed_schema_audit": False,
                    "notes": str(exc)[:120],
                })
                continue

            passed = final_state.get("validation_errors") is None
            revisions = final_state.get("revision_count", 0)

            if passed:
                success_count += 1
                total_revisions += revisions
                print(f"PASS  (revisions={revisions})")
            else:
                total_revisions += revisions
                print(f"FAIL  (revisions={revisions})")

            results.append({
                "transaction_id": tx_id,
                "model_label": label,
                "model_tag": model_tag,
                "revisions_needed": revisions,
                "passed_schema_audit": passed,
                "notes": "",
            })

        compliance_rate = (success_count / len(transactions)) * 100
        avg_revisions = total_revisions / len(transactions)
        print(f"\n  [{label}] Compliance Rate : {compliance_rate:.1f}%")
        print(f"  [{label}] Avg Revisions   : {avg_revisions:.2f}")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fieldnames = ["transaction_id", "model_label", "model_tag",
                  "revisions_needed", "passed_schema_audit", "notes"]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 60}")
    print(f"  Benchmark complete. Results saved to:")
    print(f"  {OUTPUT_PATH}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run_evaluation()
