# Agentic AML Explainability Benchmark

A local evaluation framework testing how LLM quantization impacts structured Explainable AI (XAI) compliance in a LangGraph-based Anti-Money Laundering (AML) pipeline.

## The Problem

German regulators (BaFin) demand explainability for automated banking decisions. If an AI flags a transaction, it must produce a structured audit trail. Highly quantized local models (Q4) can often correctly flag fraud, but their ability to consistently output complex, nested JSON schemas—without hallucinating brackets or missing required fields—degrades under compression.

This project measures that degradation across two quantization levels of the same model family (Qwen 2.5 7B).

## Architecture

The pipeline is built with LangGraph and local Ollama inference. Three nodes execute in sequence:

```
[Transaction Data]
      │
      ▼
┌─────────────┐
│ Investigator│  LLM — flags Smurfing / Velocity Fraud
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Explainer  │  LLM — drafts BaFin-compliant JSON audit report
└──────┬──────┘
       │
       ▼
┌─────────────┐     Schema OK?   ──► END
│   Auditor   │  Pydantic ─────┤
└─────────────┘     Failed?    ──► loop back to Explainer (max 3×)
```

1. **Investigator (LLM):** Ingests transaction data and flags smurfing / velocity fraud patterns.
2. **Explainer (LLM):** Drafts a structured JSON explanation justifying the verdict.
3. **Auditor (Deterministic):** A strict Pydantic validator enforcing the BaFin schema (`verdict`, `confidence_score`, `rule_violated`, `evidence_list`).

If the Explainer outputs malformed JSON or omits a required field, the Auditor rejects it, passes the exact parser error back to the Explainer for a self-correction loop (max 3 revisions).

## Stack

| Layer | Technology |
|---|---|
| Orchestration | `langgraph` |
| Validation | `pydantic` |
| Local Inference | `ollama` |
| Teacher Model | Qwen 2.5 7B — 8-bit (`qwen2.5:7b-instruct-q8_0`, ~7.7 GB) |
| Student Model | Qwen 2.5 7B — 4-bit (`qwen2.5:7b`, ~4.7 GB, Ollama default) |
| Data Generator | Llama 3.2 (3B) via Ollama |

## Repository Layout

```
Agentic-AML-Explainability/
├── data/
│   ├── transactions.jsonl       # Synthetic AML dataset (Llama-3.2 generated)
│   └── benchmark_results.csv   # Benchmark output
├── src/
│   ├── data_generator.py        # Prompts Llama-3.2 to create synthetic transactions
│   ├── state.py                 # LangGraph shared state (TypedDict)
│   ├── graph.py                 # Assembles and compiles the LangGraph pipeline
│   ├── benchmark.py             # Runs Teacher vs Student evaluation
│   └── nodes/
│       ├── investigator.py      # Node 1: Fraud detection
│       ├── explainer.py         # Node 2: BaFin JSON generation
│       └── auditor.py           # Node 3: Pydantic schema enforcement
├── requirements.txt
└── daily_log.md
```

## Running the Benchmark Locally

**Prerequisites:** Ollama must be running locally with the required models pulled.

```bash
# 1. Pull models (one-time setup)
ollama pull llama3.2
ollama pull qwen2.5:7b
ollama pull qwen2.5:7b-instruct-q8_0

# 2. Set up virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Generate synthetic dataset
python src/data_generator.py

# 4. Test the pipeline end-to-end with one mock transaction
python src/graph.py

# 5. Run the full Teacher vs Student benchmark
python src/benchmark.py
```

Results are logged to `data/benchmark_results.csv`.

## What the Benchmark Measures

- **Schema Compliance Rate:** percentage of transactions where the Explainer produced valid Pydantic output on the final attempt.
- **Revisions Needed:** how many times the self-correction loop fired before the Auditor accepted the JSON.

A meaningful drop in compliance rate or a spike in average revisions for the Q4 Student vs the Q8 Teacher quantifies the cost of model compression in a regulated AI workflow.
