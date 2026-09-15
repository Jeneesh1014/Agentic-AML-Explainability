# Agentic AML Explainability Benchmark

A local evaluation framework testing how LLM quantization impacts structured Explainable AI (XAI) compliance in a LangGraph-based Anti-Money Laundering (AML) pipeline.

## The Problem

German regulators (BaFin) demand explainability for automated banking decisions. If an AI flags a transaction, it must produce a structured audit trail. Highly quantized local models (Q4) can often correctly flag fraud. However, their ability to consistently output complex, nested JSON schemas (without syntax errors or missing required fields) can degrade under compression.

This project measures that degradation across two quantization levels of the same model family: Qwen 2.5 7B.

## Core Engineering Principles

- **Synthetic Data Only:** No real financial, customer, or patient data is used anywhere in this repository or its commit history. All records are synthetic transactions generated locally.
- **Deterministic Schema Enforcement:** The final decision on schema compliance is made by deterministic code (Pydantic), never by an LLM call. LLMs extract patterns and draft explanations; deterministic code validates and decides.
- **Zero Monetary Cost:** Runs entirely on free local open-source models via Ollama. No paid APIs, cloud credits, or external services are required.
- **Full Provenance:** Every generated explanation is grounded in the exact transaction fields (transaction ID, accounts, amount, timestamp, currency).

## Architecture

The pipeline is built with LangGraph and local Ollama inference. Three nodes execute in sequence:

```
[Transaction Data]
      │
      ▼
┌─────────────┐
│ Investigator│  LLM: flags Smurfing or Velocity Fraud
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Explainer  │  LLM: drafts BaFin-compliant JSON audit report
└──────┬──────┘
       │
       ▼
┌─────────────┐     Schema OK?   ──► END
│   Auditor   │  Pydantic ─────┤
└─────────────┘     Failed?    ──► loop back to Explainer (max 3 revisions)
```

1. **Investigator (LLM):** Ingests transaction data and flags smurfing or velocity fraud patterns.
2. **Explainer (LLM):** Drafts a structured JSON explanation justifying the verdict.
3. **Auditor (Deterministic):** A strict Pydantic validator enforcing the BaFin schema (`verdict`, `confidence_score`, `rule_violated`, `evidence_list`).

If the Explainer outputs malformed JSON or omits a required field, the Auditor rejects it. The Auditor passes the exact parser error back to the Explainer for a self-correction loop (maximum 3 revisions).

## Stack

| Layer | Technology | Purpose |
|---|---|---|
| Orchestration | `langgraph` | Stateful agent execution and conditional retry routing |
| Validation | `pydantic` | Deterministic schema and constraint enforcement |
| Local Inference | `ollama` | Zero-cost local LLM runtime |
| Teacher Model | Qwen 2.5 7B (8-bit) | `qwen2.5:7b-instruct-q8_0` (~8.1 GB) |
| Student Model | Qwen 2.5 7B (4-bit) | `qwen2.5:7b` (~4.7 GB, Ollama default Q4_K_M) |
| Data Generator | Llama 3.2 (3B) | Synthetic AML transaction generation |
| Testing | `pytest` | Automated test suite for validators and routing |

## Repository Layout

```
Agentic-AML-Explainability/
├── data/
│   ├── transactions.jsonl       # Synthetic AML dataset (Llama-3.2 generated)
│   └── benchmark_results.csv    # Benchmark evaluation output
├── src/
│   ├── data_generator.py        # Prompts Llama-3.2 to create synthetic transactions
│   ├── state.py                 # LangGraph shared state (TypedDict)
│   ├── graph.py                 # Assembles and compiles the LangGraph pipeline
│   ├── benchmark.py             # Runs Teacher vs Student evaluation
│   └── nodes/
│       ├── investigator.py      # Node 1: Fraud detection
│       ├── explainer.py         # Node 2: BaFin JSON generation
│       └── auditor.py           # Node 3: Pydantic schema enforcement
├── tests/
│   ├── test_auditor.py          # Unit tests for schema validation and edge cases
│   └── test_graph.py            # Unit tests for graph routing and retry limits
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Project dependencies
├── PROJECT_BREAKDOWN.md         # Detailed technical documentation
└── daily_log.md                 # Daily implementation log
```

## Running the Project Locally

### Prerequisites

Install and run Ollama locally with the required models:

```bash
# Pull required local models
ollama pull llama3.2
ollama pull qwen2.5:7b
ollama pull qwen2.5:7b-instruct-q8_0
```

### Environment Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Running Automated Tests

Execute the unit test suite:

```bash
pytest
```

The test suite validates:
- `BaFinComplianceSchema` acceptance of compliant payloads.
- Strict rejection of invalid verdicts and out-of-range confidence scores.
- Auditor node error handling on missing or malformed inputs.
- Deterministic graph routing and the 3-revision hard stop.

### Generating Synthetic Data and Running Pipeline

```bash
# 1. Generate synthetic dataset (50 transactions)
python src/data_generator.py

# 2. Test pipeline with one mock transaction
python src/graph.py

# 3. Run full Teacher vs Student benchmark (40 executions across 20 transactions)
python src/benchmark.py
```

Results are saved to `data/benchmark_results.csv`.

## Benchmark Results

| Model | Quantization | Size | Schema Compliance Rate | Avg Revisions Needed |
|---|---|---|---|---|
| `qwen2.5:7b-instruct-q8_0` (Teacher) | 8-bit | 8.1 GB | **100%** | 0.00 |
| `qwen2.5:7b` (Student) | 4-bit (Q4_K_M) | 4.7 GB | **100%** | 0.00 |

### Findings

At 7B parameters, Qwen 2.5 instruction following is sufficiently robust that 4-bit quantization introduces no measurable compliance degradation compared to 8-bit. Both models produced valid, Pydantic-passing BaFin JSON on the first attempt across all 20 evaluated transactions. No self-correction loops were triggered.

This result establishes an empirical floor for where quantization degradation begins. For models at this parameter scale, Q4 is sufficient for structured regulatory output tasks. Measuring degradation requires evaluating smaller model families (1B to 3B) or more aggressive quantization formats (Q2 or Q3).

## Evaluation Metrics

- **Schema Compliance Rate:** Percentage of transactions where the Explainer produced valid Pydantic output on or before the final attempt.
- **Average Revisions Needed:** Mean count of self-correction loops triggered by schema rejection per transaction.
