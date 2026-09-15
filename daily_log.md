# Project Log: Agentic AML Explainability Benchmark

## Day 1: Project Initialization & Data Generation
* **Objective:** Establish repo architecture and generate synthetic FinCrime data.
* **Actions Completed:**
  * Initialized Git repository and Python virtual environment (`.venv`).
  * Installed dependencies: `langgraph`, `pydantic`, `langchain-ollama`.
  * Pulled models locally via Ollama: `llama3.2`, `qwen2.5:7b`.
  * Wrote `data_generator.py` to produce 50 transactions with correct fraud distribution.
* **Blockers/Solutions:**
  * **Blocker:** Ollama's `format="json"` forces a single JSON object (the model kept outputting one dict instead of a 10-item array regardless of the prompt).
  * **Solution:** Switched to generating ONE transaction per LLM call in a loop. The single-object constraint becomes a feature: each call explicitly specifies the fraud type, giving us deterministic distribution (2 Smurfing, 1 Velocity Fraud, 7 Normal per batch of 10).

---

## Day 2-4: Investigator, Explainer, Auditor & Graph Assembly
* **Objective:** Build all three nodes and wire the LangGraph pipeline.
* **Actions Completed:**
  * `state.py`: Defined `AgentState` TypedDict including `model_name` for dynamic model switching.
  * `nodes/investigator.py`: Built from scratch: analyses transactions for Smurfing/Velocity Fraud.
  * `nodes/explainer.py`: Generates BaFin-compliant JSON audit reports; reads model from state.
  * `nodes/auditor.py`: Strict Pydantic validator (`verdict`, `confidence_score`, `rule_violated`, `evidence_list`). On failure, routes back to Explainer with parser error (max 3 loops).
  * `graph.py`: Compiled full pipeline: Investigator → Explainer → Auditor with conditional retry edge.
* **Pipeline Smoke Test Result (2026-07-17 14:34):**
  ```
  INVESTIGATOR: SUSPICIOUS ⚠️ | Amount of 9950 EUR is within smurfing range.
  EXPLAINER:    Generated structured JSON explanation.
  AUDITOR:      PASSED (Schema is 100% compliant): FIRST ATTEMPT, no retries needed.
  ```
  Final output: `verdict: SUSPICIOUS`, `confidence_score: 0.85`, `rule_violated: Smurfing`, 3-item evidence list ✅
* **Blockers/Solutions:**
  * **Blocker:** `qwen2.5:7b-q4` and `qwen2.5:7b-q8_0` tags do not exist in Ollama registry.
  * **Solution:** Correct tags are `qwen2.5:7b` (Q4_K_M default, 4.7 GB) and `qwen2.5:7b-instruct-q8_0` (8-bit, 8.1 GB). Updated all references throughout codebase.

---

## Day 5: Benchmark Runner
* **Objective:** Build `benchmark.py` and run Teacher vs Student evaluation.
* **Actions Completed:**
  * Created `src/benchmark.py`: loads 20 transactions, runs full pipeline per model, writes results CSV.
  * Pulled Teacher model `qwen2.5:7b-instruct-q8_0` (8.1 GB).
  * **Benchmark run complete (2026-07-17):**
    * Teacher (Q8): **100% compliance rate**, 0 avg revisions
    * Student (Q4): **100% compliance rate**, 0 avg revisions
  * Results saved to `data/benchmark_results.csv`.
* **Finding:** No compliance degradation observed between Q4 and Q8 at the 7B parameter scale. Qwen2.5's instruction-following is robust enough that 4-bit quantization does not affect structured JSON schema adherence. This establishes a floor: the degradation hypothesis would require testing smaller models (1B-3B) or more aggressive quantization (Q2/Q3).

---

## Day 6-7: Documentation & Outreach
* **Objective:** Finalize README, push to GitHub, send outreach email.
* **Actions Completed:**
  * `README.md`: Written with ASCII architecture diagram, benchmark methodology, full setup instructions.
* **Pending:**
  * Push final benchmark CSV to GitHub after run completes.
  * Update LinkedIn profile with this project.
  * Send email/InMail to Felix at Hawk AI.