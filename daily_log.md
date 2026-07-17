## Day 1: Project Initialization & Data Generation
**Date:** 2026-07-17
* **Objective:** Establish repo architecture and generate synthetic FinCrime data.
* **Actions Completed:**
  * Initialized Git repository and virtual environment (`langgraph`, `pydantic`, `langchain-ollama`).
  * Verified M1 Pro metal inference engine via Ollama.
  * Pulled `llama3.2`, `qwen2.5:7b`, and `qwen2.5:7b-q4`.
  * Wrote and executed `data_generator.py` to create 100 transactions with injected smurfing and velocity fraud patterns.