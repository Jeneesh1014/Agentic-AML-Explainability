# PROJECT_BREAKDOWN.md
# Complete Technical Deep-Dive: Agentic AML Explainability Benchmark

> **Author:** AI Senior Engineer Analysis
> **Scope:** Every file, folder, class, function, line, decision, and design pattern — explained from zero.
> **Goal:** After reading this, you can explain, modify, debug, and present every piece of this project with confidence.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Real-World Problem It Solves](#2-real-world-problem-it-solves)
3. [Technology Stack — Every Library Explained](#3-technology-stack--every-library-explained)
4. [Overall Architecture](#4-overall-architecture)
5. [Full Project Execution Flow](#5-full-project-execution-flow)
6. [Project Structure](#6-project-structure)
7. [Folder: Root Level](#7-folder-root-level)
8. [File: `.gitignore`](#8-file-gitignore)
9. [File: `requirements.txt`](#9-file-requirementstxt)
10. [File: `daily_log.md`](#10-file-daily_logmd)
11. [File: `README.md`](#11-file-readmemd)
12. [Folder: `data/`](#12-folder-data)
13. [File: `data/transactions.jsonl`](#13-file-datatransactionsjsonl)
14. [File: `data/benchmark_results.csv`](#14-file-databenchmark_resultscsv)
15. [Folder: `src/`](#15-folder-src)
16. [File: `src/__init__.py`](#16-file-src__init__py)
17. [File: `src/state.py`](#17-file-srcstatepy)
18. [File: `src/graph.py`](#18-file-srcgraphpy)
19. [File: `src/data_generator.py`](#19-file-srcdata_generatorpy)
20. [File: `src/benchmark.py`](#20-file-srcbenchmarkpy)
21. [Folder: `src/nodes/`](#21-folder-srcnodes)
22. [File: `src/nodes/__init__.py`](#22-file-srcnodes__init__py)
23. [File: `src/nodes/investigator.py`](#23-file-srcnodesinvestigatorpy)
24. [File: `src/nodes/explainer.py`](#24-file-srcnodesexplainerpy)
25. [File: `src/nodes/auditor.py`](#25-file-srcnodesauditorpy)
26. [Data Flow: End to End](#26-data-flow-end-to-end)
27. [Design Patterns Used](#27-design-patterns-used)
28. [Hidden Assumptions, Edge Cases and Potential Bugs](#28-hidden-assumptions-edge-cases-and-potential-bugs)
29. [Benchmark Results Explained](#29-benchmark-results-explained)
30. [How to Present This Project Professionally](#30-how-to-present-this-project-professionally)

---

## 1. What Is This Project?

### Plain-English Summary

This project is a **research benchmark** that answers one very specific engineering question:

> "If you compress an AI model to use less memory, does it get worse at producing perfectly structured JSON reports for banking regulators?"

To answer this, the project builds a complete **automated Anti-Money Laundering (AML) pipeline** using AI agents. The pipeline reads a bank transaction, decides if it looks like financial crime, writes a formal explanation as a JSON document, then checks if that JSON document follows the strict legal schema required by German banking regulators (BaFin).

It then runs this pipeline twice — once using a high-quality 8-bit model, once using a smaller 4-bit model — and compares the results.

### Why Would Someone Build This?

Banks and financial institutions increasingly use AI to flag suspicious transactions. Regulators (especially in the EU) now require that when AI makes a decision about money, it must provide a human-readable, machine-verifiable **audit trail** — this is called **Explainable AI (XAI) compliance**.

The research question is: *Can you use a cheaper, more compressed model and still meet regulatory explainability standards?*

This project answers that question empirically.

---

### Interview Explanation

**30-second version:**
"I built a research benchmark that tests whether quantized LLMs — compressed AI models — can still produce valid, legally compliant JSON audit reports for flagged bank transactions. It uses a LangGraph pipeline with three agents: one flags fraud, one writes the report, one validates it against BaFin's schema. The benchmark shows 4-bit and 8-bit quantization of Qwen 2.5 7B perform identically for this task."

**2-minute version:**
"The project addresses a real regulatory problem: BaFin, Germany's financial regulator, requires AI systems that flag suspicious transactions to produce structured, auditable explanations. The question I explored is whether model quantization — reducing a model from 8-bit to 4-bit precision to save memory and compute — degrades the model's ability to output valid structured JSON. I built a LangGraph agent pipeline where three nodes execute in sequence: an Investigator LLM that reads the transaction and flags fraud patterns like Smurfing or Velocity Fraud; an Explainer LLM that writes a JSON audit report; and a deterministic Pydantic Auditor that validates the schema. If validation fails, the system loops back to the Explainer with the error message for self-correction, up to 3 times. I generated 50 synthetic transactions, ran 20 through each of two model variants — Q8 and Q4 — and measured schema compliance rates and average retries needed. Both models achieved 100% compliance with zero retries, establishing that at 7B parameters, Q4 quantization introduces no measurable compliance degradation."

**Common follow-up questions and strong answers:**

| Question | Strong Answer |
|---|---|
| Why LangGraph instead of a simple function chain? | LangGraph gives us a real-time state machine with conditional routing — the retry loop is expressed as a conditional edge, not imperative control flow. This is more maintainable and more aligned with how production agent systems are built. |
| What is quantization? | Quantization reduces the numerical precision of model weights. An 8-bit model stores each weight as a number between 0-255; a 4-bit model uses 0-15. This shrinks model size roughly by half and reduces RAM use, but can introduce precision loss. |
| What is BaFin? | BaFin (Bundesanstalt fur Finanzdienstleistungsaufsicht) is Germany's Federal Financial Supervisory Authority — the regulatory body that oversees banks, insurance companies, and financial services firms. |
| Why Pydantic for validation? | Pydantic provides declarative, strongly typed schema enforcement with rich error messages. This is exactly what a compliance system needs — it's not just a check but a source of specific, actionable error feedback for the self-correction loop. |
| What's Smurfing? | Smurfing is a money laundering technique where large sums are broken into smaller transactions just below reporting thresholds (in Germany, 10,000 EUR) to avoid triggering mandatory reports. |
| What would you change? | Test smaller model sizes (1B, 3B) and more aggressive quantization (Q2, Q3) to find the actual failure threshold. Add temperature sensitivity analysis. Use a real validated transaction dataset. |

---

### Why We Built It

**What problem existed before this code?**
No empirical evidence existed (at the personal project scale) about whether cheap quantized models could satisfy financial regulatory compliance schemas.

**Why wasn't the previous approach good enough?**
The naive approach would be to just use the best possible model — but that has real cost. A 7B Q8 model needs approximately 8 GB of GPU RAM. A Q4 model needs approximately 5 GB. At scale, across thousands of transactions per second, that difference is enormous. Without benchmarking, you can't make a justified engineering decision.

**What benefit does this provide?**
The benchmark provides a reproducible, evidence-based answer: at 7B parameters, Q4 is sufficient. This means real production systems could use the smaller model without compliance risk.

**What trade-offs were made?**
The test dataset is synthetic, not real financial data. The fraud patterns are simple (Smurfing, Velocity Fraud). Real AML is more complex. The benchmark should be treated as a directional result, not a production certification.

### Beginner Version

Imagine a bank uses a robot to check if someone is laundering money. When the robot says "suspicious!", it has to write a report — like filling in a form — and the form has very strict rules (e.g., you MUST write "SUSPICIOUS" or "CLEAN", nothing else). This project tests whether a cheaper robot (less memory, smaller) can still fill in the form correctly. Turns out: yes, the cheaper robot does just as well as the expensive one.

### Professional Version

This is an empirical benchmark of quantization-induced degradation in structured output compliance for LLM-based regulatory AI pipelines. The system implements a LangGraph DAG with a Pydantic-gated self-correction loop, testing whether W4A16 quantization (Q4_K_M) vs. W8A16 (Q8_0) of the Qwen2.5-7B-Instruct model family produces statistically different BaFin XAI schema adherence rates. The null result — 100% compliance, 0 average revisions, for both quantization levels across 20 synthetic FATF-aligned AML test cases — establishes 7B/Q4 as the upper-floor for structured output reliability at this parameter scale.

---

## 2. Real-World Problem It Solves

### Who Are the Users?

- ML engineers at banks who need to choose which model variant to deploy
- Compliance officers who need to prove that AI-generated audit trails meet regulatory standards
- Researchers studying the intersection of model compression and regulated AI

### What Real-World Regulation Is Involved?

The EU AI Act (2024) and German BaFin circular requirements mandate that:
1. Automated financial decisions must be explainable.
2. Explanations must follow a structured, auditable format.
3. Audit trails must be machine-verifiable.

This project directly operationalizes those requirements into a working system.

---

## 3. Technology Stack — Every Library Explained

### `langgraph`

**What it is:** A Python library from LangChain for building stateful, multi-step AI agent workflows as directed graphs.

**Why it exists here:** The pipeline has three sequential steps with a conditional loop back. LangGraph models this naturally as a graph — each step is a node, connections between steps are edges, and the loop-back condition is a "conditional edge."

**What it does:** It manages the execution order of the nodes, passes state between them, and handles the routing logic.

**Where it's used:** `graph.py` builds the graph; `state.py` defines the state structure it manages.

**Alternative approaches:**
- Plain Python functions: You could chain investigate(), explain(), audit() directly. But adding the retry loop would require while loops and manual state passing — messy and error-prone.
- LangChain LCEL: LangChain's newer "chain" syntax — but LCEL doesn't natively handle stateful conditional loops.
- Prefect / Airflow: Overkill for a local research pipeline.

**Why LangGraph was chosen:** It's the industry-standard tool for exactly this pattern — stateful multi-agent workflows with conditional routing. Knowing LangGraph is directly employable.

---

### `pydantic`

**What it is:** A Python library for data validation using Python type annotations.

**Why it exists here:** The Auditor needs to check that the LLM-generated JSON exactly matches a required schema. Pydantic makes this trivially easy and gives descriptive error messages when something is wrong.

**What it does:** Defines the BaFinComplianceSchema class. When you call `BaFinComplianceSchema(**data)`, Pydantic checks every field against its type, constraints, and patterns. If anything is wrong, it raises a ValidationError with a detailed message.

**Where it's used:** `nodes/auditor.py` — exclusively for schema enforcement.

**Alternatives:**
- `jsonschema` library: More verbose, less Pythonic.
- Manual `if` checks: Fragile, hard to maintain.
- Marshmallow: Similar but less integrated with Python types.

**Why Pydantic:** It's the Python standard for this. FastAPI uses it. LangChain uses it. It's the right tool.

---

### `langchain-ollama`

**What it is:** A LangChain integration that lets you talk to locally running Ollama models using LangChain's standard ChatOllama interface.

**Why it exists here:** The project runs models locally using Ollama (not cloud APIs like OpenAI). This library bridges LangChain's standard interface to Ollama's local server.

**What it does:** `ChatOllama(model="qwen2.5:7b")` creates an object that, when you call `.invoke(prompt)`, sends that prompt to your locally running Ollama server and returns the response.

**Where it's used:** `investigator.py`, `explainer.py`, `data_generator.py` — everywhere an LLM is called.

**Alternatives:**
- `ollama` Python library directly: Exists, but doesn't integrate with LangChain's chain/graph ecosystem.
- OpenAI API: Would cost money, require internet, and eliminate the local/private nature of the pipeline.
- HuggingFace Transformers: Could run models locally but requires much more setup and GPU configuration.

**Why this choice:** Privacy (no data leaves the machine), zero cost, and it integrates cleanly with LangGraph.

---

### `Ollama` (external tool, not a pip package)

**What it is:** A desktop application and server that runs large language models locally on your machine — like a local version of the OpenAI API.

**Why it exists here:** The project needs to run LLMs without cloud access or API keys. Ollama handles all the complexity of model loading, GPU memory management, and serving.

**What it does:** You pull models with `ollama pull qwen2.5:7b`, and Ollama stores them locally. Then it runs an HTTP server at `localhost:11434` that accepts prompts and returns completions. `langchain-ollama` talks to this server.

**Alternatives:**
- vLLM: Production-grade local server — overkill for research.
- llama.cpp directly: Low-level, requires compilation.

---

## 4. Overall Architecture

```
+----------------------------------------------------------+
|                    BENCHMARK RUNNER                      |
|                   (benchmark.py)                         |
|  Loops over: Teacher (Q8) and Student (Q4) models        |
|  For each model, loops over 20 transactions              |
+----------------------+-----------------------------------+
                       |
                       v invokes with initial state
+----------------------------------------------------------+
|              LANGGRAPH PIPELINE (graph.py)               |
|                                                          |
|  +--------------+   +-------------+   +-------------+   |
|  | INVESTIGATOR |-->|  EXPLAINER  |-->|   AUDITOR   |   |
|  |  (LLM)       |   |  (LLM)      |   | (Pydantic)  |   |
|  +--------------+   +-------------+   +------+------+   |
|                            ^                 |           |
|                            |  <--------------+           |
|                            |  (on failure, max 3x)       |
+----------------------------------------------------------+
                       |
                       v
+----------------------------------------------------------+
|                   SHARED STATE                           |
|               (AgentState TypedDict)                     |
|  transaction_data | is_suspicious | explanation_json     |
|  validation_errors | revision_count | model_name         |
+----------------------------------------------------------+
```

### What Is a "Node" in This Context?

A node is a Python function that:
1. Receives the current shared state (a Python dictionary)
2. Does some work (calls an LLM, validates data, etc.)
3. Returns a partial dictionary of updates to merge back into the state

Nodes never talk to each other directly. They only read and write shared state. This is the **Actor Model / Blackboard Pattern** — a core design principle.

---

## 5. Full Project Execution Flow

### Step-by-Step: What Happens When You Run `benchmark.py`

```
User runs: python src/benchmark.py
      |
      v
benchmark.py: reads data/transactions.jsonl
      |
      v
FOR EACH MODEL (Teacher Q8, then Student Q4):
  FOR EACH TRANSACTION (20 transactions):
    |
    v
    Constructs initial_state dict with:
      - transaction_data = the transaction
      - model_name = current model tag
      - is_suspicious = False (default)
      - explanation_json = None
      - validation_errors = None
      - revision_count = 0
    |
    v
    app.invoke(initial_state)
    |
    v
    +---------- LANGGRAPH EXECUTION --------------+
    |                                              |
    |  INVESTIGATOR NODE                           |
    |    Reads transaction_data from state         |
    |    Reads model_name from state               |
    |    Creates ChatOllama(model=model_name)      |
    |    Sends prompt: "Is this suspicious?"       |
    |    Parses JSON: {is_suspicious, reason}      |
    |    Writes {is_suspicious} to state           |
    |                                              |
    |  EXPLAINER NODE                              |
    |    Reads transaction_data, is_suspicious     |
    |    If NOT suspicious: returns CLEAN JSON     |
    |    If suspicious + prev errors: adds error   |
    |    Sends prompt: "Write BaFin audit JSON"    |
    |    Writes {explanation_json} to state        |
    |                                              |
    |  AUDITOR NODE                                |
    |    Reads explanation_json from state         |
    |    BaFinComplianceSchema(**explanation_json) |
    |    PASS: validation_errors = None            |
    |    FAIL: validation_errors = error msg       |
    |          revision_count += 1                 |
    |                                              |
    |  ROUTE AFTER AUDITOR                         |
    |    No errors? -> END                         |
    |    Errors + count < 3? -> EXPLAINER again    |
    |    Errors + count >= 3? -> END (hard stop)   |
    +----------------------------------------------+
    |
    v
    benchmark.py checks final_state:
      passed = (validation_errors is None)
      revisions = revision_count
    |
    v
    Appends row to results list
|
v
Writes data/benchmark_results.csv
```

---

## 6. Project Structure

```
Agentic-AML-Explainability/        <- Root: the entire project
|
+-- .git/                          <- Git version control (internal, never edit)
+-- .venv/                         <- Python virtual environment (not committed)
+-- .gitignore                     <- Tells Git what to ignore
+-- README.md                      <- Project documentation for GitHub
+-- daily_log.md                   <- Developer's build journal
+-- requirements.txt               <- Python dependencies list
+-- PROJECT_BREAKDOWN.md           <- This file
|
+-- data/                          <- All data files (inputs + outputs)
|   +-- transactions.jsonl         <- Synthetic AML transaction dataset
|   +-- benchmark_results.csv      <- Output of benchmark run
|
+-- src/                           <- All source code
    +-- __init__.py                <- Makes src/ a Python package
    +-- state.py                   <- Shared LangGraph state definition
    +-- graph.py                   <- Pipeline assembly and wiring
    +-- data_generator.py          <- Script that created transactions.jsonl
    +-- benchmark.py               <- Main evaluation script
    |
    +-- nodes/                     <- Individual pipeline step implementations
        +-- __init__.py            <- Makes nodes/ a Python package
        +-- investigator.py        <- Node 1: Fraud detection
        +-- explainer.py           <- Node 2: Audit report generation
        +-- auditor.py             <- Node 3: Schema validation
```

---

## 7. Folder: Root Level

### Why This Folder Exists
This is the Git repository root. Everything lives here. Python projects conventionally place the root package inside `src/`, configuration files at the root, and data in a `data/` directory. This layout is called **src-layout** and prevents accidental imports of un-installed code.

### Responsibilities
- House configuration files (`.gitignore`, `requirements.txt`)
- Host documentation (`README.md`, `daily_log.md`)
- Serve as the entry point for all commands

---

## 8. File: `.gitignore`

### What Is It?
A configuration file that tells Git which files and folders to **never track or commit**.

### Why Does It Exist?
Without it, running `git add .` would accidentally commit:
- `.venv/` — the entire Python virtual environment (hundreds of MB of packages that can be reinstalled from `requirements.txt`)
- `__pycache__/` — Python's compiled bytecode (auto-generated, machine-specific, wasteful to store)
- `.DS_Store` — macOS folder metadata files (irrelevant to other developers)
- `.env` — environment variable files that may contain secrets

### Line-by-Line Explanation

```
# Python
__pycache__/      <- Python's auto-generated bytecode cache folders
*.py[cod]         <- Compiled Python files (.pyc, .pyo, .pyd) all variants
*.pyo             <- Old-style optimized Python bytecode
*.pyd             <- Python extension modules (C-compiled on Windows)
.Python           <- Python interpreter symlink (virtual env artifact)
*.so              <- Shared object files (C extensions on Linux/Mac)

# Virtual environments
.venv/            <- The project's virtual environment folder (pip packages)
venv/             <- Alternative venv name
env/              <- Another alternative venv name

# Distribution
dist/             <- Built package distributions
build/            <- Build artifacts
*.egg-info/       <- Package metadata generated by setuptools

# IDE
.vscode/          <- VS Code editor settings (personal preference files)
.idea/            <- PyCharm settings
*.swp, *.swo      <- Vim temporary swap files

# OS
.DS_Store         <- macOS Finder metadata (useless to other devs)
Thumbs.db         <- Windows thumbnail cache

# Jupyter
.ipynb_checkpoints/ <- Jupyter notebook auto-save files

# Environment files
.env              <- API keys, secrets -- NEVER commit these
.env.local        <- Local overrides of environment variables
```

### What Would Happen If We Removed It?
The next `git add .` would commit gigabytes of virtual environment files and cache. The repository would become bloated and may expose sensitive environment variables.

### Design Pattern
This follows the **separation of configuration from code** principle. Secrets and machine-specific files never go into version control.

### Interview Explanation

**30-second:** "The `.gitignore` prevents generated files, secrets, and machine-specific artifacts from being committed to version control. The most important entries are `.venv/` (the package environment, which is reproducible from `requirements.txt`) and `.env` (which might contain API keys)."

**Common follow-up:** "Why not just add files manually when needed?" — Because `git add .` or bulk-staging commands would catch them accidentally. Defense in depth is better than relying on developer discipline.

---

## 9. File: `requirements.txt`

### What Is It?
A plain text file listing all Python packages that need to be installed for this project to run.

### Why Does It Exist?
When someone else clones the repository (or you set up on a new machine), they run `pip install -r requirements.txt` to get exactly the packages needed. It's the contract between the code and the environment.

### Contents

```
langgraph        <- The agent orchestration framework
pydantic         <- The data validation library
langchain-ollama <- The Ollama LLM interface
```

### Installed Packages Breakdown

| Package | What Gets Installed | What It Enables |
|---|---|---|
| `langgraph` | LangGraph + LangChain core | StateGraph, END, conditional edges |
| `pydantic` | Pydantic v2 | BaseModel, Field, ValidationError |
| `langchain-ollama` | LangChain's Ollama adapter | ChatOllama class |

### What Would Happen If We Removed It?
The project becomes un-reproducible. Someone would need to guess which packages to install.

### Hidden Assumption (Potential Bug)
There are **no pinned versions** (e.g., `langgraph==0.2.14`). This means `pip install -r requirements.txt` always installs the **latest** version. LangGraph in particular has breaking API changes between minor versions. This is a potential bug — future installs might break.

**Best practice fix:**
```
langgraph==0.2.14
pydantic==2.7.1
langchain-ollama==0.1.3
```

### Interview Explanation

**30-second:** "`requirements.txt` is the Python dependency manifest. When someone clones this repo, `pip install -r requirements.txt` recreates the environment. A key technical debt is the lack of version pinning — a future install could get a breaking version of LangGraph. The production fix is to use `pip freeze > requirements.txt` to pin exact versions, or switch to `poetry` or `pyproject.toml`."

---

## 10. File: `daily_log.md`

### What Is It?
A developer's build journal — a day-by-day record of what was built, what problems were encountered, and how they were solved.

### Why Does It Exist?
Two purposes:
1. **Personal reference:** When you return to the code months later, this tells you why specific decisions were made.
2. **Portfolio value:** This document is unusually honest about problems encountered. For job applications, this shows genuine engineering process — not just polished results.

### Key Engineering Decisions Documented

**The Ollama JSON Array Bug (Day 1):**
> "Ollama's `format='json'` forces a single JSON object — the model kept outputting one dict instead of a 10-item array."

This is a real Ollama limitation. When you set `format="json"`, Ollama's grammar-constrained decoding forces the output to be a single `{}` object, never a `[]` array. The fix was to generate one transaction per API call in a loop, using Ollama's single-object constraint as a feature rather than fighting it.

**The Model Tag Discovery (Day 2-4):**
> "`qwen2.5:7b-q4` and `qwen2.5:7b-q8_0` tags do not exist in Ollama registry."

The correct Ollama model tags were `qwen2.5:7b` (which happens to be Q4_K_M) and `qwen2.5:7b-instruct-q8_0`. This is non-obvious and caused wasted time — the log preserves this hard-won knowledge.

**The Surprising Finding (Day 5):**
Both models achieved 100% compliance, zero retries. The self-correction loop was never needed. This is a valid scientific negative result — the hypothesis (that Q4 degrades compliance) was not supported at this scale.

### Beginner Version

This is like a developer's diary. Every day, the developer wrote down what they tried to build, what broke, and how they fixed it. It's valuable because it shows the real messy process of software development, not just the final polished result.

### Professional Version

The `daily_log.md` serves as an Architecture Decision Record (ADR) and an engineering runbook. It documents discovered API constraints (Ollama's JSON mode limitation), corrects non-obvious model naming conventions in the Ollama registry, and records empirical findings that informed architectural choices. This is a standard practice in senior engineering — capturing the "why" alongside the "what".

---

## 11. File: `README.md`

### What Is It?
The public-facing documentation for the project. GitHub automatically renders this on the repository homepage.

### Why Does It Exist?
It's the first thing any visitor, recruiter, or collaborator reads. It must answer: what is this? How do I run it? What did it find?

### Structure Breakdown

| Section | Purpose |
|---|---|
| Title + one-liner | Immediately tells the reader what this is |
| The Problem | Real-world context — why this matters |
| Architecture diagram | Visual overview of the three nodes |
| Stack table | Technology choices at a glance |
| Repository Layout | File map so readers can navigate |
| Running Locally | Reproducible setup instructions |
| Benchmark Results | The actual finding — the payoff |
| What the Benchmark Measures | Methodology transparency |

### Why This Structure Was Chosen
This follows the **Inverted Pyramid** style — most important information first. A recruiter scanning for 10 seconds gets: what it is, what it found. A developer who wants to run it gets: exact setup commands. A researcher who wants methodology gets: the measurement definitions.

---

## 12. Folder: `data/`

### Why This Folder Exists
Separating data from code is a standard project layout principle. Code changes often; data does not. Keeping them separate:
- Makes it easy to .gitignore large data files
- Makes the data purpose obvious
- Prevents accidental data corruption when modifying code

### Responsibilities
- Stores the **input dataset** (`transactions.jsonl`)
- Stores the **benchmark output** (`benchmark_results.csv`)
- Is the only folder that `data_generator.py` and `benchmark.py` write to

### How the Project Flows Through This Folder

```
data_generator.py -> WRITES -> data/transactions.jsonl
                                        |
                                        v
                              benchmark.py -> READS
                                        |
                                        v
                              benchmark.py -> WRITES -> data/benchmark_results.csv
```

---

## 13. File: `data/transactions.jsonl`

### What Is It?
The synthetic dataset of bank transactions. JSONL (JSON Lines) format means **one JSON object per line**.

### Why JSONL Instead of a JSON Array?
- JSONL can be streamed line by line — you don't need to load the whole file into memory
- Each line is independently valid JSON — a corruption in one line doesn't break the rest
- It's easy to append new transactions (just `file.write(json.dumps(tx) + "\n")`)
- Many ML tools (HuggingFace datasets, LangChain loaders) natively support JSONL

### Sample Records

```json
{"transaction_id": "TXN-0001", "sender_account": "ACC-8173", "receiver_account": "ACC-2148", "amount": 2345.23, "date": "2024-03-12T14:30:00Z", "currency": "EUR", "is_fraudulent": false, "fraud_type": "None"}
{"transaction_id": "TXN-0003", "sender_account": "ACC-9821", "receiver_account": "ACC-9821", "amount": 9578.12, "date": "2024-02-22T14:30:00Z", "currency": "EUR", "is_fraudulent": true, "fraud_type": "Smurfing"}
```

### Field-by-Field Explanation

| Field | Type | What It Is | Why It Exists |
|---|---|---|---|
| `transaction_id` | string | Unique ID like "TXN-0001" | Identifies each transaction in results |
| `sender_account` | string | Account that sent money (e.g., "ACC-8173") | Source of funds |
| `receiver_account` | string | Account that received money | Destination of funds |
| `amount` | float | Transaction amount in EUR | Key signal for fraud detection |
| `date` | string (ISO 8601) | When the transaction occurred | Temporal context |
| `currency` | string | Always "EUR" here | Required for real AML analysis |
| `is_fraudulent` | boolean | Ground truth label | For dataset construction control |
| `fraud_type` | string | "Smurfing", "Velocity Fraud", or "None" | Fraud category label |

### Fraud Distribution
Per batch of 10: **2 Smurfing + 1 Velocity Fraud + 7 Normal**.
This is a 20% fraud rate — realistic for synthetic AML datasets (real-world fraud rates are typically 0.1-1%, but synthetic datasets use higher rates to ensure the model sees enough positive examples).

### What Is Smurfing?
Smurfing is structuring transactions just below reporting thresholds to avoid automatic detection. In the EU, banks must report transactions above 10,000 EUR. Smurfing transactions are typically 9,500-9,990 EUR. The name comes from the Smurfs cartoon — many small actors (smurfs) doing what one large actor cannot.

In this dataset: `amount` between 9500 and 9990 EUR signals Smurfing.

### What Is Velocity Fraud?
A pattern where an account makes an unusually high number of transactions in a short time window — "velocity" referring to speed/rate. In this dataset, it's marked with `fraud_type: "Velocity Fraud"` and `is_fraudulent: true`.

### Hidden Assumption
`sender_account == receiver_account` appears in some transactions (e.g., TXN-0003). In real banking, this would be a self-transfer, not a suspicious pattern. However, the LLM investigator uses the `is_fraudulent` flag from the data, so this doesn't affect benchmark correctness.

### Interview Explanation

**30-second:** "The JSONL format was chosen over JSON array because it supports streaming — you can read one transaction at a time without loading the whole file. It also tolerates single-line corruption without invalidating the whole dataset. The fraud distribution (20% fraud rate) is intentionally higher than real-world rates to ensure sufficient positive examples for a 50-transaction synthetic dataset."

---

## 14. File: `data/benchmark_results.csv`

### What Is It?
The output file produced by `benchmark.py`. A CSV (Comma Separated Values) table — one row per transaction per model.

### Why CSV?
- Human-readable (opens in Excel, Google Sheets)
- Machine-readable (pandas, any analytics tool)
- Lightweight — no database needed
- Industry standard for experiment results

### Column Definitions

| Column | What It Means |
|---|---|
| `transaction_id` | Which transaction was processed |
| `model_label` | Human-readable label ("Teacher (Q8)" or "Student (Q4)") |
| `model_tag` | Exact Ollama model tag used |
| `revisions_needed` | How many times the Explainer had to retry (0 = first-try success) |
| `passed_schema_audit` | True/False — did the final output pass Pydantic validation? |
| `notes` | Error message if pipeline crashed (empty if success) |

### Reading the Results
All 40 rows show `revisions_needed=0` and `passed_schema_audit=True`. This means:
- No transaction caused a schema validation failure
- The self-correction loop was never triggered
- Both models are equally capable at this task

### The Sentinel Value `-1`
Looking at `benchmark.py`, if the pipeline throws an unhandled exception, `revisions_needed` is set to `-1` as a sentinel value indicating a crash rather than a schema failure. None appear in the actual results, which means the pipeline never crashed.

---

## 15. Folder: `src/`

### Why This Folder Exists
All Python source code lives here. The `src/` layout (also called "src-layout") is a Python best practice that:
- Prevents the local `src/` directory from accidentally being importable without installation
- Clearly separates source code from configuration, documentation, and data
- Makes the project structure predictable to other Python developers

### Responsibilities
- Contains all executable Python files
- Houses the `nodes/` sub-package
- Is the only directory `python` commands run from

### How the Project Flows Through This Folder

```
data_generator.py  <- Run first (one time) to create dataset
graph.py           <- Run standalone to test the pipeline
benchmark.py       <- Run last to generate results
```

The `state.py`, `graph.py`, `nodes/` files are a **library** used by `benchmark.py`.

---

## 16. File: `src/__init__.py`

### What Is It?
An empty file. Literally 0 bytes.

### Why Does It Exist?
In Python, a directory becomes an **importable package** only if it contains an `__init__.py` file. Without this file, `from state import AgentState` would fail — Python wouldn't know to look inside `src/`.

### Why Is It Empty?
No initialization logic is needed. The file just needs to exist to mark `src/` as a package. When you need package-level setup (like configuring logging or exposing a public API), you put code here. For this project, it's a pure marker file.

### What Would Happen If We Removed It?
Depending on Python version and how scripts are run, import statements like `from state import AgentState` might fail. The `sys.path.append()` calls in some files compensate for this — which is the workaround for running scripts directly rather than as a package.

---

## 17. File: `src/state.py`

### What Is It?
A single Python class definition that describes the **shared memory** that all three pipeline nodes read from and write to.

### Full File, Line by Line

```python
from typing import TypedDict, Annotated, Optional   # Line 1
import operator                                       # Line 2
```

**Line 1:** Imports three types from Python's `typing` module:
- `TypedDict`: Lets you define a dictionary with a fixed set of typed keys — like a struct in C or an interface in TypeScript.
- `Annotated`: Lets you attach extra metadata to a type annotation. Used here to give LangGraph a merge instruction for `revision_count`.
- `Optional`: Marks a field as "can be this type OR None."

**Line 2:** Imports Python's `operator` module. `operator.add` is the `+` operator as a function — used by LangGraph to merge `revision_count` updates via addition instead of override.

```python
class AgentState(TypedDict):              # Line 4
    transaction_data: dict                # Line 5
    is_suspicious: bool                   # Line 6
    explanation_json: Optional[dict]      # Line 7
    validation_errors: Optional[str]      # Line 8
    revision_count: Annotated[int, operator.add]  # Line 9
    model_name: str                       # Line 10
```

**Line 4:** `class AgentState(TypedDict)` — Defines a typed dictionary. This is a type contract — every node agrees to work with this shape.

**Line 5:** `transaction_data: dict` — The raw bank transaction object. Contains `transaction_id`, `amount`, `sender_account`, etc. This is the input data that flows through the entire pipeline unchanged.

**Line 6:** `is_suspicious: bool` — Set by the Investigator node. `True` = fraud flagged. `False` = transaction appears clean. The Explainer reads this to decide what to do next.

**Line 7:** `explanation_json: Optional[dict]` — The JSON audit report written by the Explainer. Starts as `None`, becomes a dict after the Explainer runs.

**Line 8:** `validation_errors: Optional[str]` — Holds the Pydantic error message if the Auditor rejects the explanation. `None` means validation passed. This is the "feedback signal" in the self-correction loop.

**Line 9:** `revision_count: Annotated[int, operator.add]` — The most architecturally interesting line.

In LangGraph, when a node returns an update to state, the update is **merged** with the existing state. For most fields, the merge is a simple override. But for `revision_count`, `Annotated[int, operator.add]` tells LangGraph: **"use addition as the merge strategy."** So when the Auditor returns `{"revision_count": 1}`, LangGraph adds 1 to the current value instead of replacing it. This is how a counter accumulates across multiple graph steps.

**Line 10:** `model_name: str` — The Ollama model tag to use. This is how `benchmark.py` switches between Teacher and Student without modifying any node source code. The benchmark runner injects the model name into the initial state, and every node reads it from there.

### Design Pattern: Blackboard Pattern

All three nodes communicate exclusively through the `AgentState` dictionary. They never call each other directly. This is the **Blackboard Pattern** — a shared workspace where agents read current state, do work, and write updates. Benefits:
- Nodes are completely decoupled — you can test each one in isolation
- Adding a new node is trivial — just define what state keys it reads/writes
- The state provides a complete audit trail of what happened at each step

### What Would Happen If We Removed This File?
Every node would fail to import. The entire graph would be un-buildable.

### Interview Explanation

**30-second:** "The `AgentState` TypedDict is the shared memory for the entire LangGraph pipeline. The key insight is the `revision_count` field annotated with `operator.add` as a merge reducer — each Auditor failure returns `{'revision_count': 1}` and LangGraph accumulates these additions rather than overriding the count. The `model_name` field enables zero-code model switching by making the LLM selection part of the runtime state rather than a compile-time constant."

**2-minute:** "In LangGraph, state is immutable between steps — each node returns only the keys it changes, and LangGraph merges them with the previous state. The merge strategy is configurable per field. For booleans, strings, and dicts, the default merge is override. But for `revision_count`, we use `Annotated[int, operator.add]`, which tells LangGraph to add the new value to the existing one. This turns revision counting into an accumulator. The `Optional` typing on `explanation_json` and `validation_errors` explicitly models the fact that these start as `None` and become populated only after specific nodes run — this is self-documenting code that makes the pipeline's lifecycle obvious."

**Common follow-up questions:**

| Question | Strong Answer |
|---|---|
| Why TypedDict instead of a Pydantic model? | TypedDict is lighter — it's a pure type hint, not a runtime class. LangGraph is designed around TypedDicts. Using Pydantic for state would add overhead and isn't the convention. |
| What happens if a node returns a key not in AgentState? | LangGraph will raise an error at runtime. The schema enforcement is why TypedDict is used — it documents the contract. |
| Why is revision_count not reset between transactions? | It IS reset — benchmark.py creates a fresh `initial_state` with `revision_count: 0` for each transaction invocation. |

---

## 18. File: `src/graph.py`

### What Is It?
The file that assembles the three nodes into a working pipeline. This is the "wiring diagram" of the project.

### Full File, Line by Line

```python
import sys, os, json         # Lines 1-3
```
Standard library imports. `sys` and `os` for path manipulation. `json` for formatting output in the test block.

```python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Line 6
```
**This line is critical.** When you run `python src/graph.py` from the project root, Python's import system doesn't automatically know to look inside `src/`. This line:
1. `__file__` = the current file path (`src/graph.py`)
2. `os.path.abspath(__file__)` = absolute path
3. `os.path.dirname(...)` = the directory containing it (`src/`)
4. `sys.path.append(...)` = adds `src/` to Python's search path

After this line, `from state import AgentState` works because Python now searches inside `src/`.

```python
from langgraph.graph import StateGraph, END  # Line 8
```
Imports:
- `StateGraph`: The graph class — you add nodes and edges to it
- `END`: A special sentinel constant that, when a node routes to it, means "stop the pipeline"

```python
from state import AgentState                  # Line 9
from nodes.investigator import investigate    # Line 10
from nodes.explainer import explain_verdict   # Line 11
from nodes.auditor import audit_explanation   # Line 12
```
Imports the state definition and all three node functions — the building blocks assembled into the graph.

```python
def route_after_auditor(state: AgentState):  # Line 14
```
A **routing function** — a pure Python function that LangGraph calls to decide which node to execute next after the Auditor runs.

```python
    if state.get("validation_errors") is None:   # Line 16
        return END                                # Line 17
```
If `validation_errors` is `None`, the explanation passed validation. Return `END` to terminate the pipeline.

```python
    if state.get("revision_count", 0) >= 3:   # Line 20
        print("MAX REVISIONS REACHED...")      # Line 21
        return END                             # Line 22
```
**Hard stop (Circuit Breaker Pattern).** If the Explainer has already tried 3 times and still can't produce valid JSON, give up. Without this guard, a stubborn LLM could loop forever, consuming infinite compute. This prevents infinite loops in AI systems.

```python
    return "explainer"                        # Line 25
```
If validation failed but we haven't hit 3 attempts, route back to the `"explainer"` node.

```python
workflow = StateGraph(AgentState)             # Line 28
```
Creates an empty graph that will manage state of type `AgentState`. Note: this runs at **module level** — it executes when the file is imported.

```python
workflow.add_node("investigator", investigate)    # Line 31
workflow.add_node("explainer", explain_verdict)   # Line 32
workflow.add_node("auditor", audit_explanation)   # Line 33
```
Registers each node function with a string name. The name is how edges reference the node.

```python
workflow.set_entry_point("investigator")      # Line 36
```
Declares that when the graph starts executing, the first node to run is `"investigator"`.

```python
workflow.add_edge("investigator", "explainer")  # Line 37
workflow.add_edge("explainer", "auditor")        # Line 38
```
**Unconditional edges.** After the Investigator always go to the Explainer. After the Explainer always go to the Auditor.

```python
workflow.add_conditional_edges(              # Line 39
    "auditor",                               # Line 40 - source node
    route_after_auditor,                     # Line 41 - routing function
    {                                        # Line 42 - mapping of return values to destinations
        END: END,                            # Line 43
        "explainer": "explainer"             # Line 44
    }
)
```
**Conditional edge.** After the Auditor runs, call `route_after_auditor(state)`. The return value is mapped through the dict. The dict is a safety mapping — LangGraph validates that all possible return values are handled.

```python
app = workflow.compile()                     # Line 49
```
**Compilation.** Converts the graph definition into an executable Runnable object. After this, `app.invoke(state)` runs the full pipeline. This is where LangGraph validates the graph structure.

```python
if __name__ == "__main__":                   # Line 51
```
The test block. Code inside this `if` only runs when you execute `python src/graph.py` directly — NOT when another file imports from `graph.py`. This is a Python standard for putting standalone test code in the same file as library code.

### Why the Graph Is Defined at Module Level

The `workflow` and `app` objects are defined at module level (not inside a function). This means they're created once when `graph.py` is imported. When `benchmark.py` does `from graph import app`, the graph is already compiled and ready — no re-compilation for each transaction. This is a performance optimization.

### Interview Explanation

**30-second:** "`graph.py` is the wiring diagram — it registers the three node functions with LangGraph, defines the edges (investigator → explainer → auditor), and adds a conditional edge out of the Auditor that either terminates the pipeline or loops back to the Explainer with a hard cap of 3 revisions."

**2-minute:** "The key architectural decision is compiling the graph at module level, not inside a function. This means when `benchmark.py` imports `from graph import app`, the compilation happens once. For 40 benchmark executions, this saves 40 compilation calls. The `route_after_auditor` function is a pure Python function — no LLM involved — making routing deterministic, fast, and testable. The `revision_count >= 3` circuit breaker is critical: without it, a model that consistently hallucinates wrong JSON would create an infinite loop."

---

## 19. File: `src/data_generator.py`

### What Is It?
A one-time script that was run before the benchmark to generate the synthetic transaction dataset (`data/transactions.jsonl`).

### Why Does It Exist?
The benchmark needs realistic bank transaction data. Real bank data is:
1. Confidential — can't use it in a public project
2. Complex — needs cleaning and anonymization
3. Imbalanced — fraud rate is approximately 0.1%, making 50-transaction benchmarks useless

So the solution is to generate synthetic data using an LLM, with precise control over fraud distribution.

### Why Use an LLM to Generate Data?
An LLM generates transaction data that looks realistic — plausible account numbers, realistic amounts, ISO 8601 dates. A simple random number generator would produce data that no investigator LLM would find contextually believable.

However, the LLM is **constrained** — we don't let it decide whether a transaction is fraudulent. We tell it what to generate and override its output with the ground truth values.

### Module-Level Constants

```python
TOTAL_TRANSACTIONS = 50   # Total to write across all batches
BATCH_SIZE = 10           # Transactions per LLM call loop
OUTPUT_PATH = "data/transactions.jsonl"

FRAUD_SCHEDULE = (
    ["Smurfing"] * 2 + ["Velocity Fraud"] * 1 + ["None"] * 7
)
```

`FRAUD_SCHEDULE` creates the list: `["Smurfing", "Smurfing", "Velocity Fraud", "None", "None", "None", "None", "None", "None", "None"]` — 10 elements representing the fraud distribution per batch. It's shuffled before each batch so the order is random but the count is always exactly 2 Smurfing, 1 Velocity, 7 Normal.

### Function: `make_prompt(fraud_type, tx_index)`

**What it does:** Builds a text prompt instructing the LLM to generate exactly one transaction of the specified fraud type.

**Key design:** The `amount_hint` is fraud-type specific:
- Smurfing: amount MUST be 9500-9990 (the smurfing range)
- Velocity Fraud: any amount 1000-50000
- Normal: realistic everyday amount under 5000

The `{{` and `}}` in the f-string are escaped braces — in Python f-strings, `{{` produces a literal `{` in the output. The `{tx_index:04d}` part formats the number as a 4-digit zero-padded integer (1 becomes "0001").

### Function: `generate_one(llm, fraud_type, tx_index)`

```python
response = llm.invoke(prompt)         # Call the LLM
raw = response.content.strip()        # Get text, remove whitespace
data = json.loads(raw)                # Parse as JSON
```

**Validation gauntlet:**
1. Check it's a `dict` (not an array or string)
2. Check all required keys exist
3. If any check fails, return `None`

**Override safety:**
```python
data["fraud_type"] = fraud_type
data["is_fraudulent"] = (fraud_type != "None")
data["transaction_id"] = f"TXN-{tx_index:04d}"
```
Even if the LLM "decided" to change the fraud type, we override it with what we asked for. This ensures dataset integrity — no hallucinated fraud types, no wrong IDs.

### Function: `generate_transactions()`

```python
llm = ChatOllama(model="llama3.2", temperature=0.7, format="json")
```
Uses Llama 3.2 (a small 3B model) for data generation — it's faster than Qwen 2.5 7B for this simpler task. `temperature=0.7` gives variety in the generated data.

```python
schedule = FRAUD_SCHEDULE.copy()
random.shuffle(schedule)
```
Key: `.copy()` creates a fresh copy before shuffling, so the master template is never modified.

```python
with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
    for tx in batch_rows:
        f.write(json.dumps(tx) + "\n")
```
Opens the file in **append mode** (`"a"`). This means running the script multiple times will keep adding entries (potentially duplicating data). For one-time use, this is fine.

### Hidden Bug
Running `data_generator.py` twice produces duplicate transactions. No idempotency check exists. Production fix: open in write mode (`"w"`) or check if file already exists.

### Interview Explanation

**30-second:** "`data_generator.py` uses Llama 3.2 locally to generate synthetic AML transaction data with controlled fraud distribution — 20% fraud rate with exactly 2 Smurfing and 1 Velocity Fraud per 10 transactions. The key engineering decision was generating one transaction per LLM call rather than a batch, because Ollama's JSON mode only produces single objects, not arrays. LLM outputs are overridden with ground truth labels to ensure dataset integrity."

---

## 20. File: `src/benchmark.py`

### What Is It?
The main evaluation script — the "experiment runner." This is what you run to execute the actual research question.

### Its Responsibilities
1. Load 20 transactions from the dataset
2. For each of two models (Teacher Q8, Student Q4): run all 20 transactions through the pipeline
3. Record the outcome of each run
4. Write a CSV of all results
5. Print a summary

### Module-Level Configuration

```python
MODELS = {
    "Teacher (Q8)": "qwen2.5:7b-instruct-q8_0",
    "Student (Q4)": "qwen2.5:7b",
}
```
Adding a third model for comparison would be as simple as adding one line here.

```python
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "transactions.jsonl")
```
Constructs an absolute path anchored to the location of `benchmark.py` itself. Works regardless of where you run the script from.

### Function: `load_transactions(path, limit)`

Defensive reading — each line is:
1. Stripped of whitespace
2. Skipped if empty
3. Parsed as JSON — if it fails, skip the line (don't crash)
4. Checked to be a dict
5. Stops early once `limit` transactions are collected

This is **robust parsing** — the function handles malformed files gracefully.

### Function: `run_evaluation()`

**Key pattern — model injection:**
```python
initial_state = {
    ...
    "model_name": model_tag,  # <- injected here
}
```
The model tag is injected as part of the initial state. This is how the same compiled `app` graph runs with different models — the model selection is part of the state, not hardcoded in the nodes.

**Outer try/except:**
```python
try:
    final_state = app.invoke(initial_state)
except Exception as exc:
    results.append({ ..., "revisions_needed": -1, "passed_schema_audit": False })
    continue
```
Catches any catastrophic pipeline failure (e.g., Ollama server not running). Records it as `revisions_needed: -1` — a sentinel value distinguishing "pipeline crash" from "schema validation failure."

**Metrics extraction:**
```python
passed = final_state.get("validation_errors") is None
revisions = final_state.get("revision_count", 0)
compliance_rate = (success_count / len(transactions)) * 100
avg_revisions = total_revisions / len(transactions)
```

**CSV writing:**
```python
writer = csv.DictWriter(f, fieldnames=fieldnames)
writer.writeheader()
writer.writerows(results)
```
`csv.DictWriter` takes a list of dicts and writes each as a CSV row. `newline=""` is required by the `csv` module on Windows to avoid double newlines.

### Interview Explanation

**30-second:** "`benchmark.py` is the experiment runner. It loads 20 transactions, then for each of two models runs the full LangGraph pipeline and records whether the Pydantic schema validation passed and how many self-correction loops were needed. The model is injected into the initial state, making the same compiled graph object work for both model variants. Results are written to CSV."

**2-minute:** "The architecture of `benchmark.py` demonstrates the Strategy Pattern — the model selection is a runtime parameter injected into the state, not compiled into the graph. This means zero code changes between Teacher and Student benchmarks. The outer try/except is important: it distinguishes catastrophic failures (pipeline crashes, Ollama unavailable) from the expected failure mode (schema validation failing), using -1 as a sentinel for the former. The compliance rate formula is `(success_count / len(transactions)) * 100` — where success means `validation_errors is None` in the final state, regardless of how many revisions were needed."

---

## 21. Folder: `src/nodes/`

### Why This Folder Exists
Each pipeline step (Investigator, Explainer, Auditor) is a separate concern with different responsibilities. Placing them in their own sub-folder:
- Groups related code logically
- Makes each node independently testable
- Makes the pipeline's structure visible in the directory layout
- Follows the **Single Responsibility Principle**: each file does exactly one thing

### Common Interface (Implicit Contract)
All three nodes follow this contract:
```python
def node_function(state: AgentState) -> dict:
    # Read from state
    # Do work
    # Return dict of state updates
```
This uniformity is what makes LangGraph's graph model work.

---

## 22. File: `src/nodes/__init__.py`

### What Is It?
Empty file. Same purpose as `src/__init__.py` — marks `nodes/` as a Python package.

### Why Does It Exist?
Without it, `from nodes.investigator import investigate` in `graph.py` would fail.

---

## 23. File: `src/nodes/investigator.py`

### What Is It?
Node 1 of the pipeline. The LLM-based fraud detector. Given a raw transaction, it decides if the transaction looks suspicious.

### Full File, Line by Line

```python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
From two directories deep (`nodes/investigator.py`), `dirname` is called twice to walk up to `src/`.

```python
llm = ChatOllama(model=model, temperature=0.0, format="json")
```
- `temperature=0.0`: Zero randomness. The Investigator should make the same decision every time given the same transaction — reproducibility matters for compliance. High temperature would produce inconsistent verdicts.
- `format="json"`: Instructs Ollama to constrain output to valid JSON.

**The Prompt:**
```
You are an Anti-Money Laundering (AML) specialist.
Suspicious patterns to look for:
- Smurfing: amount between 9500 and 9990 EUR
- Velocity Fraud: is_fraudulent flag is true with fraud_type "Velocity Fraud"
- Any transaction marked is_fraudulent = true.
```

Note: The prompt includes `is_fraudulent` as a detection hint. This means the Investigator is essentially reading the ground truth label from the data. In a real AML system, you wouldn't have this label. This is acceptable here because the benchmark is testing the **Explainer's output quality**, not the Investigator's detection accuracy.

**The Fail-Safe Default:**
```python
except (json.JSONDecodeError, ValueError):
    is_suspicious = True  # Default to suspicious on parse error
    reason = "Parse error – defaulting to suspicious."
```
If the LLM produces malformed JSON and we can't parse the response, we default to `is_suspicious = True`. This is the **Fail-Safe Pattern** — better to flag a false positive than to miss a real fraud case. This is called the **Precautionary Principle** in regulatory AI design.

```python
return {"is_suspicious": is_suspicious}
```
Returns only the key that this node changes. The `reason` is printed but not stored in state — a minor trade-off. In a production system, the reason would be stored for the Explainer to use.

### Hidden Assumption
The node reads `is_fraudulent` from the transaction data in its prompt. This creates a **data leak** — the "label" from the synthetic dataset is visible to the detector. In a real system, this field wouldn't exist in incoming transactions.

### Interview Explanation

**30-second:** "The Investigator is Node 1 — it sends the raw transaction to the LLM with a prompt describing Smurfing and Velocity Fraud patterns, then parses the JSON response. Temperature is 0.0 for deterministic results. The fail-safe is to default to suspicious on parse errors — better false positives than false negatives in a compliance context."

**Common follow-up:** "Why not use rule-based detection instead of an LLM?" — Rule-based detection would be more reliable for the specific patterns in this synthetic dataset. The LLM approach is used because (1) the benchmark is testing LLM behavior, (2) real-world AML patterns are too complex and evolving for simple rules, and (3) LLMs can detect novel patterns not explicitly programmed.

---

## 24. File: `src/nodes/explainer.py`

### What Is It?
Node 2 of the pipeline. The LLM-based compliance writer. Given a suspicious transaction and the Investigator's verdict, it writes a formal JSON audit report in BaFin-required format.

### Full File, Line by Line

```python
model = state["model_name"]
llm = ChatOllama(model=model, temperature=0.1, format="json")
```
`temperature=0.1`: Very low randomness. The Explainer needs to produce structured, consistent JSON — high temperature would cause structural hallucinations (missing brackets, wrong key names). A small amount of randomness (0.1 vs 0.0) is used to allow slight variation in evidence wording while keeping structure stable.

**Early Return for Clean Transactions:**
```python
if not is_suspicious:
    clean_json = {
        "verdict": "CLEAN",
        "confidence_score": 1.0,
        "rule_violated": "None",
        "evidence_list": []
    }
    return {"explanation_json": clean_json, "validation_errors": None}
```
If the Investigator said it's clean, the Explainer doesn't call the LLM at all — it immediately returns a hardcoded CLEAN response. This:
1. Saves LLM compute for non-suspicious transactions
2. Ensures the CLEAN response always passes schema validation (it's hardcoded, not LLM-generated)
3. Sets `validation_errors: None` to signal to the Auditor that validation already passed

**The Self-Correction Mechanism:**
```python
feedback_loop = ""
if prev_errors:
    feedback_loop = f"\nCRITICAL: Your previous response was REJECTED by the Auditor for violating the schema constraints.\nParser Error: {prev_errors}\nFix your structural output accordingly."
```
On a second or third pass through the Explainer, `prev_errors` contains the Pydantic error message. This message is injected into the prompt so the LLM knows exactly what went wrong. The word "CRITICAL" and all-caps formatting is deliberate — LLMs respond to emphasis cues in prompts. This is **prompt engineering** for self-correction.

**The BaFin Schema Blueprint in the Prompt:**
```
"verdict": "Must be 'SUSPICIOUS'",
"confidence_score": "Float value between 0.0 and 1.0",
"rule_violated": "Name of pattern detected",
"evidence_list": ["List of empirical strings"]
```
The schema description is deliberately inline (not a formal JSON Schema). This is because the schema is simple enough that describing it in prose is clearer for the LLM than a formal specification.

### Design Pattern: Self-Healing Loop (Feedback Control)

```
Explainer -> Auditor -> (error?) -> Explainer (with error context) -> Auditor -> ...
```

This is a **Feedback Control Loop** — a control theory concept applied to LLM output quality. The error message is the "control signal" fed back into the system to correct its behavior.

### Interview Explanation

**30-second:** "The Explainer is Node 2. It short-circuits for clean transactions — returns a hardcoded CLEAN JSON without an LLM call. For suspicious transactions, it generates a BaFin audit JSON using the LLM. If this is a retry, the Pydantic error from the Auditor is injected into the prompt as a 'CRITICAL' correction directive. This is the self-healing feedback loop."

**Common follow-up:** "What if the LLM ignores the error correction?" — After 3 attempts, the circuit breaker in `route_after_auditor` terminates the pipeline. The transaction is recorded as `passed_schema_audit: False`. In this benchmark, this never happened.

---

## 25. File: `src/nodes/auditor.py`

### What Is It?
Node 3 of the pipeline. The deterministic (non-AI) validator. Given the Explainer's JSON output, it checks whether it conforms to the required BaFin compliance schema.

### Why Is This Node NOT an LLM?

This is a crucial architectural decision. The Auditor uses **Pydantic** (deterministic, rule-based validation), not an LLM. Why?

1. **Determinism:** Pydantic always gives the same answer for the same input. An LLM-based validator might sometimes pass malformed JSON and sometimes reject valid JSON.
2. **Regulatory requirement:** A compliance audit trail must be mathematically verifiable. "The LLM thought it was valid" is not a compliance argument. "Pydantic v2 with pattern `^(SUSPICIOUS|CLEAN)$`" is.
3. **Speed:** Pydantic validation takes microseconds. An LLM call takes seconds.
4. **Error precision:** Pydantic's `ValidationError` gives exact, structured error messages that pinpoint which field failed and why — exactly what the Explainer needs for self-correction.

### The Schema — Field by Field

```python
class BaFinComplianceSchema(BaseModel):
    verdict: str = Field(..., pattern="^(SUSPICIOUS|CLEAN)$")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    rule_violated: str
    evidence_list: List[str]
```

**`verdict`:** Must be exactly "SUSPICIOUS" or "CLEAN". The regex `^(SUSPICIOUS|CLEAN)$` anchors to start and end — "Suspicious" (lowercase s) would fail. The `...` means required — no default value.

**`confidence_score`:** Must be a float between 0.0 and 1.0 inclusive. `ge=0.0` means "greater than or equal to 0.0". `le=1.0` means "less than or equal to 1.0". If the LLM outputs `0.85` as a string `"0.85"`, Pydantic v2 auto-coerces it to float — so `"0.85"` passes. But `"high"` would fail.

**`rule_violated`:** Must be a string. No further constraints. Any string passes.

**`evidence_list`:** Must be a list where every element is a string. An empty list `[]` is valid.

### The Validation Logic

```python
if not explanation:
    return {"validation_errors": "Missing explanation JSON completely.", "revision_count": 1}
```
First check: is there even anything to validate? An empty dict `{}` is falsy in Python.

```python
BaFinComplianceSchema(**explanation)
```
Dict unpacking — expands `{"verdict": "SUSPICIOUS", ...}` into `BaFinComplianceSchema(verdict="SUSPICIOUS", ...)`. Pydantic validates all fields in one call.

```python
except ValidationError as e:
    error_msg = str(e)
    return {"validation_errors": error_msg, "revision_count": 1}
```
The error message is converted to a string and stored. `revision_count: 1` increments the counter via the `operator.add` reducer.

### Interview Explanation

**30-second:** "The Auditor is the deterministic validation gate — it uses Pydantic's `BaseModel` to enforce the BaFin schema with zero tolerance. The choice to use Pydantic instead of an LLM-based validator is a critical architectural decision: compliance validation must be deterministic, fast, and auditable. Pydantic's `ValidationError` message is then fed back into the Explainer's prompt for self-correction."

**Common follow-up:** "What's the regex pattern for verdict?" — `^(SUSPICIOUS|CLEAN)$`. The `^` anchors to start, `$` anchors to end, so there can be no extra characters. "SUSPICIOUS " (trailing space) would fail. This strictness is intentional for regulatory purposes.

---

## 26. Data Flow: End to End

```
data/transactions.jsonl (JSONL file on disk)
    |
    v benchmark.py reads line by line
transaction dict = {transaction_id, sender, receiver, amount, date, currency, is_fraudulent, fraud_type}
    |
    v benchmark.py constructs initial_state
AgentState = {
    transaction_data: <the dict above>,
    is_suspicious: False,        <- default
    explanation_json: None,      <- not yet generated
    validation_errors: None,     <- not yet checked
    revision_count: 0,           <- counter starts at 0
    model_name: "qwen2.5:7b"     <- injected by benchmark
}
    |
    v app.invoke(initial_state) -- enters LangGraph pipeline
-------------------------------------------------------------
INVESTIGATOR NODE
  reads:  transaction_data, model_name
  sends:  prompt to LLM -> "is this suspicious?"
  gets:   {"is_suspicious": true, "reason": "..."}
  writes: {is_suspicious: True} -> merged into state
-------------------------------------------------------------
    v unconditional edge
EXPLAINER NODE
  reads:  transaction_data, is_suspicious, model_name, validation_errors (None on 1st pass)
  if not suspicious: returns hardcoded CLEAN JSON, skip LLM call
  else:
    sends:  prompt to LLM -> "write BaFin audit JSON"
    gets:   {"verdict": "SUSPICIOUS", "confidence_score": 0.85, ...}
    writes: {explanation_json: <above dict>} -> merged into state
-------------------------------------------------------------
    v unconditional edge
AUDITOR NODE
  reads:  explanation_json
  validates: BaFinComplianceSchema(**explanation_json)
  if valid:  writes {validation_errors: None, revision_count: 0}
  if invalid: writes {validation_errors: "error msg", revision_count: 1}
-------------------------------------------------------------
    v conditional edge -> route_after_auditor()
if validation_errors is None -> END
if validation_errors + revision_count >= 3 -> END (with failure)
if validation_errors + revision_count < 3 -> back to EXPLAINER NODE
-------------------------------------------------------------
END -> final_state returned to benchmark.py
    |
    v
benchmark.py reads:
  final_state["validation_errors"] -> passed = True if None
  final_state["revision_count"] -> how many retries
    |
    v
Appends row to results list
    |
    v after all transactions and both models
Writes data/benchmark_results.csv
```

---

## 27. Design Patterns Used

### 1. State Machine (via LangGraph)
The pipeline is a **Finite State Machine** — at any point, the system is in one of three states (Investigator, Explainer, Auditor), transitions between states are defined, and conditional transitions (routing) are encoded as functions.

### 2. Blackboard Pattern
All agents share a single `AgentState` "blackboard." No direct agent-to-agent communication. Reduces coupling, improves testability.

### 3. Feedback Control Loop
The Auditor -> Explainer retry loop is a **feedback control system**: the Auditor's error message is the feedback signal that corrects the Explainer's behavior.

### 4. Circuit Breaker
The `revision_count >= 3` hard stop in `route_after_auditor` is a **circuit breaker** — it prevents cascading failure (infinite loops) when a downstream system (the LLM) behaves unexpectedly.

### 5. Strategy Pattern
The `model_name` field in state allows dynamic model selection without modifying any node code. The "strategy" (which LLM to use) is injected at runtime.

### 6. Fail-Safe Default
The Investigator's `except` clause defaults to `is_suspicious = True` — the **Fail-Safe Pattern**. When the system can't make a determination, it defaults to the safer outcome (flag for review rather than clear).

### 7. Defensive Programming
In `data_generator.py`, after the LLM generates a transaction, critical fields are overridden to ensure dataset integrity — don't trust external inputs to produce exactly what you asked for.

### 8. Single Responsibility Principle
Each file has exactly one job:
- `state.py` — defines shared state
- `graph.py` — wires the pipeline
- `investigator.py` — detects fraud
- `explainer.py` — generates reports
- `auditor.py` — validates schemas
- `benchmark.py` — runs the experiment
- `data_generator.py` — creates the dataset

---

## 28. Hidden Assumptions, Edge Cases and Potential Bugs

### Bug 1: Unpinned Dependencies
`requirements.txt` has no version pins. LangGraph has breaking API changes between minor versions. Future installs may fail silently or produce different behavior. **Fix:** Pin versions with `pip freeze > requirements.txt`.

### Bug 2: Investigator Reads Ground Truth Label
The Investigator prompt includes `json.dumps(tx)` which contains `is_fraudulent: true/false` and `fraud_type: "Smurfing"`. The LLM essentially reads the answer from the data. Fine for benchmarking the Explainer, but the Investigator's "intelligence" is artificial here.

### Bug 3: Append Mode in data_generator.py
Opening the output file in append mode (`"a"`) means running the generator twice produces duplicate transactions. **Fix:** Check if file exists, or use write mode.

### Bug 4: sys.path Manipulation
The `sys.path.append()` calls are a workaround that can cause subtle import issues depending on working directory and Python version. **Fix:** Use an installable package with `pyproject.toml`.

### Bug 5: validation_errors Set by Both Explainer and Auditor
Both nodes can set `validation_errors`, but the distinction is lost in the final state — you can't tell if failure was a parse error or a schema violation.

### Edge Case: Empty evidence_list Passes Validation
`evidence_list: List[str]` accepts an empty list `[]`. A suspicious transaction with an empty evidence list would pass schema validation but be meaningless for compliance purposes.

### Edge Case: Unicode in Error Messages
Pydantic v2 error messages can contain Unicode. If any downstream code doesn't handle Unicode, encoding errors could occur.

### Hidden Assumption: Ollama is Running
Every LLM call assumes Ollama is running at `localhost:11434`. If Ollama is stopped, the outer try/except in `benchmark.py` catches this but reports it as `revisions_needed=-1`, losing the distinction between "model failed" and "server unavailable."

---

## 29. Benchmark Results Explained

### What the Results Show

| Model | Quantization | Size | Schema Compliance | Avg Revisions |
|---|---|---|---|---|
| qwen2.5:7b-instruct-q8_0 | 8-bit | 8.1 GB | 100% | 0.00 |
| qwen2.5:7b | 4-bit (Q4_K_M) | 4.7 GB | 100% | 0.00 |

### What This Means
- Neither model ever needed the self-correction loop
- Both models produced perfectly valid BaFin-compliant JSON on the first attempt, 20/20 times
- Halving the model size (8 GB -> 5 GB) produced zero compliance degradation

### What This Does NOT Mean
- It doesn't mean Q4 is always sufficient for all structured output tasks
- It doesn't mean the models accurately detect fraud (ground truth labels are in the data)
- It doesn't establish whether the compliance rate drops at smaller model sizes (1B, 3B)

### The Scientific Interpretation
This is a **null result** — the hypothesis (Q4 degrades compliance) was not supported. Null results are scientifically valid. The result establishes a **lower bound**: the degradation threshold, if it exists, is below 7B parameters or below Q4 quantization.

### What to Test Next
1. Run the same benchmark with `qwen2.5:1.5b` and `qwen2.5:3b`
2. Try `Q3_K_M` and `Q2_K` quantization of the 7B model
3. Use a more complex schema (nested JSON, more fields, stricter patterns)
4. Increase the number of test transactions from 20 to 200

### Interview Explanation

**30-second:** "The benchmark found no compliance degradation between Q4 and Q8 quantization of Qwen2.5 7B — both achieved 100% schema compliance with zero retries across 20 transactions. This is a null result that's actually meaningful: it establishes that at 7B parameters, Q4 is sufficient for structured regulatory JSON output. The experiment would need to be repeated with smaller models (1B, 3B) or more aggressive quantization (Q2, Q3) to find the actual degradation threshold."

---

## 30. How to Present This Project Professionally

### To a Recruiter (30 seconds)
"I built a research benchmark in Python using LangGraph and local Ollama models to test whether quantizing AI models affects their ability to produce legally compliant output for banking regulators. I found that at 7B parameters, 4-bit and 8-bit quantization perform identically for structured JSON compliance tasks."

### To a Senior Engineer (2 minutes)
"The project is a LangGraph pipeline with three nodes — an LLM-based fraud detector, an LLM-based compliance writer, and a deterministic Pydantic validator. The validator can route back to the writer with the exact validation error for self-correction, up to 3 times. I benchmarked this pipeline across two Qwen 2.5 7B model variants — Q8 and Q4 quantization — measuring schema compliance rate and average retries across 20 synthetic AML transactions. Both models achieved 100% compliance with zero retries. The architecture uses shared TypedDict state with per-field merge reducers, a circuit breaker on the retry loop, and a Strategy pattern for zero-code model switching via state injection."

### To a Client (30 seconds)
"I built a system that automatically reviews bank transactions for suspicious patterns and generates legally required audit explanations. The system can self-correct its own reports if they don't meet regulatory standards, and I tested it with two different AI models to see which performs better."

### Master Interview Questions

| Question | Strong Answer |
|---|---|
| What is LangGraph? | A Python framework for building stateful, multi-step AI agent workflows as directed graphs with shared state, conditional routing, and parallel execution. |
| Why not use a simple function pipeline? | A function pipeline has no built-in state management, no conditional branching without if statements, and no retry loop without while loops. LangGraph expresses these patterns declaratively. |
| What is model quantization? | Reducing the bit-depth of model weights — from 8-bit (256 values per weight) to 4-bit (16 values per weight). Halves model size and RAM requirements but can reduce precision. |
| Why Pydantic for the Auditor? | Compliance validation must be deterministic. Pydantic is mathematically certain, produces structured error messages, and takes microseconds vs. seconds for an LLM call. |
| What would you improve? | Pin dependency versions, test smaller model sizes and more aggressive quantization, add timestamp to AgentState, store Investigator's reason in state for the Explainer to use, make the data generator idempotent. |
| What is BaFin? | Germany's Federal Financial Supervisory Authority — the regulatory body that oversees banks and requires structured, auditable AI decision explanations. |
| What is Smurfing? | A money laundering technique where large amounts are split into smaller transactions just below reporting thresholds (10,000 EUR in Germany) to avoid automatic detection. |
| What is the Blackboard Pattern? | A design pattern where multiple agents communicate exclusively through shared state rather than directly calling each other. Reduces coupling and improves testability. |
| What is a Circuit Breaker? | A pattern that stops a failing operation after a threshold of failures to prevent cascading failures. Here: the revision_count >= 3 hard stop prevents infinite LLM retry loops. |
| How does the model switching work? | The model name is part of the initial state injected by benchmark.py. Every node reads `state["model_name"]` to create its ChatOllama instance. No source code changes needed to switch models. |

---

## Dependency Flow Diagram

```
benchmark.py
    |
    v imports
graph.py (compiled app object)
    |
    v imports
state.py (AgentState TypedDict)
nodes/investigator.py (investigate function)
nodes/explainer.py (explain_verdict function)
nodes/auditor.py (audit_explanation function)
    |
    v each module uses
langchain_ollama (ChatOllama) -- investigator.py, explainer.py, data_generator.py
pydantic (BaseModel, ValidationError) -- auditor.py
langgraph (StateGraph, END) -- graph.py
state.py (AgentState) -- investigator.py, explainer.py, auditor.py, graph.py
```

---

## Request Flow: One Transaction Through the System

```
benchmark.py calls: app.invoke(initial_state)
    |
    v
LangGraph starts at "investigator" node
    |
    v
investigate(state) called:
  ChatOllama("qwen2.5:7b").invoke(prompt) --> HTTP POST to localhost:11434
  Ollama loads/uses cached qwen2.5:7b model
  LLM generates: {"is_suspicious": true, "reason": "Amount in smurfing range"}
  json.loads() parses response
  returns {"is_suspicious": True}
  LangGraph merges: state["is_suspicious"] = True
    |
    v LangGraph follows edge: investigator -> explainer
    |
    v
explain_verdict(state) called:
  reads is_suspicious = True, validation_errors = None
  ChatOllama("qwen2.5:7b").invoke(prompt) --> HTTP POST to localhost:11434
  LLM generates: {"verdict": "SUSPICIOUS", "confidence_score": 0.85, ...}
  json.loads() parses response
  returns {"explanation_json": {...}}
  LangGraph merges: state["explanation_json"] = {...}
    |
    v LangGraph follows edge: explainer -> auditor
    |
    v
audit_explanation(state) called:
  reads explanation_json = {"verdict": "SUSPICIOUS", ...}
  BaFinComplianceSchema(**explanation_json) -- Pydantic validates in microseconds
  All fields valid -- no ValidationError raised
  returns {"validation_errors": None, "revision_count": 0}
  LangGraph merges: state["validation_errors"] = None, state["revision_count"] += 0
    |
    v LangGraph calls: route_after_auditor(state)
  state["validation_errors"] is None -- return END
    |
    v
Pipeline exits. final_state returned to benchmark.py.
    |
    v
benchmark.py: passed = True, revisions = 0
Appends to results list.
```

---

*This document covers every file, folder, class, function, design decision, and engineering trade-off in the Agentic AML Explainability Benchmark project. Use it as your preparation guide for technical interviews, client presentations, and code modifications.*
