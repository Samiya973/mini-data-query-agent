# Mini Data-Query Agent

A small agent that answers natural-language questions about a dataset by
calling a **real Python tool** — not by generating a plausible-sounding
answer from memory.

> "What's the average ROC-AUC across my LightGBM runs?"
> → the agent calls a tool that actually computes the mean over the
> real CSV, and returns a grounded, verifiable answer.

## Why this exists

Demonstrates the core "tool-calling" / function-calling mechanic behind
agentic LLM applications: the model doesn't execute anything itself — it
requests a function call with structured arguments, the code runs the
real computation, and the result is fed back to the model to produce
the final answer.

## Architecture

User question
│
▼
LLM (openai/gpt-oss-20b via Groq) ──decides──▶ needs a number? call query_dataset(...)
│                                                   │
│                                                   ▼
│                                          agent/tools.py runs real
│                                          pandas computation over
│                                          data/experiment_runs.csv
│                                                   │
◀───────────────── tool_result ────────────────────┘
│
▼
Final answer, grounded in the real computed number

Key design decisions:
- **No free-text queries.** The model can't write arbitrary pandas/SQL —
  it can only pick from a whitelisted set of operations (`mean`, `max`,
  `min`, `sum`, `count`, `median`, `std`) plus a column name, filter, and
  optional group-by, all validated against the dataset's real schema.
- **Column names come from the live CSV**, not a hardcoded list — this
  is what stops the model from hallucinating a column that doesn't exist.
- **Errors are structured, not exceptions** — a bad column or filter
  returns `{"success": False, "error": "..."}` so the agent can explain
  the problem instead of crashing.
- **Retry logic around tool calls** — open-weight models occasionally
  emit malformed tool-call syntax; failed parses are retried automatically
  rather than crashing the session.

## Project structure

mini-data-query-agent/
├── agent/
│   ├── tools.py      # the real tool: whitelisted pandas aggregations
│   └── agent.py      # the tool-use loop (Groq API, OpenAI-compatible)
├── data/
│   └── experiment_runs.csv   # sample dataset: LightGBM credit-risk runs
├── tests/
│   └── test_tools.py         # verifies the tool's math against pandas directly
├── main.py            # CLI entry point
├── requirements.txt
└── .env.example

## Setup

```bash
git clone <your-repo-url>
cd mini-data-query-agent
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# then edit .env and add your real GROQ_API_KEY (free, no card — console.groq.com/keys)
```

## Usage

```bash
python main.py
```
What's the average ROC-AUC across all runs?
The average ROC-AUC across all 24 runs is 0.8125.

And what about just the ones where credit_utilization was the top SHAP feature?
Only one run had credit_utilization as its top SHAP feature (run 20),
with ROC-AUC of 0.8889.

## Running tests

```bash
python -m pytest tests/ -v
```

These check the tool's output against pandas computed directly — i.e.,
they verify the *tool* is correct, independent of anything the LLM does.

## Using your own data

Replace `data/experiment_runs.csv` with your own CSV (any columns work —
the tool schema is built dynamically from whatever columns are present).

## What this deliberately does NOT do

- No arbitrary code execution against the dataset (security + reliability).
- No framework (LangChain, etc.) — uses the Groq/OpenAI-compatible API's
  native tool-use directly, to keep the mechanic visible rather than
  abstracted away.