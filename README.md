<div align="center">

# 🧮 Mini Data-Query Agent

**A tool-calling LLM agent that answers questions with real numbers — never plausible-sounding guesses.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20gpt--oss--20b-F55036?style=flat-square)
![pandas](https://img.shields.io/badge/pandas-grounded%20computation-150458?style=flat-square&logo=pandas&logoColor=white)
![Tests](https://img.shields.io/badge/tests-9%20passing-4c9a2a?style=flat-square)
![No framework](https://img.shields.io/badge/agent%20framework-none-lightgrey?style=flat-square)

</div>

```
"What's the average ROC-AUC across my LightGBM runs?"
→ 0.8125, computed live from the actual CSV. Not remembered. Not estimated.
```

Most "LLM + data" demos let the model generate and execute its own code — fast to build, fragile in practice. One hallucinated column name and it either crashes or, worse, returns a confident wrong answer. This project takes the opposite bet: **the model never touches the data directly.** It can only request a call to one whitelisted, schema-validated tool. Every answer traces back to a real `pandas` computation, independently verified by a 9-test suite.

<br>

## 📋 Contents

- [Why this exists](#-why-this-exists)
- [How it works](#-how-it-works)
- [Design decisions](#-design-decisions-and-the-reasoning-behind-each)
- [Debugging notes](#-debugging-notes-what-actually-broke-in-practice)
- [Project structure](#-project-structure)
- [Quickstart](#-quickstart)
- [Tests](#-running-the-tests)
- [What this doesn't do](#-what-this-deliberately-does-not-do)

<br>

## 💡 Why this exists

Grounding is the actual hard problem in "agents over data" — not prompting. The interesting engineering here isn't getting an LLM to call a function; it's making sure that when it does, the result is **provably correct**, **can't silently fail**, and **degrades safely** when the model misbehaves. This project is a small, fully-inspectable proof of that discipline, built to be read end-to-end in one sitting.

<br>

## ⚙️ How it works

```
 User question
      │
      ▼
 LLM (openai/gpt-oss-20b via Groq)
      │
      │  decides: "I need a number — call query_dataset(...)"
      ▼
 agent/tools.py
      │  runs a real pandas aggregation against data/experiment_runs.csv
      │  validated against a schema generated from the CSV's actual columns
      ▼
 Structured result  →  fed back to the LLM  →  final grounded answer
```

> **The core mechanic, in one sentence:** the LLM requests a function call with structured arguments; it never executes anything itself — the code does, and only the code decides what "success" looks like.

<br>

## 🧠 Design decisions (and the reasoning behind each)

| Decision | Why it matters |
|---|---|
| **No free-text code generation** | The model can't write arbitrary pandas/SQL. It can only pick from a whitelisted set of operations (`mean`, `max`, `min`, `sum`, `count`, `median`, `std`) plus a column, filter, and optional group-by — all validated before anything runs. |
| **Schema generated from the live CSV** | Column names aren't hardcoded. The schema is built from whatever's actually in the dataset, so the model can't hallucinate a column that doesn't exist — and swapping in a new CSV doesn't require touching the code. |
| **9-test verification suite** | Checks the tool's math against pandas computed directly, and confirms invalid columns/operations are rejected *before* execution — independent of anything the LLM does. |
| **Structured errors, not exceptions** | A bad column or filter returns `{"success": False, "error": "..."}` so the agent can explain the problem to the user instead of the session crashing. |
| **Retry logic on malformed tool calls** | Open-weight models occasionally emit malformed tool-call syntax. Failed parses retry automatically (up to 3 attempts) instead of killing the session. |
| **4-hop cap per question** | Bounds worst-case latency and cost, and prevents a runaway multi-step chain. The honest tradeoff: a question needing a 5th hop fails rather than completing — reliability over unbounded flexibility. |
| **Smaller model on purpose** | `gpt-oss-20b` via Groq, not a 70B model. Narrow, structured tool-calling over 12 known columns doesn't need the largest model available — that capacity is better spent on open-domain reasoning tasks. |

<br>

## 🐛 Debugging notes: what actually broke in practice

Open-weight tool-calling through Groq's API turned out to be far less consistent than the docs suggest, and most of the real engineering time on this project went into hardening against it rather than writing the happy path:

- **Malformed, XML-style function calls** — the model would occasionally emit tool-call syntax that didn't match the expected JSON schema at all, closer to hand-rolled XML tags than a structured call. These had to be caught, not silently swallowed, and fed back into the retry loop.
- **`null` `tool_calls` in message history** — certain conversation states caused the API to return a message with `tool_calls: null` where a call was expected, which, if unhandled, breaks the agent's assumption that a tool-call turn always contains a call. This required explicit checks in the loop rather than trusting the response shape.

> Both failure modes are the reason the retry logic and structured-error contract exist — they weren't designed in from the start, they were added because the model broke the naive version first.

<br>

## 📁 Project structure

```
mini-data-query-agent/
├── agent/
│   ├── tools.py       # the real tool: whitelisted pandas aggregations
│   └── agent.py       # the tool-use loop (Groq API, OpenAI-compatible)
├── data/
│   └── experiment_runs.csv   # sample dataset: 24 LightGBM credit-risk runs
├── tests/
│   └── test_tools.py         # 9 tests verifying tool math against pandas directly
├── main.py             # CLI entry point
├── requirements.txt
└── .env.example
```

<br>

## 🚀 Quickstart

```bash
git clone https://github.com/Samiya973/mini-data-query-agent.git
cd mini-data-query-agent
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# add your GROQ_API_KEY to .env — free, no card required: console.groq.com/keys
python main.py
```

**Example session**

```
> What's the average ROC-AUC across all runs?
The average ROC-AUC across all 24 runs is 0.8125.

> And what about just the ones where credit_utilization was the top SHAP feature?
Only one run had credit_utilization as its top SHAP feature (run 20),
with ROC-AUC of 0.8889.
```

**Using your own data:** drop any CSV into `data/` and point the agent at it — the tool schema is built dynamically from whatever columns are present. No code changes needed.

<br>

## ✅ Running the tests

```bash
python -m pytest tests/ -v
```

9 tests validate the tool's correctness independently of the LLM — this is what makes the agent's answers verifiable rather than "probably right."

<br>

## 🚫 What this deliberately does *not* do

- **No arbitrary code execution** against the dataset — a fixed, whitelisted tool only.
- **No agent framework** (LangChain, etc.) — uses the Groq/OpenAI-compatible API's native tool-use directly, so the underlying mechanic stays visible instead of abstracted behind a library.
- **No UI.** This is a CLI by design — the project is about agent reliability engineering (grounding, validation, bounded execution), not a product demo. A Streamlit UI lives in a separate project where a user-facing interface actually adds value.

<br>

