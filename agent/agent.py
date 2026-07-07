import os
import json
from dotenv import load_dotenv
import time
from groq import Groq, BadRequestError

from agent.tools import query_dataset, get_columns, VALID_OPERATIONS

load_dotenv()

MODEL = "openai/gpt-oss-20b"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def build_tool_schema() -> list:
    columns = get_columns()
    return [
        {
            "type": "function",
            "function": {
                "name": "query_dataset",
                "description": (
                    "Run an aggregation over the experiment-run dataset "
                    "(LightGBM credit-risk hyperparameter runs). Returns a real "
                    "computed number, not an estimate. Use this any time the user "
                    "asks about averages, max/min values, counts, or comparisons "
                    "between subsets of runs. Never answer numeric questions from "
                    "memory — always call this tool."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": VALID_OPERATIONS,
                                      "description": "The aggregation to perform."},
                        "column": {"type": "string", "enum": columns,
                                   "description": "Column to aggregate. Ignored for 'count'."},
                        "filter_column": {"type": "string", "enum": columns,
                                           "description": "Optional: restrict to rows where this equals filter_value."},
                        "filter_value": {"description": "Optional: value filter_column must equal."},
                        "group_by": {"type": "string", "enum": columns,
                                     "description": "Optional: compute aggregation separately per group."},
                    },
                    "required": ["operation", "column"],
                },
            },
        }
    ]


SYSTEM_PROMPT = (
    "You are a data-query assistant for a machine learning experiment log. "
    "The dataset contains LightGBM credit-risk model runs, one row per run, "
    "with hyperparameters and evaluation metrics. Always use the query_dataset "
    "tool for any question involving a number. Never estimate or guess a "
    "numeric answer yourself. If the tool returns an error, explain it in "
    "plain language rather than guessing what the user meant."
)


def run_tool(tool_name: str, tool_args: dict) -> str:
    if tool_name == "query_dataset":
        result = query_dataset(
            operation=tool_args.get("operation"),
            column=tool_args.get("column"),
            filter_column=tool_args.get("filter_column"),
            filter_value=tool_args.get("filter_value"),
            group_by=tool_args.get("group_by"),
        )
        return json.dumps(result)
    return json.dumps({"success": False, "error": f"Unknown tool: {tool_name}"})

def call_model_with_retry(messages, tools, max_retries=3):
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=1024,
            )
        except BadRequestError as e:
            if "tool_use_failed" in str(e):
                last_error = e
                time.sleep(1)
                continue
            raise
    raise last_error

def ask(question: str, history: list = None, max_tool_hops: int = 4):
    history = history or []
    if not history:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages = history + [{"role": "user", "content": question}]
    tools = build_tool_schema()

    for _ in range(max_tool_hops):
        response = call_model_with_retry(messages, tools)
        message = response.choices[0].message

        assistant_msg = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_msg["tool_calls"] = message.tool_calls
        messages.append(assistant_msg)

        if not message.tool_calls:
            return message.content, messages

        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result_json = run_tool(tool_call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_json,
            })

    return "Couldn't resolve within the tool-call limit.", messages