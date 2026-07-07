from agent.agent import ask

BANNER = """
Mini Data-Query Agent
Dataset: LightGBM credit-risk experiment runs (data/experiment_runs.csv)
Ask a question, or type 'exit' to quit.
"""


def main():
    print(BANNER)
    history = []
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        answer, history = ask(question, history)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()