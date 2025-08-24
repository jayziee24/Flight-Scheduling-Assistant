import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import initialize_agent, AgentType


OUTDIR = "outputs"

def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)

def load_dataframes():
    ensure_outdir()
    dfs = {}
    for fname in os.listdir(OUTDIR):
        if fname.endswith(".csv"):
            fpath = os.path.join(OUTDIR, fname)
            dfs[fname] = pd.read_csv(fpath)
    return dfs


def main():
    dfs = load_dataframes()
    if not dfs:
        print("⚠️ No CSVs found in outputs/. Please run the report generator first.")
        return

    print(f"✅ Loaded {len(dfs)} dataframes: {list(dfs.keys())}")
    for name, df in dfs.items():
        print(f"📊 {name}: {df.shape[0]} rows, {df.shape[1]} columns")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # tools: dataframe inspection + Python execution
    tools = [
        PythonREPLTool(),  # gives agent a real Python REPL
    ]

    # create agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    # inject dataframes into globals so agent can use them
    global_vars = {f"df{i+1}": df for i, df in enumerate(dfs.values())}
    globals().update(global_vars)

    print("\n💡 Ask questions about the data (type 'exit' to quit)\n")

    while True:
        q = input("❓ Your question: ")
        if q.lower() in ["exit", "quit", "q"]:
            break
        try:
            result = agent.run(q)
            print("🤖", result)
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
