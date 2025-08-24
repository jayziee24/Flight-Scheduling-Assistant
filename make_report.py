import os
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_experimental.agents import create_pandas_dataframe_agent

# ========================
# Load CSVs
# ========================
OUTDIR = "."  # Files are in current directory

csv_files = {
    "avg_delay": "avg_delay_by_hour.csv",
    "busiest": "busiest_hours.csv", 
    "best": "best_hours.csv",
    "cascade": "top_cascade.csv",
    "recommendations": "schedule_reco_samples.csv"
}

dfs = {}
for key, filename in csv_files.items():
    path = os.path.join(OUTDIR, filename)
    if os.path.exists(path):
        dfs[key] = pd.read_csv(path)
        print(f"✅ Loaded {filename} ({len(dfs[key])} rows)")
    else:
        print(f"⚠️ File not found: {path}")

if not dfs:
    print("❌ No CSVs loaded! Exiting...")
    exit(1)

# ========================
# Initialize Ollama LLaMA model
# ========================
try:
    llm = OllamaLLM(model="llama3", temperature=0)
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    print("Make sure Ollama is running and llama3 model is installed")
    exit(1)

# ========================
# Create Pandas agent
# ========================
try:
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=list(dfs.values()),  # Changed from 'dataframes' to 'df' parameter
        verbose=True,
        allow_dangerous_code=True
    )
    print("✅ Agent created successfully")
except Exception as e:
    print(f"❌ Error creating agent: {e}")
    exit(1)

# ========================
# Interactive Query Loop  
# ========================
def main():
    print("\n" + "="*50)
    print("FLIGHT SCHEDULING ANALYSIS ASSISTANT")
    print("="*50)
    print("\nLoaded datasets:")
    for key, df in dfs.items():
        print(f"  - {key}: {len(df)} rows, columns: {list(df.columns)}")
    
    print("\nAsk a question about flight schedules (type 'exit' to quit)")
    print("Examples:")
    print("  - What hour has the lowest average delay?")
    print("  - Which routes are most popular?")
    print("  - Show me the best hours for scheduling flights")
    print("\n")
    
    while True:
        try:
            query = input(">> ")
            if query.lower().strip() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
            
            if not query.strip():
                continue
                
            print("🤔 Thinking...")
            result = agent.invoke({"input": query})
            print("\n" + "="*30)
            print("📊 ANSWER:")
            print("="*30)
            print(result.get("output", "No output received"))
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            print("Please try a different question.\n")

if __name__ == "__main__":
    main()