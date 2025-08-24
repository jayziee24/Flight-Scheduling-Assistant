import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Import our new analysis functions from the other script
from analysis import process_flight_data, predict_delay_for_new_time, find_top_cascading_flights

print("--- EXECUTING THE FINAL POLISHED SCRIPT ---")
print("Running initial analysis...")

# 1. Run the analysis to get the processed dataframes
full_df, avg_delay_df = process_flight_data()

# Load the other summary dataframes
busiest_df = pd.read_csv('data/busiest_hours.csv')
best_df = pd.read_csv('data/best_hours.csv')

print("\nInitializing AI Agents...")
llm = OllamaLLM(model="llama3", temperature=0)

# 2. Create the specialized "Data Analyst" agent
pandas_agent = create_pandas_dataframe_agent(
    llm=llm,
    df={
        "busiest_hours_data": busiest_df,
        "best_hours_data": best_df,
        "average_delay_data": avg_delay_df
    },
    verbose=True,
    allow_dangerous_code=True,
    # Add a clear instruction to provide a final answer
    agent_executor_kwargs={"handle_parsing_errors": True}
)

# 3. Define the list of tools for our main "Manager" agent
tools = [
    Tool(
        name="Flight Data Analysis",
        # --- MODIFICATION 1: Add "Final Answer:" prefix to the output ---
        func=lambda q: "Final Answer: " + pandas_agent.invoke({"input": q})['output'],
        description="""
        Use this tool for any questions about analyzing flight data to find the busiest, best, or freest (least busy) times.
        The input to this tool MUST be the full, original question from the user.
        """
    ),
    Tool(
        name="Predict Schedule Impact",
        # --- MODIFICATION 2: Add "Final Answer:" prefix to the output ---
        func=lambda hour_str: "Final Answer: " + predict_delay_for_new_time(int("".join(filter(str.isdigit, hour_str))), avg_delay_df),
        description="Use this to predict the delay if a flight is moved to a new hour. The input is a single integer representing the hour (e.g., '14')."
    ),
    Tool(
        name="Find Cascade Flights",
        # --- MODIFICATION 3: Add "Final Answer:" prefix to the output ---
        func=lambda empty_str: "Final Answer: \n" + str(find_top_cascading_flights(full_df)),
        description="Use this to find flights that cause major knock-on (cascading) delays. This tool takes no input."
    )
]

# 4. Initialize our main agent (we no longer need the complex suffix)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

print("✅ AI Manager Agent is ready.")

def main():
    print("\n" + "="*50)
    print("FLIGHT SCHEDULING ANALYSIS ASSISTANT")
    print("="*50)
    print("\nI can answer questions about the flight data or run predictive models.")
    print("Examples:")
    print("  - What are the 3 busiest hours?")
    print("  - Predict the delay for a flight moved to 15:00")
    print("  - Find the flights that cause the biggest cascading delays")
    print("\nType 'exit' to quit.")
    
    while True:
        try:
            query = input(">> ")
            if query.lower().strip() in ["exit", "quit", 'q']:
                print("Goodbye!")
                break
            
            if not query.strip():
                continue
            
            result = agent.invoke({"input": query})
            print("\n🤖 Assistant:", result['output'])
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n⚠️ An error occurred: {e}")

if __name__ == "__main__":
    main()