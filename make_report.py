import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Import our new analysis functions from the other script
from analysis import process_flight_data, predict_delay_for_new_time, find_top_cascading_flights

print("--- EXECUTING THE FINAL VERSION OF MAKE_REPORT.PY SCRIPT ---")
print("Running initial analysis to prepare data for the agent...")

# 1. Run the analysis to get the processed dataframes
full_df, avg_delay_df = process_flight_data()

# Load the other summary dataframes
busiest_df = pd.read_csv('data/busiest_hours.csv')
best_df = pd.read_csv('data/best_hours.csv')

print("\nInitializing AI Agents...")
# Initialize the local Llama3 model
llm = OllamaLLM(model="llama3", temperature=0)

# 2. Create the specialized "Data Analyst" agent
# This agent is given the dataframes and knows how to query them.
pandas_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=[busiest_df, best_df, avg_delay_df], # Pass a list of dataframes
    verbose=True,
    allow_dangerous_code=True
)

# 3. Define the list of tools for our main "Manager" agent
tools = [
    # Give the Manager agent access to the Data Analyst agent as a tool
    Tool(
        name="Flight Data Analysis",
        func=pandas_agent.invoke, # Use the specialized agent's invoke method
        description="""
        Use this tool for any general questions about flight data, like finding the busiest or best hours, 
        calculating averages, or counting flights from the provided dataframes.
        Example: 'What are the 3 busiest hours?'
        """
    ),
    Tool(
        name="Predict Schedule Impact",
        # NEW, MORE ROBUST LINE
        func=lambda hour_str: predict_delay_for_new_time(int("".join(filter(str.isdigit, hour_str))), avg_delay_df),
        description="Use this to predict the delay if a flight is moved to a new hour. The input is a single integer representing the hour (e.g., '14')."
    ),
    Tool(
        name="Find Cascade Flights",
        func=lambda empty_str: str(find_top_cascading_flights(full_df)), # Convert output to string
        description="Use this to find flights that cause major knock-on (cascading) delays. This tool takes no input."
    )
]

# 4. Initialize our main "Manager" agent with the complete tool list
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
            if query.lower().strip() in ["exit", "quit", "q"]:
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