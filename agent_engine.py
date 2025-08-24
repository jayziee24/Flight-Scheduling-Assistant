import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.agents import AgentType, initialize_agent, Tool

# Import our new analysis functions from the other script
from analysis import process_flight_data, predict_delay_for_new_time, find_top_cascading_flights

def load_agent():
    print("--- LOADING FINAL AGENT ENGINE ---")
    
    # 1. Run the analysis and load all necessary data
    full_df, avg_delay_df = process_flight_data()
    busiest_df = pd.read_csv('data/busiest_hours.csv')
    best_df = pd.read_csv('data/best_hours.csv')

    print("\nInitializing AI Agent...")
    llm = OllamaLLM(model="llama3", temperature=0)

    # 2. Define simple, fast Python functions to be used as tools
    def get_busiest_hours(n_str: str = "5") -> str:
        """Returns the top N busiest hours."""
        try:
            n = int("".join(filter(str.isdigit, n_str)))
            return busiest_df.head(n).to_string()
        except:
            return busiest_df.head(5).to_string()

    def get_best_hours(n_str: str = "5") -> str:
        """Returns the top N best (least delayed) hours."""
        try:
            n = int("".join(filter(str.isdigit, n_str)))
            return best_df.head(n).to_string()
        except:
            return best_df.head(5).to_string()

    # 3. Define the list of tools using our new, fast functions
    tools = [
        Tool(
            name="Get Busiest Hours",
            func=get_busiest_hours,
            description="Use this to find the busiest hours with the most flight operations. Input can be the number of hours to show, e.g., '3'."
        ),
        Tool(
            name="Get Best Hours",
            func=get_best_hours,
            description="Use this to find the best (least busy or least delayed) hours to schedule a flight. Input can be the number of hours to show, e.g., '5'."
        ),
        Tool(
            name="Predict Schedule Impact",
            func=lambda hour_str: predict_delay_for_new_time(int("".join(filter(str.isdigit, hour_str))), avg_delay_df),
            description="Use this to predict the delay if a flight is moved to a new hour. The input must be the hour to predict for, e.g., '14'."
        ),
        Tool(
            name="Find Cascade Flights",
            func=lambda empty_str: find_top_cascading_flights(full_df).to_string(),
            description="Use this to find the top 10 flights that cause the most significant knock-on (cascading) delays. Takes no input."
        )
    ]

    # 4. Initialize our single, efficient agent
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    print("✅ Final, efficient AI Agent is ready.")
    return agent