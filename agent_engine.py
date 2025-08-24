import pandas as pd
import json
from langchain_ollama import OllamaLLM
from langchain.agents import AgentType, initialize_agent, Tool
from analysis import (predict_delay_for_new_time, find_top_cascading_flights, 
                      optimize_flight_schedule, parse_delay_from_string,
                      run_system_wide_optimization, process_flight_data)

def load_agent_and_precomputed_data():
    """
    This is the main function that loads all data and initializes the agent.
    """
    print("--- LOADING LIGHTWEIGHT AGENT ENGINE ---")
    
    # Load all data from files, no heavy computation on startup
    full_df = pd.read_csv('data/bom_week_flights_synthetic.csv')
    full_df['sched_time_local'] = pd.to_datetime(full_df['sched_time_local'])
    avg_delay_df = pd.read_csv('data/avg_delay_by_hour.csv')
    busiest_df = pd.read_csv('data/busiest_hours.csv')
    best_df = pd.read_csv('data/best_hours.csv')
    
    # Load the pre-computed optimization summary from the JSON file
    try:
        with open('data/optimization_summary.json', 'r') as f:
            optimization_summary = json.load(f)
    except FileNotFoundError:
        print("WARNING: optimization_summary.json not found. Running optimization now...")
        # Fallback: run optimization if the file doesn't exist
        optimization_summary = run_system_wide_optimization(full_df, avg_delay_df)


    print("\nInitializing AI Agent...")
    llm = OllamaLLM(model="llama3", temperature=0)

    tools = [
        Tool(name="Get Busiest Hours",
             func=lambda n="5": busiest_df.head(int("".join(filter(str.isdigit, n))) if n else 5).to_markdown(),
             description="Use for finding the busiest hours. Input can be the number of hours to show."),
        Tool(name="Get Best Hours",
             func=lambda n="5": best_df.head(int("".join(filter(str.isdigit, n))) if n else 5).to_markdown(),
             description="Use for finding the best (least delayed) hours. Input can be the number of hours to show."),
        Tool(name="Predict Schedule Impact",
             func=lambda inputs: predict_delay_for_new_time(
                 flight_id=inputs.split(',')[0].strip(), 
                 new_time_hour=int("".join(filter(str.isdigit, inputs.split(',')[1]))), 
                 full_flight_df=full_df),
             description="Use for a 'what-if' analysis on a SPECIFIC FLIGHT. Input must be the FLIGHT ID and the new hour, separated by a comma. Example: 'SQ279, 14'."),
        Tool(name="Optimize Single Flight Schedule",
             func=lambda flight_id: optimize_flight_schedule(flight_id, full_df, avg_delay_df),
             description="Use this to find a better, less-delayed time for a single, specific flight. The input MUST be the flight ID string (e.g., 'SQ279')."),
        Tool(name="Find Cascade Flights",
             func=lambda x: find_top_cascading_flights(full_df).to_markdown(),
             description="Use this to find the top 10 flights that cause cascading delays. Takes no input."),
        Tool(name="Get System-Wide Optimization Summary",
             func=lambda x: pd.DataFrame.from_dict(optimization_summary, orient='index', columns=['Value']).to_markdown(),
             description="Use this to get the summary of the system-wide optimization, including total delays, savings, and costs. Takes no input.")
    ]

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)

    print("✅ Final, efficient AI Agent is ready.")
    
    # Prepare the pre-computed answers dictionary for the UI's safety net
    precomputed_answers = {
        "What are the 3 busiest hours?": busiest_df.head(3).to_markdown(),
        "What are the best hours to fly?": best_df.to_markdown(),
        "Which flights are the biggest cascade risks?": find_top_cascading_flights(full_df).to_markdown(),
        "Show me the optimization summary": pd.DataFrame.from_dict(optimization_summary, orient='index', columns=['Value']).to_markdown()
    }
    
    return agent, precomputed_answers
