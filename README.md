# ✈️ Flight Scheduling AI Assistant - Honeywell Hackathon

A conversational AI assistant designed to help airport operators make smarter, data-driven scheduling decisions to de-congest flight traffic.

## Problem Statement

Due to capacity limitations and heavy passenger load, flight operations at busy airports are a scheduling nightmare. A single delay can cause a costly chain reaction. Operators need intelligent tools to find efficiencies and mitigate risks within the system's constraints.

## Our Solution

We built an AI-powered assistant that provides a simple, conversational interface to a powerful backend. An operator can simply ask questions in natural language to:

- Analyze historical flight data for patterns.
- Predict the impact of schedule changes using a "what-if" model.
- Identify high-risk flights that are likely to cause cascading delays.

## Key Features

- **Conversational Interface:** Ask questions in plain English, no code required.
- **Data Analysis:** Instantly find the busiest, quietest, and most delayed times at the airport.
- **Predictive Modeling:** Simulate the effect of moving a flight to a new time slot to see the expected delay.
- **Risk Assessment:** Proactively identify which specific flights pose the biggest threat to the day's schedule.

## Technical Architecture

This project is built as a **Router Agent** using Python and LangChain. The main agent understands the user's intent and delegates tasks to specialized tools:

1.  A **Data Analyst Agent** (powered by `pandas-agent`) for EDA queries.
2.  A custom **Prediction Tool** for "what-if" scenarios.
3.  A custom **Cascade Analysis Tool** for risk assessment.

This modular architecture is efficient, scalable, and demonstrates a modern approach to building AI systems.

**Tech Stack:**

- **AI:** Python, LangChain, Ollama with Llama3
- **Data Analysis:** Pandas
- **UI:** Streamlit
- **Visualization:** Matplotlib, Seaborn

## How to Run the Project

1.  **Setup:**
    ```bash
    # Create and activate a virtual environment
    python -m venv .venv
    source .venv/bin/activate
    # Install dependencies
    pip install -r requirements.txt
    # Make sure you have Ollama running with the llama3 model
    ```
2.  **Generate Analysis Files:**
    ```bash
    python analysis.py
    ```
3.  **Run the Web App:**
    ```bash
    streamlit run app.py
    ```

_(Note: Don't forget to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your terminal!)_
