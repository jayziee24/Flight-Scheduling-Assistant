import streamlit as st
from agent_engine import load_agent

# Set up the page title and icon
st.set_page_config(page_title="Flight Scheduling AI Assistant", page_icon="✈️")

st.title("✈️ Flight Scheduling AI Assistant")
st.write("""
Welcome! This AI assistant helps flight schedulers make smarter decisions. 
Ask a question below to get started.
""")

# Examples for the user to click
st.markdown("""
**Example Questions:**
- `What are the 3 busiest hours?`
- `Predict the delay for a flight moved to 3 AM`
- `Find the flights that cause the biggest cascading delays`
""")

# --- MODIFIED SECTION ---
# We moved the agent loading logic here and wrapped it in a spinner.
# The @st.cache_resource decorator ensures this heavy function runs only once.
@st.cache_resource
def get_agent():
    # This will show a loading message in the UI while the agent is being created
    with st.spinner("Loading AI Assistant... This may take a moment."):
        agent = load_agent()
    return agent

agent = get_agent()
# -------------------------

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        # Show a spinner while the agent is "thinking"
        with st.spinner("The AI is thinking..."):
            # We are calling the agent here
            response = agent.invoke({"input": prompt})
            st.markdown(response['output'])
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response['output']})