import streamlit as st
from agent_engine import load_agent_and_precomputed_data

st.set_page_config(page_title="Flight Scheduling AI Assistant", page_icon="✈️", layout="wide")

st.title("✈️ Flight Scheduling AI Assistant")
st.write("A conversational AI co-pilot for airport operators, powered by an XGBoost model and a LangChain Router Agent.")

@st.cache_resource
def load_resources():
    return load_agent_and_precomputed_data()

agent, precomputed_answers = load_resources()

# --- Section 1: The "Safety Net" Pre-Computed Insights ---


# Create a dictionary of button labels and their corresponding pre-computed answer keys
buttons = {
    "Show Busiest Hours": "What are the 3 busiest hours?",
    "Show Best Hours": "What are the best hours to fly?",
    "Show Cascade Risks": "Which flights are the biggest cascade risks?",
    "Show Optimization Summary": "Show me the optimization summary"
}

cols = st.columns(len(buttons))
for i, (btn_label, question) in enumerate(buttons.items()):
    if cols[i].button(btn_label):
        st.session_state.last_precomputed_answer = precomputed_answers[question]

if "last_precomputed_answer" in st.session_state:
    st.markdown(st.session_state.last_precomputed_answer)
    st.divider()

# --- Section 2: The Live, Interactive AI Agent ---
st.header("Live AI Assistant")
st.success("Ask any question, or try a 'what-if' prediction or optimization below. (e.g., 'Optimize flight SQ279')")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a dynamic question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI is thinking..."):
            response = agent.invoke({"input": prompt})
            st.markdown(response['output'])
    
    st.session_state.messages.append({"role": "assistant", "content": response['output']})