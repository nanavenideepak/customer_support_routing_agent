import os
import traceback
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent import classify_query, extract_entities, route_and_fetch, generate_response

# Page config
st.set_page_config(page_title="Support Routing Agent", layout="wide")
st.title("🛎️ Customer Support Routing Agent")
st.caption("24/7 instant support for your queries")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "loaded_query" not in st.session_state:
    st.session_state["loaded_query"] = ""

# Sidebar — Test Queries Panel
st.sidebar.header("Sample Queries")
st.sidebar.write("Click any query to load it:")

queries = [
    "Where is my order?",
    "I didn't receive the package. I need my refund.",
    "We are looking to place a large order. Please connect us with a poc.",
    "I had to visit your shop. What is the best time?"
]

for q in queries:
    if st.sidebar.button(q, key=f"btn_{q}"):
        st.session_state["loaded_query"] = q
        st.rerun()

st.sidebar.divider()

# Clear conversation button
if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state["messages"] = []
    st.session_state["loaded_query"] = ""
    st.rerun()



# Render conversation history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Re-render expanders for assistant messages that have stored state
        if msg["role"] == "assistant" and "agent_state" in msg:
            state = msg["agent_state"]
            if state.get("escalate"):
                st.error("🚨 This case has been escalated to the support team.")
            with st.expander("🧠 Agent Reasoning"):
                st.write(state["reasoning"])
                st.write("**Classification:**", state["classification"])
                st.write("**Entities:**", state["entities"])
            with st.expander("🔧 Full State (Debug)"):
                display_state = {k: v for k, v in state.items() if k != "knowledge_context"}
                st.json(display_state)

# Show selected query hint
if st.session_state["loaded_query"]:
    st.caption(f"📋 Selected query: \"{st.session_state['loaded_query']}\"")

# Chat input
query = st.chat_input("Type a customer query or select from sidebar...")

# Handle loaded query from sidebar button
if st.session_state["loaded_query"] and not query:
    query = st.session_state["loaded_query"]
    st.session_state["loaded_query"] = ""

if query:
    # Clear loaded_query after use
    st.session_state["loaded_query"] = ""

    # Append user message to history
    st.session_state["messages"].append({"role": "user", "content": query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Build history context from previous messages (exclude current)
    history_text = ""
    for msg in st.session_state["messages"][:-1]:
        role = "Customer" if msg["role"] == "user" else "Agent"
        history_text += f"{role}: {msg['content']}\n"

    # Prepend history to current query for context
    full_query = query
    if history_text:
        full_query = f"""Previous conversation:
{history_text}
Current message: {query}

Use the conversation history above to understand context. If the user is providing information that was requested (like an order ID), treat this message in the context of the full conversation."""

    # Validate API key
    if not os.environ.get("GROQ_API_KEY"):
        with st.chat_message("assistant"):
            st.error("❌ GROQ_API_KEY not found. Please set it in your .env file.")
        st.stop()

    try:
        # Initialize fresh state with full_query
        state = {
            "query": full_query,
            "classification": {},
            "entities": {},
            "missing_entities": [],
            "requires_clarification": False,
            "clarification_question": "",
            "order_data": None,
            "knowledge_context": "",
            "response": "",
            "reasoning": "",
            "escalate": False
        }

        with st.chat_message("assistant"):
            # STATUS BLOCK 1 — Classification
            with st.status("🔍 Classifying query...", expanded=True) as s1:
                state = classify_query(state)
                urgency = state["classification"].get("urgency", "Low")
                if urgency == "High":
                    st.error(f"🔴 Urgency: {urgency}")
                elif urgency == "Medium":
                    st.warning(f"🟡 Urgency: {urgency}")
                else:
                    st.success(f"🟢 Urgency: {urgency}")
                st.write(f"**Department:** {state['classification'].get('department', 'N/A')}")
                st.write(f"**Intent:** {state['classification'].get('intent', 'N/A')}")
                st.caption(state["classification"].get("reasoning", ""))
                s1.update(label="✅ Query classified", state="complete", expanded=False)

            # STATUS BLOCK 2 — Entity Extraction
            with st.status("🧩 Extracting entities...", expanded=True) as s2:
                state = extract_entities(state)
                entities = state["entities"]
                for key, val in entities.items():
                    if val is not None and key not in ("missing_required", "requires_clarification", "clarification_question"):
                        st.write(f"**{key}:** {val}")
                if state["requires_clarification"]:
                    st.warning(f"⚠️ Missing: {state['missing_entities']}")
                s2.update(label="✅ Entities extracted", state="complete", expanded=False)

            # STATUS BLOCK 3 — Routing
            with st.status("🔀 Routing query...", expanded=True) as s3:
                state = route_and_fetch(state)
                intent = state["classification"].get("intent")
                if intent == "irrelevant":
                    st.warning("⚠️ Irrelevant query detected — deflecting")
                elif state["requires_clarification"]:
                    st.warning("⚠️ Clarification required — stopping for user input")
                elif state["order_data"]:
                    st.success("✅ Order data fetched successfully")
                    st.json(state["order_data"])
                elif state["knowledge_context"]:
                    st.success("✅ Company knowledge base loaded")
                if state["escalate"]:
                    st.error("🚨 ESCALATION FLAGGED — routing to human agent")
                s3.update(label="✅ Routing complete", state="complete", expanded=False)

            # STATUS BLOCK 4 — Response Generation
            with st.status("💬 Generating response...", expanded=True) as s4:
                state = generate_response(state)
                s4.update(label="✅ Response generated", state="complete", expanded=False)

            # Final response
            urgency = state["classification"].get("urgency", "Low")
            dept = state["classification"].get("department", "General")
            st.markdown(f"**Urgency:** {urgency} | **Dept:** {dept}")

            if state["escalate"]:
                st.error("🚨 This case has been escalated to the support team.")

            st.markdown(state["response"])

            st.divider()

            with st.expander("🧠 Agent Reasoning"):
                st.write(state["reasoning"])
                st.write("**Classification:**", state["classification"])
                st.write("**Entities:**", state["entities"])

            with st.expander("🔧 Full State (Debug)"):
                display_state = {k: v for k, v in state.items() if k != "knowledge_context"}
                st.json(display_state)

        # Append assistant response to history (store agent_state for re-rendering)
        st.session_state["messages"].append({
            "role": "assistant",
            "content": state["response"],
            "agent_state": state
        })

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"❌ Agent error: {str(e)}")
            st.code(traceback.format_exc())
