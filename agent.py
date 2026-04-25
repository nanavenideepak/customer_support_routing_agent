# NOTE: Company knowledge is loaded as plain text context for this demo.
# In production, this would use a vector database (e.g. pgvector +
# embeddings) for scalable semantic search over large document collections.

import os
import re
import json
from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from prompts import CLASSIFIER_PROMPT, ENTITY_EXTRACTOR_PROMPT, RESPONSE_GENERATOR_PROMPT
from data.orders import ORDER_DATABASE

load_dotenv()

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

# Load all knowledge base files
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    with open(os.path.join(data_dir, "company_info.md"), "r", encoding="utf-8") as f:
        company_info = f.read()
    with open(os.path.join(data_dir, "policies.md"), "r", encoding="utf-8") as f:
        policies = f.read()
    with open(os.path.join(data_dir, "sales.md"), "r", encoding="utf-8") as f:
        sales = f.read()

    KNOWLEDGE_BASE = f"{company_info}\n\n---\n\n{policies}\n\n---\n\n{sales}"
except FileNotFoundError as e:
    print(f"WARNING: Knowledge base file not found: {e}")
    KNOWLEDGE_BASE = ""


# Agent State
class SupportAgentState(TypedDict):
    query: str
    classification: dict
    entities: dict
    missing_entities: list[str]
    requires_clarification: bool
    clarification_question: str
    order_data: dict | None
    knowledge_context: str
    response: str
    reasoning: str
    escalate: bool


def _parse_json_response(text: str) -> dict | None:
    """Strip markdown fences and parse JSON from LLM response."""
    # Remove markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# STEP 1 — Classify the query
def classify_query(state: SupportAgentState) -> SupportAgentState:
    """Classifies the query by urgency, department, and intent."""
    prompt = CLASSIFIER_PROMPT.format(query=state["query"])
    response = llm.invoke([HumanMessage(content=prompt)])

    parsed = _parse_json_response(response.content)
    if parsed is None:
        parsed = {
            "urgency": "Low",
            "department": "General",
            "intent": "faq",
            "reasoning": "Could not classify"
        }

    state["classification"] = parsed
    return state


# STEP 2 — Extract entities
def extract_entities(state: SupportAgentState) -> SupportAgentState:
    """Extracts key entities and identifies what is missing."""
    prompt = ENTITY_EXTRACTOR_PROMPT.format(query=state["query"])
    response = llm.invoke([HumanMessage(content=prompt)])

    parsed = _parse_json_response(response.content)
    if parsed is None:
        parsed = {
            "order_id": None,
            "product_name": None,
            "date": None,
            "customer_name": None,
            "missing_required": [],
            "requires_clarification": False,
            "clarification_question": None
        }

    state["entities"] = parsed
    state["missing_entities"] = parsed.get("missing_required", [])
    state["requires_clarification"] = parsed.get("requires_clarification", False)
    state["clarification_question"] = parsed.get("clarification_question", "")
    return state


# STEP 3 — Route and fetch data
def route_and_fetch(state: SupportAgentState) -> SupportAgentState:
    """Decision router — the agentic branching step.
    Decides what data to fetch based on classification and entities."""
    intent = state["classification"].get("intent")

    # BRANCH 1 — Irrelevant query
    if intent == "irrelevant":
        state["response"] = (
            "That doesn't seem related to our products or services. "
            "I'm here to help with orders, refunds, product queries, "
            "or store information. Is there anything I can help you with?"
        )
        state["reasoning"] = (
            "Query classified as irrelevant — not related to business domain. "
            "Deflecting gracefully."
        )
        state["escalate"] = False
        return state

    # BRANCH 2 — Requires clarification
    elif state["requires_clarification"]:
        state["response"] = state["clarification_question"]
        state["reasoning"] = (
            f"Required entity missing: {state['missing_entities']}. "
            "Asking user for clarification."
        )
        state["escalate"] = False
        return state

    # BRANCH 3 — Order tracking (has order ID)
    elif intent == "track_order" and state["entities"].get("order_id"):
        order_id = state["entities"]["order_id"]
        order = ORDER_DATABASE.get(order_id)
        if order:
            state["order_data"] = order
        else:
            state["order_data"] = {"error": f"Order {order_id} not found in our system"}
        state["knowledge_context"] = ""

    # BRANCH 4 — Refund request
    elif intent == "refund":
        state["knowledge_context"] = KNOWLEDGE_BASE
        state["escalate"] = True  # refunds always escalate
        # If entity has order_id, also fetch order data
        if state["entities"].get("order_id"):
            order_id = state["entities"]["order_id"]
            order = ORDER_DATABASE.get(order_id)
            if order:
                state["order_data"] = order
            else:
                state["order_data"] = {"error": f"Order {order_id} not found in our system"}

    # BRANCH 5 — Sales inquiry
    elif intent == "sales_inquiry":
        state["knowledge_context"] = KNOWLEDGE_BASE
        state["escalate"] = False

    # BRANCH 6 — FAQ / General
    else:
        state["knowledge_context"] = KNOWLEDGE_BASE
        state["escalate"] = False

    return state


# STEP 4 — Generate response
def generate_response(state: SupportAgentState) -> SupportAgentState:
    """Generates the final customer-facing response using all gathered context.
    Only runs if Step 3 did not return early."""
    # If response is already set (early return from Step 3), skip LLM call
    if state["response"]:
        return state

    # Build context
    classification = state["classification"]
    entities = state["entities"]
    order_data = state["order_data"]
    knowledge_context = state["knowledge_context"]
    escalate = state["escalate"]

    # Format order data as readable text
    if order_data:
        order_data_str = json.dumps(order_data, indent=2)
    else:
        order_data_str = "None"

    prompt = RESPONSE_GENERATOR_PROMPT.format(
        query=state["query"],
        classification=json.dumps(classification, indent=2),
        entities=json.dumps(entities, indent=2),
        order_data=order_data_str,
        knowledge_context=knowledge_context if knowledge_context else "No additional context available.",
        escalate=escalate
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    state["response"] = response.content

    intent = classification.get("intent", "unknown")
    urgency = classification.get("urgency", "unknown")
    dept = classification.get("department", "unknown")

    state["reasoning"] = (
        f"Intent: {intent} | Urgency: {urgency} | "
        f"Department: {dept} | Escalate: {escalate}"
    )

    return state


# ORCHESTRATOR
def run_support_agent(query: str) -> SupportAgentState:
    state: SupportAgentState = {
        "query": query,
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
    state = classify_query(state)
    state = extract_entities(state)
    state = route_and_fetch(state)
    state = generate_response(state)
    return state
