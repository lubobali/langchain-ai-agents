"""
Assignment 1: The Context Overflow Tool — SOLUTION
====================================================
Fix: 3-layer defense against context overflow.

Layer 1: Query parsing — extract order_id/email/customer_id, filter ORDERS_DB
Layer 2: Field stripping — remove sensitive/bloat fields (card, internal_notes, warehouse_log)
Layer 3: Output size guard — cap at 10 results, max 4000 chars
"""

import json
import re
import random
import string
from datetime import datetime, timedelta
import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Simulated database — DO NOT MODIFY this section
# ---------------------------------------------------------------------------

def _generate_fake_orders(n=10000):
    """Generate a large fake orders table."""
    statuses = ["shipped", "delivered", "processing", "returned", "cancelled"]
    products = [
        "Wireless Mouse", "USB-C Hub", "Mechanical Keyboard", "Monitor Stand",
        "Webcam HD", "Desk Lamp", "Laptop Sleeve", "Cable Organizer",
        "Noise-Cancelling Headphones", "Portable Charger", "Ergonomic Chair",
        "Standing Desk Mat", "Blue Light Glasses", "Wireless Charger",
        "Smart Power Strip", "Document Scanner", "Label Printer",
    ]
    customers = [f"customer_{i:04d}" for i in range(500)]
    orders = []
    for i in range(n):
        order_date = datetime.now() - timedelta(days=random.randint(1, 365))
        items = random.sample(products, k=random.randint(1, 5))
        orders.append({
            "order_id": f"ORD-{i+1:06d}",
            "customer_id": random.choice(customers),
            "customer_email": f"user{random.randint(1,9999)}@example.com",
            "status": random.choice(statuses),
            "order_date": order_date.isoformat(),
            "delivery_date": (order_date + timedelta(days=random.randint(2, 14))).isoformat(),
            "items": items,
            "quantities": [random.randint(1, 3) for _ in items],
            "prices": [round(random.uniform(9.99, 299.99), 2) for _ in items],
            "shipping_address": {
                "street": f"{random.randint(1,9999)} {random.choice(['Main', 'Oak', 'Elm', 'Pine', 'Cedar'])} St",
                "city": random.choice(["Portland", "Austin", "Denver", "Seattle", "Chicago"]),
                "state": random.choice(["OR", "TX", "CO", "WA", "IL"]),
                "zip": f"{random.randint(10000, 99999)}",
            },
            "payment_method": random.choice(["credit_card", "debit_card", "paypal"]),
            "card_last_four": ''.join(random.choices(string.digits, k=4)),
            "internal_notes": f"Batch processed. Agent: auto-{random.randint(1,50)}. "
                              f"Priority: {''.join(random.choices(string.ascii_lowercase, k=200))}",
            "warehouse_log": f"Picked from aisle {random.randint(1,99)}, "
                             f"shelf {random.choice(string.ascii_uppercase)}{random.randint(1,20)}. "
                             f"Weight: {round(random.uniform(0.5, 25.0), 2)}kg. "
                             f"Dimensions: {random.randint(10,80)}x{random.randint(10,80)}x{random.randint(10,80)}cm. "
                             f"Tracking events: {'|'.join([f'event_{j}' for j in range(random.randint(5,20))])}",
        })
    return orders


ORDERS_DB = _generate_fake_orders(2000)

# Seed a known order so we can test with it
ORDERS_DB[0] = {
    "order_id": "ORD-000001",
    "customer_id": "customer_0042",
    "customer_email": "alice@example.com",
    "status": "shipped",
    "order_date": "2026-03-15T10:30:00",
    "delivery_date": "2026-03-22T14:00:00",
    "items": ["Mechanical Keyboard", "USB-C Hub"],
    "quantities": [1, 2],
    "prices": [149.99, 34.99],
    "shipping_address": {
        "street": "742 Evergreen Terrace",
        "city": "Portland",
        "state": "OR",
        "zip": "97201",
    },
    "payment_method": "credit_card",
    "card_last_four": "4242",
    "internal_notes": "VIP customer, handle with care. Escalation history: ...",
    "warehouse_log": "Picked from aisle 12, shelf B7. Weight: 2.3kg. ...",
}


# ---------------------------------------------------------------------------
# FIXED TOOL — 3-layer defense against context overflow
# ---------------------------------------------------------------------------

# Fields safe to return to the LLM (no PCI data, no internal bloat)
SAFE_FIELDS = {
    "order_id", "customer_id", "customer_email", "status",
    "order_date", "delivery_date", "items", "quantities",
    "prices", "shipping_address",
}

MAX_RESULTS = 10
MAX_OUTPUT_CHARS = 4000


@tool
def lookup_order_info(query: str) -> str:
    """Look up order information from the database.

    Use this tool when a customer asks about their order status,
    delivery date, or order details.

    Args:
        query: The customer's question about their order.
    """
    # --- Layer 1: Parse query to extract identifiers ---
    order_id_match = re.search(r"ORD-\d{6}", query)
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", query)
    customer_id_match = re.search(r"customer_\d{4}", query)

    matching = []

    if order_id_match:
        order_id = order_id_match.group()
        matching = [o for o in ORDERS_DB if o["order_id"] == order_id]
    elif email_match:
        email = email_match.group()
        matching = [o for o in ORDERS_DB if o["customer_email"] == email]
    elif customer_id_match:
        cust_id = customer_id_match.group()
        matching = [o for o in ORDERS_DB if o["customer_id"] == cust_id]
    else:
        return (
            "I couldn't find an order ID, email, or customer ID in your query. "
            "Please provide an order number (e.g., ORD-000001) or customer email."
        )

    if not matching:
        return f"No orders found matching your query: {query}"

    # --- Layer 2: Cap results ---
    if len(matching) > MAX_RESULTS:
        total = len(matching)
        matching = matching[:MAX_RESULTS]
        truncation_note = f"\n\n(Showing {MAX_RESULTS} of {total} matching orders)"
    else:
        truncation_note = ""

    # --- Layer 3: Strip sensitive/bloat fields ---
    sanitized = []
    for order in matching:
        clean = {k: v for k, v in order.items() if k in SAFE_FIELDS}
        sanitized.append(clean)

    result = json.dumps(sanitized, indent=2) + truncation_note

    # --- Output size guard ---
    if len(result) > MAX_OUTPUT_CHARS:
        result = result[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"

    return result


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

def create_agent():
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        # base_url=os.getenv("OPENAI_BASE_URL"),  # Optional: set for custom proxy
        streaming=True,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful customer support agent. Use the lookup_order_info "
         "tool to find information about customer orders. Be concise and helpful."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [lookup_order_info], prompt)
    return AgentExecutor(agent=agent, tools=[lookup_order_info], verbose=True)


# ---------------------------------------------------------------------------
# Test queries — DO NOT MODIFY
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    "What's the status of order ORD-000001?",
    "When will order ORD-000001 be delivered?",
    "What items are in order ORD-000001?",
    "Can you look up the order for customer alice@example.com?",
]


def main():
    agent = create_agent()

    for query in TEST_QUERIES:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")
        try:
            result = agent.invoke({"input": query})
            print(f"\nRESPONSE: {result['output']}")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
