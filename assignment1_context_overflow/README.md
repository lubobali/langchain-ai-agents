# Assignment 1: The Context Overflow Tool

## Scenario

You've inherited a customer support agent that uses a tool to look up order information.
The previous developer connected it to the database, but **every time the tool is called,
it dumps the ENTIRE orders table** (thousands of rows) into the LLM context window.

The agent works... sometimes. But it's:
- Extremely slow (processing 100K+ tokens per query)
- Expensive ($0.50+ per simple question)
- Frequently hitting context window limits and crashing
- Hallucinating because the relevant data is buried in noise

## Your Task

1. **Run the starter code** (`starter.py`) and observe the failure mode
2. **Diagnose** exactly why the context window is being overwhelmed
3. **Fix the tool** so it returns only relevant information
4. **Add safeguards** so this class of bug can't happen again

## Starter Code

See `starter.py` — this is the broken agent. Do NOT modify the simulated database
or the user queries. Focus on fixing the tool definition and adding protective middleware.

## Deliverables

1. `solution.py` — Your fixed version
2. `writeup.md` — Explanation of the problem and your approach (300-500 words)
3. Token usage comparison (before vs. after)

---


