# Assignment 2: The Infinite Researcher

## Scenario

Your team built a "deep research" agent that autonomously searches the web,
reads pages, and compiles reports. It uses a **ReAct loop** with search and
note-taking tools.

The problem: **it never stops.** Every answer leads to more questions.
Every source references five more sources. The agent keeps researching
until it hits the token limit, the API rate limit, or your wallet limit — whichever
comes first.

A single query like "What are the pros and cons of microservices?" has racked up
**$12 in API costs** and taken 8 minutes before crashing.

## Your Task

1. **Run the starter code** (`starter.py`) and observe the runaway behavior
2. **Diagnose** why the agent can't stop
3. **Implement stopping conditions** that make the agent converge on an answer
4. **Add cost controls** so a single query can never exceed reasonable limits

## Starter Code

See `starter.py` — the runaway research agent. The tools themselves are fine;
the problem is in the agent loop, the prompt, and the total lack of guardrails.

## Deliverables

1. `solution.py` — Your fixed version
2. `writeup.md` — Explanation of the problem and your approach (300-500 words)
3. Sample output showing the agent completing research within reasonable bounds

---

