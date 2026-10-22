# The Context Overflow Agent

## The problem

A customer support agent looks up orders with one tool. That tool returned
the **entire orders table** (2,000 orders) on every call, so a question like
"What's the status of order ORD-000001?" sent about **571K tokens** to the
model: slow, expensive, crashing on context limits, and guessing because the
answer was buried in noise.

## The fix

Three layers, so this class of bug can't come back:

1. **Query parsing**: the tool reads what the user asked and fetches only that.
2. **Field stripping**: only the fields the answer needs reach the model
   (card numbers never do).
3. **Output size guard**: a hard cap on what any tool may return.

**Result:** 571K tokens down to 133 per question (99.98% less).

## Files

- `broken_agent.py`: the original agent, kept to show the failure
- `solution.py`: the fixed agent
- `test_solution.py`: the tests
- `writeup.md`: the diagnosis and the numbers
