# The Infinite Researcher

## The problem

A "deep research" agent searches, reads and takes notes in a ReAct loop. It
never stops: every answer leads to more questions, every source to five more.
One question burned **$12 and 8 minutes** before it crashed.

## The fix

- **BudgetManager**: tracks the real token cost of every step and stops in time.
- **A convergent prompt** with clear stopping criteria.
- **Hard iteration limits.**
- **Graceful degradation**: when the budget runs out, the agent writes its
  answer instead of crashing.

**Result:** $12 and 8 minutes down to $0.001 and 15 seconds per question.

## Files

- `broken_agent.py`: the original agent, kept to show the failure
- `solution.py`: the fixed agent
- `test_solution.py`: the tests
- `writeup.md`: the diagnosis and the numbers
