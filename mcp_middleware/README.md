# The MCP Tool Overload

## The problem

An agent connected to three MCP servers (GitHub, Slack and a database) got
**25 tools** in every prompt. It picked the wrong tool, paid for tool
descriptions it didn't need, and got slow.

## The fix

A **keyword router** middleware: it classifies each question and loads only
the tools that question needs. No AI is used for routing, so it costs nothing
and always behaves the same way.

**Result:** 25 tools down to 6–11 per question, and queries that span servers
(find a GitHub issue, post a summary to Slack) still work.

## Files

- `broken_agent.py`: the original agent, kept to show the failure
- `solution.py`: the fixed agent
- `test_solution.py`: the tests
- `writeup.md`: the diagnosis and the numbers
