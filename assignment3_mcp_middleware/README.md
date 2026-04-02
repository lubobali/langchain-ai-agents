# Assignment 3: MCP Tool Overload

## Scenario

Your company connects its AI agent to **three MCP (Model Context Protocol) servers**:

1. **GitHub MCP Server** — 18 tools (create_issue, list_repos, search_code, create_pr, ...)
2. **Slack MCP Server** — 15 tools (send_message, list_channels, search_messages, add_reaction, ...)
3. **Database MCP Server** — 20 tools (query_sql, list_tables, describe_table, insert_row, ...)

That's **53 tools** all injected into the system prompt. The result:

- The system prompt is **~8,000 tokens** just for tool descriptions
- The LLM gets confused choosing between similar tools (`search_code` vs `query_sql` for "find the bug")
- Response latency is 3-4x slower because the model processes all tool schemas every turn
- The agent sometimes calls the wrong tool entirely (e.g., Slack's `search_messages` when it should use GitHub's `search_code`)
- Cost is inflated because every single turn pays for 8K tokens of tool definitions

## Your Task

1. **Run the starter code** (`starter.py`) and observe the tool confusion
2. **Design and implement middleware** that sits between the agent and the MCP tools
3. Your middleware should **dynamically select** only the relevant tools for each query
4. Bonus: implement **tool routing** that pre-classifies queries to the right MCP server

## Starter Code

See `starter.py` — an agent with all 53 tools loaded. The tools are simulated
(no real MCP servers needed), but they faithfully reproduce the tool-overload problem.

## Deliverables

1. `solution.py` — Your middleware implementation
2. `writeup.md` — Architecture explanation with a diagram (ASCII or Mermaid) (300-500 words)
3. Before/after comparison of: tool count per query, response accuracy, token usage

---

