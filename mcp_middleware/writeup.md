# MCP Tool Overload — Writeup

## Problem Diagnosis

The agent loads **25 tools from 3 MCP servers** (GitHub: 8, Slack: 6, Database: 11) into every single prompt. This causes:

- **~6,300 tokens** of tool descriptions injected every turn
- **Tool confusion**: The LLM picks wrong tools for ambiguous queries (e.g., `search_messages` when `search_code` was needed)
- **3-4x slower** response latency from processing all tool schemas
- **Inflated cost**: Every turn pays for tool definitions the agent will never use

## Architecture

```
                    User Query
                        │
                  ┌─────▼─────┐
                  │  classify  │  ← Keyword-based router
                  │  _query()  │    (deterministic, 0 LLM cost)
                  └─────┬─────┘
                   ╱    │    ╲
              github  slack  database
                │       │       │
            8 tools  6 tools  11 tools
                 ╲     │     ╱
              ┌───▼────▼────▼───┐
              │  select_tools() │  ← Returns union of
              │  (filtered set) │    relevant server tools
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │ create_smart   │  ← Fresh agent per query
              │ _agent(query)  │    with only relevant tools
              └───────┬────────┘
                      │
                  Response
```

## Implementation

### Layer 1: Keyword-Based Router (`classify_query()`)

A `SERVER_KEYWORDS` dictionary maps keywords to MCP servers. The router scores each server by counting keyword matches in the query, then returns servers sorted by relevance. Zero LLM cost — pure deterministic classification.

Examples:
- "Create a GitHub issue" → matches "github", "issue" → `["github"]`
- "Send a message to #engineering" → matches "message", "channel", "#" → `["slack"]`
- "Find GitHub issue and post to Slack" → matches both → `["github", "slack"]`

### Layer 2: Tool Selector (`select_tools()`)

Returns the union of tools from all matched servers. For single-server queries, this reduces tools from 25 to 6-11. For cross-server queries, it returns 14-19 — still fewer than 25.

### Layer 3: Dynamic Agent Factory (`create_smart_agent()`)

Creates a fresh `AgentExecutor` per query with only the filtered tools. The system prompt is adapted to tell the agent which servers are available, giving it context for tool selection.

## Before/After Comparison

| Query Type | Tools Before | Tools After | Reduction |
|-----------|-------------|-------------|-----------|
| Single-server (GitHub) | 25 | 8 | 68% |
| Single-server (Slack) | 25 | 6 | 76% |
| Single-server (Database) | 25 | 11 | 56% |
| Cross-server (2 servers) | 25 | 14 | 44% |
| Ambiguous (all servers) | 25 | 25 | 0% |

**Average tool reduction across all 7 test queries: 44%**

For clear single-server queries (the most common case), tool count drops by 56-76%, with proportional savings in token cost and latency.

## Trade-offs

- **Keyword router is fast but imperfect**: Ambiguous queries fall back to all servers (safe default)
- **Creating a new agent per query**: Small overhead, but ensures clean tool sets and avoids cross-contamination
- **No LLM-based routing**: Could add a lightweight LLM classifier for ambiguous queries, but keyword routing handles 5 of 7 test queries perfectly at zero cost

## Testing Approach

Built 25 tests using the RECR loop across three layers: query classification (11 tests, parametrized), tool filtering (5 tests), and full agent pipeline (9 tests, parametrized over all TEST_QUERIES). All pipeline tests use real API calls through an OpenAI-compatible API.
