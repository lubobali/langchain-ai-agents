# Assignment 2: The Infinite Researcher — Writeup

## Problem Diagnosis

The research agent has four compounding failure modes:

1. **Adversarial prompt**: 11 directives pushing infinite research — "NEVER settle", "keep researching until COMPLETE knowledge", "never stop asking questions."
2. **No AgentExecutor guardrails**: No `max_iterations`, no `max_execution_time`.
3. **Tool feedback loop**: `save_notes()` returns "Continue researching"; `web_search()` returns "Related searches"; `read_webpage()` recommends "further reading on at least 3 references."
4. **No token/cost budget**: The agent has zero awareness of spend — no tracking, no limits, no graceful degradation when costs spiral.

Result: a single query costs **$12+** and runs **8+ minutes** before crashing.

## Solution: Four-Layer Defense

### 1. Convergent Prompt
Replaced "never stop" with a structured 5-step methodology with explicit stopping criteria: stop when you have 3+ sources, 3+ searches, or enough to answer. Added budget awareness — if a tool warns about budget limits, stop immediately.

### 2. AgentExecutor Hard Limits
`max_iterations=8` and `max_execution_time=120s` as final backstops.

### 3. Tool Wrapper
`save_research_notes()` wraps `save_notes()`, replaces "Continue researching" with neutral feedback. When budget is low, returns a warning telling the agent to finalize.

### 4. BudgetManager with Per-Step Cost Tracking
A `BudgetManager` class tracks real token usage via a `BudgetTrackingCallback` that hooks into `on_llm_end` (reads `message.usage_metadata` from the API response) and `on_tool_start` (assigns tool costs). It enforces `max_cost` and `max_tokens` limits, sets a `budget_exceeded` flag for graceful degradation, and logs per-step cumulative costs.

Note: `web_search` and `read_webpage` are simulated tools (no real web requests) -- only LLM calls go through the OpenAI-compatible API. Tool costs are assigned estimates ($0.01/search, $0.01/read). LLM token costs are tracked from actual API response metadata.

## Before/After Comparison

| Metric | Before (Broken) | After (Fixed) |
|--------|-----------------|---------------|
| Search calls | 15+ (crash) | 1 |
| Page reads | 15+ (crash) | 3 |
| Wall time | 8+ min (crash) | 13-16s |
| LLM tokens | Unlimited | ~5,000 |
| Total cost | $12+ (crash) | $0.001 |
| Budget tracking | None | Per-step with graceful stop |

## Testing

Built 15 tests: convergence bounds (3), BudgetManager unit tests (6), budget integration with real LLM calls (2), output quality parametrized over TEST_QUERIES (4). All LLM calls go through the OpenAI-compatible API.
