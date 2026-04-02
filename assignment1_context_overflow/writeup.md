# Assignment 1: Context Overflow Tool — Writeup

## Problem Diagnosis

The `lookup_order_info()` tool serializes the **entire ORDERS_DB** (2,000 orders) into JSON on every single call — regardless of what the user asked. A simple question like "What's the status of order ORD-000001?" triggers a tool response of **2.28 million characters (~571K tokens)**. This causes:

- **Context window overflow**: The LLM receives 571K tokens of irrelevant data, far exceeding most model limits
- **Extreme cost**: At ~$0.15 per 1M input tokens, each query costs ~$0.09 just for the tool output
- **Hallucination risk**: The relevant order is buried in 1,999 irrelevant records — the model guesses instead of finding
- **Latency**: Processing 571K tokens takes 10+ seconds even when it doesn't crash

## Root Cause

The tool takes a free-text `query` parameter but performs **zero filtering**. It calls `json.dumps(ORDERS_DB, indent=2)` and asks the LLM to "search through the above data" — offloading a database query to a language model. This is the equivalent of sending an entire SQL table to GPT instead of writing a WHERE clause.

## Solution: Three-Layer Defense

### Layer 1: Query Parsing & Targeted Filtering
The tool now extracts identifiers from the query using regex patterns:
- Order ID: `ORD-\d{6}` (e.g., ORD-000001)
- Email: standard email pattern (e.g., alice@example.com)
- Customer ID: `customer_\d{4}` (e.g., customer_0042)

It then filters `ORDERS_DB` to only matching records. Typical result: **1 order instead of 2,000**.

### Layer 2: Field Stripping
Each returned order is sanitized to remove sensitive and bloat fields:
- **Removed**: `card_last_four` (PCI compliance), `payment_method`, `internal_notes` (200+ chars of noise), `warehouse_log` (tracking events, dimensions)
- **Kept**: `order_id`, `status`, `order_date`, `delivery_date`, `items`, `quantities`, `prices`, `shipping_address`

This reduces per-order token count by ~60%.

### Layer 3: Output Size Guard
Hard caps prevent future regressions:
- **Max 10 results** per query (prevents broad queries from returning thousands)
- **Max 4,000 characters** total output (truncates with warning if exceeded)
- **Graceful "not found"** response when no identifiers are detected in the query

## Token Usage Comparison

| Metric | Before (Broken) | After (Fixed) | Reduction |
|--------|-----------------|---------------|-----------|
| Tool output chars | 2,284,338 | 533 | 99.98% |
| Tool output tokens (est) | ~571,084 | ~133 | 99.98% |
| Total tokens/query | ~572,000 | ~1,500 | 99.7% |
| Latency | Crash/timeout | <3s | — |

## Testing Approach

Built 14 tests using the RECR loop (Requirements, Execute, Check, Repeat). Tests cover four layers: tool filtering accuracy, field stripping security, output safeguards, and full agent pipeline correctness. All tests use real API calls through an OpenAI-compatible API -- no mocks.
