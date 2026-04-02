"""
Assignment 1: Context Overflow Tool — Tests
============================================
RECR Step R: Define requirements via tests BEFORE implementation.
All tests use real OpenAI API calls.
"""

import json
import pytest

# These will import from solution.py once we build it
from assignment1_context_overflow.solution import (
    lookup_order_info,
    create_agent,
    ORDERS_DB,
    TEST_QUERIES,
)


# =========================================================================
# Layer 1: Tool-level tests (does the tool filter correctly?)
# =========================================================================


class TestToolFiltering:
    """Test that the fixed tool returns only relevant data."""

    def test_tool_filters_by_order_id(self):
        """Tool should return ONLY the matching order, not all 2000."""
        result = lookup_order_info.invoke(
            {"query": "What's the status of order ORD-000001?"}
        )
        assert "ORD-000001" in result
        assert "shipped" in result.lower()
        # Must NOT contain other orders
        assert "ORD-000002" not in result
        assert "ORD-001000" not in result

    def test_tool_filters_by_email(self):
        """Tool should find orders by customer email."""
        result = lookup_order_info.invoke(
            {"query": "Look up the order for customer alice@example.com"}
        )
        assert "alice@example.com" in result or "ORD-000001" in result

    def test_tool_returns_not_found_for_invalid_id(self):
        """Tool should handle non-existent orders gracefully."""
        result = lookup_order_info.invoke(
            {"query": "What's the status of order ORD-999999?"}
        )
        assert "ORD-999999" not in result or "not found" in result.lower() or "no" in result.lower()


# =========================================================================
# Layer 2: Security & data hygiene tests
# =========================================================================


class TestFieldStripping:
    """Test that sensitive/bloat fields are removed."""

    def test_tool_strips_sensitive_fields(self):
        """No card numbers, internal notes, or warehouse logs in output."""
        result = lookup_order_info.invoke(
            {"query": "What's the status of order ORD-000001?"}
        )
        assert "card_last_four" not in result
        assert "4242" not in result  # the actual card digits
        assert "internal_notes" not in result
        assert "warehouse_log" not in result
        assert "payment_method" not in result

    def test_tool_keeps_useful_fields(self):
        """Essential fields should still be present."""
        result = lookup_order_info.invoke(
            {"query": "What items are in order ORD-000001?"}
        )
        assert "Mechanical Keyboard" in result
        assert "USB-C Hub" in result


# =========================================================================
# Layer 3: Safeguard tests
# =========================================================================


class TestSafeguards:
    """Test output size limits and max result caps."""

    def test_output_size_guard(self):
        """Tool output must never exceed 4000 chars."""
        result = lookup_order_info.invoke(
            {"query": "What's the status of order ORD-000001?"}
        )
        assert len(result) < 4000, f"Tool output too large: {len(result)} chars"

    def test_max_results_safeguard(self):
        """Generic query should not return unlimited results."""
        result = lookup_order_info.invoke(
            {"query": "Show me all orders"}
        )
        # Even a broad query should be capped
        assert len(result) < 4000, f"Unbounded query returned {len(result)} chars"


# =========================================================================
# Layer 4: Full agent pipeline tests (burns bootcamp tokens!)
# =========================================================================


class TestAgentPipeline:
    """End-to-end agent tests — real API calls."""

    @pytest.mark.parametrize("query", TEST_QUERIES)
    def test_agent_completes_without_error(self, query):
        """Agent should answer all test queries without crashing."""
        agent = create_agent()
        result = agent.invoke({"input": query})
        assert result["output"], f"Empty response for: {query}"
        assert len(result["output"]) > 10, f"Response too short for: {query}"

    def test_agent_answers_order_status(self):
        """Agent should correctly report ORD-000001 as shipped."""
        agent = create_agent()
        result = agent.invoke(
            {"input": "What's the status of order ORD-000001?"},
        )
        assert "shipped" in result["output"].lower()

    def test_agent_answers_delivery_date(self):
        """Agent should find delivery date for ORD-000001."""
        agent = create_agent()
        result = agent.invoke(
            {"input": "When will order ORD-000001 be delivered?"},
        )
        assert "2026-03-22" in result["output"] or "march" in result["output"].lower()

    def test_agent_answers_items(self):
        """Agent should list items in ORD-000001."""
        agent = create_agent()
        result = agent.invoke(
            {"input": "What items are in order ORD-000001?"},
        )
        output_lower = result["output"].lower()
        assert "keyboard" in output_lower or "mechanical" in output_lower
        assert "hub" in output_lower or "usb" in output_lower
