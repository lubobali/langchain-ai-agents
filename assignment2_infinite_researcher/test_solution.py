"""
Assignment 2: The Infinite Researcher — Tests
===============================================
RECR Step R: Define requirements via tests BEFORE implementation.
All LLM calls go through the OpenAI API.
Tool calls (web_search, read_webpage) are simulated — no real web requests.
"""

import time
import pytest

from assignment2_infinite_researcher import solution
from assignment2_infinite_researcher.solution import BudgetManager


class TestAgentConvergence:
    """The fixed agent must stop within bounded iterations, time, and cost."""

    def test_agent_terminates_within_iterations(self):
        agent = solution.create_research_agent()
        result = agent.invoke(
            {"input": "What are the pros and cons of microservices architecture?"},
        )
        assert result["output"], "Agent produced no output"
        assert len(result["output"]) > 50, "Output too short to be a real report"

    def test_agent_completes_within_time(self):
        agent = solution.create_research_agent()
        start = time.time()
        result = agent.invoke(
            {"input": "What are the pros and cons of microservices architecture?"},
        )
        elapsed = time.time() - start
        assert elapsed < 60, f"Agent took {elapsed:.1f}s — too slow"
        assert result["output"], "Agent produced no output"

    def test_search_calls_bounded(self):
        solution._SEARCH_CALL_COUNT = 0
        solution._READ_CALL_COUNT = 0
        agent = solution.create_research_agent()
        agent.invoke(
            {"input": "What are the pros and cons of microservices architecture?"},
        )
        assert solution._SEARCH_CALL_COUNT <= 5
        assert solution._READ_CALL_COUNT <= 8


class TestBudgetManager:
    """BudgetManager must track costs and enforce limits."""

    def test_budget_tracks_llm_usage(self):
        budget = BudgetManager(max_cost=1.00, max_tokens=50_000)
        budget.record_llm_usage(1000, 500, "test_call")
        assert budget.total_input_tokens == 1000
        assert budget.total_output_tokens == 500
        assert budget.total_tokens == 1500
        assert budget.llm_cost > 0

    def test_budget_tracks_tool_costs(self):
        budget = BudgetManager(max_cost=1.00, max_tokens=50_000)
        budget.record_tool_usage("web_search")
        budget.record_tool_usage("read_webpage")
        assert budget.total_tool_cost == 0.02

    def test_budget_exceeded_by_cost(self):
        budget = BudgetManager(max_cost=0.05, max_tokens=50_000)
        budget.record_llm_usage(100_000, 50_000, "big_call")
        assert budget.budget_exceeded is True

    def test_budget_exceeded_by_tokens(self):
        budget = BudgetManager(max_cost=100.0, max_tokens=1000)
        budget.record_llm_usage(800, 300, "big_call")
        assert budget.budget_exceeded is True

    def test_budget_step_log(self):
        budget = BudgetManager(max_cost=1.00, max_tokens=50_000)
        budget.record_llm_usage(500, 100, "step1")
        budget.record_tool_usage("web_search")
        assert len(budget.step_log) == 2
        assert budget.step_log[1]["cumulative_cost"] > budget.step_log[0]["cumulative_cost"]

    def test_budget_report(self):
        budget = BudgetManager(max_cost=1.00, max_tokens=50_000)
        budget.record_llm_usage(1000, 500, "call1")
        budget.record_tool_usage("web_search")
        report = budget.report()
        assert "LLM input tokens" in report
        assert "Tool cost" in report
        assert "Total cost" in report
        assert "Step-by-step" in report


class TestBudgetIntegration:
    """Budget tracking works end-to-end with real LLM calls."""

    def test_agent_tracks_cost_during_run(self):
        budget = BudgetManager(max_cost=1.00, max_tokens=50_000)
        agent = solution.create_research_agent(budget=budget)
        result = agent.invoke(
            {"input": "What are the pros and cons of microservices architecture?"},
        )
        assert result["output"]
        assert budget.total_tokens > 0, "No tokens tracked"
        assert budget.total_cost > 0, "No cost tracked"
        assert len(budget.step_log) > 0, "No step log entries"
        assert budget.total_cost < 1.00, f"Over budget: ${budget.total_cost:.4f}"
        print(f"\nBudget Report:\n{budget.report()}")

    def test_agent_stays_under_budget(self):
        budget = BudgetManager(max_cost=0.50, max_tokens=30_000)
        agent = solution.create_research_agent(budget=budget)
        result = agent.invoke(
            {"input": "Summarize the current state of quantum computing in 2026."},
        )
        assert result["output"]
        assert budget.total_cost < 1.00, f"Way over budget: ${budget.total_cost:.4f}"
        print(f"\nBudget Report:\n{budget.report()}")


class TestOutputQuality:
    """The agent should produce coherent, topic-relevant output."""

    @pytest.mark.parametrize("query", solution.TEST_QUERIES)
    def test_agent_produces_relevant_output(self, query):
        solution._SEARCH_CALL_COUNT = 0
        solution._READ_CALL_COUNT = 0
        agent = solution.create_research_agent()
        result = agent.invoke({"input": query})
        output = result["output"].lower()
        assert len(output) > 100, f"Output too short for: {query}"
        if "microservices" in query.lower():
            assert any(w in output for w in ["microservice", "service", "architecture", "scalab"])
        elif "quantum" in query.lower():
            assert any(w in output for w in ["quantum", "computing", "qubit", "superposition"])

    @pytest.mark.parametrize("query", solution.TEST_QUERIES)
    def test_pipeline_bounded_with_cost(self, query):
        solution._SEARCH_CALL_COUNT = 0
        solution._READ_CALL_COUNT = 0
        budget = BudgetManager(max_cost=1.00, max_tokens=50_000)
        agent = solution.create_research_agent(budget=budget)
        start = time.time()
        result = agent.invoke({"input": query})
        elapsed = time.time() - start
        assert result["output"], f"No output for: {query}"
        assert elapsed < 60
        assert solution._SEARCH_CALL_COUNT <= 5
        assert solution._READ_CALL_COUNT <= 8
        assert budget.total_cost < 1.00
        print(f"\n  {query[:50]}... | Cost: ${budget.total_cost:.4f} | Tokens: {budget.total_tokens:,}")
