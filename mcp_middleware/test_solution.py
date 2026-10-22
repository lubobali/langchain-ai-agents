"""
MCP Tool Overload — Tests
=========================================
RECR Step R: Define requirements via tests BEFORE implementation.
Tests cover: query classification, tool filtering, and full agent pipeline.
"""

import time
import pytest

from mcp_middleware.solution import (
    classify_query,
    select_tools,
    create_smart_agent,
    ALL_TOOLS,
    GITHUB_TOOLS,
    SLACK_TOOLS,
    DATABASE_TOOLS,
    TEST_QUERIES,
)


# =========================================================================
# Layer 1: Query classifier tests (deterministic, no API calls)
# =========================================================================


class TestQueryClassifier:
    """classify_query() should route queries to the correct MCP server(s)."""

    def test_classify_github_query(self):
        servers = classify_query(
            "Create a GitHub issue in myorg/myapp titled 'Fix login bug'"
        )
        assert "github" in servers

    def test_classify_slack_query(self):
        servers = classify_query(
            "Send a message to #engineering channel saying 'Deploy is complete'"
        )
        assert "slack" in servers

    def test_classify_database_query(self):
        servers = classify_query(
            "How many rows are in the users table in the production database?"
        )
        assert "database" in servers

    def test_classify_ambiguous_query(self):
        """Ambiguous queries should return multiple servers."""
        servers = classify_query(
            "Search for anything related to the authentication bug"
        )
        assert len(servers) >= 1  # Should match at least one

    def test_classify_cross_server_github_slack(self):
        """Cross-server query mentioning GitHub and Slack."""
        servers = classify_query(
            "Find the latest GitHub issue about the payment bug and post a summary to #bugs channel on Slack"
        )
        assert "github" in servers
        assert "slack" in servers

    def test_classify_cross_server_database_github(self):
        """Cross-server query mentioning database and GitHub."""
        servers = classify_query(
            "Query the errors table in the database and create a GitHub issue for each critical error"
        )
        assert "database" in servers
        assert "github" in servers

    @pytest.mark.parametrize("query,expected_servers", [
        ("Create a GitHub issue", ["github"]),
        ("Send a Slack message to #general", ["slack"]),
        ("Count rows in the users table in the database", ["database"]),
        ("Find GitHub issues and post to Slack", ["github", "slack"]),
        ("Query errors table and create GitHub issue", ["database", "github"]),
    ])
    def test_router_parametrized(self, query, expected_servers):
        servers = classify_query(query)
        for expected in expected_servers:
            assert expected in servers, f"Expected '{expected}' in {servers} for: {query}"


# =========================================================================
# Layer 2: Tool filtering tests
# =========================================================================


class TestToolFiltering:
    """select_tools() should return only relevant tools, not all 25+."""

    def test_github_query_returns_only_github_tools(self):
        tools = select_tools("Create a GitHub issue in myorg/myapp")
        assert len(tools) == len(GITHUB_TOOLS)
        for t in tools:
            assert "[github]" in t.description.lower()

    def test_slack_query_returns_only_slack_tools(self):
        tools = select_tools("Send a message to #engineering channel")
        assert len(tools) == len(SLACK_TOOLS)
        for t in tools:
            assert "[slack]" in t.description.lower()

    def test_database_query_returns_only_database_tools(self):
        tools = select_tools("How many rows in the users table in the production database?")
        assert len(tools) == len(DATABASE_TOOLS)
        for t in tools:
            assert "[database]" in t.description.lower()

    def test_filtered_tools_always_fewer_than_all(self):
        """For any single-server query, filtered tools < all tools."""
        tools = select_tools("Create a GitHub issue")
        assert len(tools) < len(ALL_TOOLS)

    def test_cross_server_returns_union(self):
        """Cross-server query returns tools from both servers."""
        tools = select_tools(
            "Find GitHub issue and post summary to Slack"
        )
        descriptions = [t.description.lower() for t in tools]
        has_github = any("[github]" in d for d in descriptions)
        has_slack = any("[slack]" in d for d in descriptions)
        assert has_github, "Missing GitHub tools for cross-server query"
        assert has_slack, "Missing Slack tools for cross-server query"


# =========================================================================
# Layer 3: Full agent pipeline tests (spends real API tokens!)
# =========================================================================


class TestAgentPipeline:
    """End-to-end tests with filtered tools through OpenAI API."""

    @pytest.mark.parametrize("query", TEST_QUERIES)
    def test_agent_completes_with_filtered_tools(self, query):
        """Agent with filtered tools should answer all queries."""
        agent, tool_count = create_smart_agent(query)
        result = agent.invoke({"input": query})
        assert result["output"], f"Empty response for: {query}"
        assert len(result["output"]) > 10, f"Response too short for: {query}"
        print(f"\n  Query: {query[:60]}...")
        print(f"  Tools: {tool_count}/{len(ALL_TOOLS)}")

    def test_single_server_query_uses_fewer_tools(self):
        """A clear GitHub query should use only GitHub tools, not all."""
        query = "Create a GitHub issue in myorg/myapp titled 'Fix login bug' with body 'The login page crashes on mobile'"
        agent, tool_count = create_smart_agent(query)
        assert tool_count < len(ALL_TOOLS), f"Expected fewer tools, got {tool_count}"

        result = agent.invoke({"input": query})
        assert result["output"]

    def test_cross_server_query_completes(self):
        """Cross-server query should work with tools from multiple servers."""
        query = "Find the latest GitHub issue about the payment bug and post a summary to #bugs channel on Slack"
        agent, tool_count = create_smart_agent(query)
        result = agent.invoke({"input": query})
        assert result["output"]
        assert tool_count < len(ALL_TOOLS)
