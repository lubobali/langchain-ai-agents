"""
Assignment 3: MCP Tool Overload — SOLUTION
============================================
Fix: Middleware that dynamically selects relevant tools per query.

Architecture:
  User Query → classify_query() → select_tools() → create_smart_agent()

  classify_query(): Keyword-based router maps query to MCP server(s)
  select_tools(): Returns only tools from relevant server(s)
  create_smart_agent(): Builds agent with filtered tool set per query
"""

import json
import time
from typing import Any
import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Simulated MCP Server tool definitions — DO NOT MODIFY these definitions
# (You CAN change how they're loaded into the agent)
# ---------------------------------------------------------------------------

def _make_tool(name: str, description: str, server: str, params: dict[str, str]):
    """Create a simulated MCP tool."""
    fields = {}
    for param_name, param_desc in params.items():
        fields[param_name] = (str, Field(description=param_desc))

    InputModel = type(f"{name}_Input", (BaseModel,), {"__annotations__": {k: str for k in params}, **{k: Field(description=v) for k, v in params.items()}})

    def tool_func(**kwargs) -> str:
        return json.dumps({
            "server": server,
            "tool": name,
            "params": kwargs,
            "result": f"[Simulated {server} response for {name}]",
            "status": "success",
        }, indent=2)

    return StructuredTool.from_function(
        func=tool_func,
        name=name,
        description=f"[{server}] {description}",
        args_schema=InputModel,
    )


# --- GitHub MCP Server (8 tools) ---
GITHUB_TOOLS = [
    _make_tool("create_issue", "Create a new GitHub issue in a repository", "github",
               {"repo": "Repository name (owner/repo)", "title": "Issue title", "body": "Issue body/description"}),
    _make_tool("list_issues", "List issues in a GitHub repository", "github",
               {"repo": "Repository name (owner/repo)", "state": "Filter by state: open, closed, all"}),
    _make_tool("get_issue", "Get details of a specific GitHub issue", "github",
               {"repo": "Repository name (owner/repo)", "issue_number": "Issue number"}),
    _make_tool("create_pull_request", "Create a new pull request", "github",
               {"repo": "Repository name", "title": "PR title", "head": "Source branch", "base": "Target branch"}),
    _make_tool("list_pull_requests", "List pull requests in a repository", "github",
               {"repo": "Repository name", "state": "Filter: open, closed, all"}),
    _make_tool("merge_pull_request", "Merge a pull request", "github",
               {"repo": "Repository name", "pr_number": "PR number", "merge_method": "merge, squash, or rebase"}),
    _make_tool("search_code", "Search for code across GitHub repositories", "github",
               {"query": "Search query", "repo": "Optional: limit to specific repo"}),
    _make_tool("search_issues", "Search issues and PRs across GitHub", "github",
               {"query": "Search query", "repo": "Optional: limit to specific repo"})
]

# --- Slack MCP Server (6 tools) ---
SLACK_TOOLS = [
    _make_tool("send_message", "Send a message to a Slack channel", "slack",
               {"channel": "Channel name or ID", "text": "Message text"}),
    _make_tool("send_dm", "Send a direct message to a user", "slack",
               {"user": "User ID or username", "text": "Message text"}),
    _make_tool("list_channels", "List all Slack channels in the workspace", "slack",
               {"type": "public, private, or all"}),
    _make_tool("search_messages", "Search for messages across Slack", "slack",
               {"query": "Search query", "channel": "Optional: limit to channel"}),
    _make_tool("get_channel_history", "Get recent messages from a channel", "slack",
               {"channel": "Channel name or ID", "limit": "Number of messages"}),
    _make_tool("add_reaction", "Add an emoji reaction to a message", "slack",
               {"channel": "Channel ID", "timestamp": "Message timestamp", "emoji": "Emoji name"}),
]

# --- Database MCP Server (11 tools) ---
DATABASE_TOOLS = [
    _make_tool("query_sql", "Execute a read-only SQL query", "database",
               {"query": "SQL SELECT query", "database": "Database name"}),
    _make_tool("list_tables", "List all tables in a database", "database",
               {"database": "Database name"}),
    _make_tool("describe_table", "Get schema/columns of a table", "database",
               {"database": "Database name", "table": "Table name"}),
    _make_tool("insert_row", "Insert a new row into a table", "database",
               {"database": "Database name", "table": "Table name", "data": "JSON object of column:value pairs"}),
    _make_tool("update_rows", "Update rows matching a condition", "database",
               {"database": "Database name", "table": "Table name", "set_values": "JSON of updates", "where": "WHERE clause"}),
    _make_tool("delete_rows", "Delete rows matching a condition", "database",
               {"database": "Database name", "table": "Table name", "where": "WHERE clause"}),
    _make_tool("count_rows", "Count rows in a table with optional filter", "database",
               {"database": "Database name", "table": "Table name", "where": "Optional WHERE clause"}),
    _make_tool("get_table_stats", "Get statistics about a table (row count, size, indexes)", "database",
               {"database": "Database name", "table": "Table name"}),
    _make_tool("list_databases", "List all available databases", "database",
               {}),
    _make_tool("create_table", "Create a new table with specified schema", "database",
               {"database": "Database name", "table": "Table name", "schema": "JSON schema definition"}),
    _make_tool("drop_table", "Drop/delete a table (DANGEROUS)", "database",
               {"database": "Database name", "table": "Table name", "confirm": "Type 'yes' to confirm"})
]


ALL_TOOLS = GITHUB_TOOLS + SLACK_TOOLS + DATABASE_TOOLS


# ---------------------------------------------------------------------------
# MIDDLEWARE: Query classifier + tool selector
# ---------------------------------------------------------------------------

SERVER_KEYWORDS = {
    "github": [
        "github", "issue", "pull request", "pr ", "repo", "repository",
        "branch", "merge", "commit", "code", "create_issue",
        "search_code", "search_issues",
    ],
    "slack": [
        "slack", "message", "channel", "dm", "direct message",
        "reaction", "emoji", "#engineering", "#bugs", "#",
        "send_message", "send_dm",
    ],
    "database": [
        "database", "sql", "table", "query", "row", "column",
        "schema", "insert", "update", "delete", "count",
        "db ", "production database",
    ],
}

SERVER_TOOLS = {
    "github": GITHUB_TOOLS,
    "slack": SLACK_TOOLS,
    "database": DATABASE_TOOLS,
}


def classify_query(query: str) -> list[str]:
    """Classify a query to relevant MCP server(s) based on keywords.

    Returns a list of server names sorted by relevance score (highest first).
    Falls back to all servers if no keywords match.
    """
    query_lower = query.lower()
    scores = {}

    for server, keywords in SERVER_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[server] = score

    if not scores:
        return ["github", "slack", "database"]

    return sorted(scores.keys(), key=lambda s: scores[s], reverse=True)


def select_tools(query: str) -> list:
    """Select only the relevant tools for a query based on classification."""
    servers = classify_query(query)
    selected = []
    for server in servers:
        selected.extend(SERVER_TOOLS[server])
    return selected


# ---------------------------------------------------------------------------
# SMART AGENT: Dynamic tool set per query
# ---------------------------------------------------------------------------

def create_smart_agent(query: str):
    """Create an agent with only the tools relevant to this query.

    Returns:
        tuple: (AgentExecutor, tool_count) for logging.
    """
    tools = select_tools(query)
    servers_used = classify_query(query)

    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        # base_url=os.getenv("OPENAI_BASE_URL"),  # Optional: set for custom proxy
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are a helpful assistant with access to tools from: {', '.join(servers_used)}. "
         f"Use the appropriate tool to fulfill the user's request. "
         f"If the task requires multiple steps across different tools, execute them in order."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )
    return executor, len(tools)


# ---------------------------------------------------------------------------
# Test queries — DO NOT MODIFY
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    # Clear single-server queries
    "Create a GitHub issue in myorg/myapp titled 'Fix login bug' with body 'The login page crashes on mobile'",
    "Send a message to #engineering channel saying 'Deploy is complete'",
    "How many rows are in the users table in the production database?",

    # Ambiguous queries (which server?)
    "Search for anything related to the authentication bug",
    "Find all messages or issues about the deployment failure last Friday",

    # Cross-server queries
    "Find the latest GitHub issue about the payment bug and post a summary to #bugs channel on Slack",
    "Query the errors table in the database and create a GitHub issue for each critical error",
]


def main():
    print(f"\n📊 Total tools available: {len(ALL_TOOLS)}")
    print(f"   GitHub:   {len(GITHUB_TOOLS)} tools")
    print(f"   Slack:    {len(SLACK_TOOLS)} tools")
    print(f"   Database: {len(DATABASE_TOOLS)} tools")
    print()

    for i, query in enumerate(TEST_QUERIES):
        print(f"\n{'='*60}")
        print(f"QUERY {i+1}/{len(TEST_QUERIES)}: {query}")
        print(f"{'='*60}")

        servers = classify_query(query)
        agent, tool_count = create_smart_agent(query)

        print(f"   🎯 Routed to: {', '.join(servers)}")
        print(f"   🔧 Tools: {tool_count}/{len(ALL_TOOLS)} (was {len(ALL_TOOLS)} every time)")

        start_time = time.time()
        try:
            result = agent.invoke({"input": query})
            print(f"\n   RESPONSE: {result['output'][:300]}...")
        except Exception as e:
            print(f"\n   ❌ ERROR: {e}")

        elapsed = time.time() - start_time
        print(f"   ⏱ Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
