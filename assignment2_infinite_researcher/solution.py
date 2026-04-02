"""
Assignment 2: The Infinite Researcher — SOLUTION
==================================================
Fix: Convergent prompt + hard limits + BudgetManager with per-step cost tracking.

Problem: The original agent has 4 compounding failure modes:
1. System prompt says "never stop asking questions"
2. No max_iterations on AgentExecutor
3. save_notes tool says "Continue researching"
4. No token/cost budget tracking — agent has no awareness of spend

Solution:
1. Rewrite prompt with explicit stopping criteria (5-step methodology)
2. AgentExecutor: max_iterations=8, max_execution_time=120s
3. Wrap save_notes to remove "Continue researching" nudge
4. BudgetManager with LLM token callback + tool cost tracking + graceful degradation
"""

import os
import time
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler

from dotenv import load_dotenv
load_dotenv()


# ===========================================================================
# BudgetManager — real-time cost tracking with graceful degradation
# ===========================================================================

# Cost model (per OpenAI gpt-4o-mini pricing estimates)
COST_PER_INPUT_TOKEN = 0.15 / 1_000_000   # $0.15 per 1M input tokens
COST_PER_OUTPUT_TOKEN = 0.60 / 1_000_000  # $0.60 per 1M output tokens
TOOL_COSTS = {
    "web_search": 0.01,      # $0.01 per search
    "read_webpage": 0.01,    # $0.01 per page read
    "save_research_notes": 0.0,
}


class BudgetManager:
    """Tracks cumulative token usage and cost. Enforces a per-query budget.

    When budget is exhausted, sets `budget_exceeded` flag so the agent
    can gracefully finalize instead of continuing research.
    """

    def __init__(self, max_cost: float = 1.00, max_tokens: int = 50_000):
        self.max_cost = max_cost
        self.max_tokens = max_tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tool_cost = 0.0
        self.step_log: list[dict] = []
        self.budget_exceeded = False

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def llm_cost(self) -> float:
        return (
            self.total_input_tokens * COST_PER_INPUT_TOKEN
            + self.total_output_tokens * COST_PER_OUTPUT_TOKEN
        )

    @property
    def total_cost(self) -> float:
        return self.llm_cost + self.total_tool_cost

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.max_cost - self.total_cost)

    def record_llm_usage(self, input_tokens: int, output_tokens: int, step_name: str = "llm"):
        """Record token usage from an LLM call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        cost = input_tokens * COST_PER_INPUT_TOKEN + output_tokens * COST_PER_OUTPUT_TOKEN
        self.step_log.append({
            "step": step_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "cumulative_cost": self.total_cost,
        })
        self._check_budget()

    def record_tool_usage(self, tool_name: str):
        """Record cost for a tool call."""
        cost = TOOL_COSTS.get(tool_name, 0.0)
        self.total_tool_cost += cost
        self.step_log.append({
            "step": f"tool:{tool_name}",
            "cost": cost,
            "cumulative_cost": self.total_cost,
        })
        self._check_budget()

    def _check_budget(self):
        if self.total_cost >= self.max_cost or self.total_tokens >= self.max_tokens:
            self.budget_exceeded = True

    def report(self) -> str:
        lines = [
            f"   LLM input tokens:  {self.total_input_tokens:,}",
            f"   LLM output tokens: {self.total_output_tokens:,}",
            f"   Total tokens:      {self.total_tokens:,}",
            f"   LLM cost:          ${self.llm_cost:.4f}",
            f"   Tool cost:         ${self.total_tool_cost:.4f}",
            f"   Total cost:        ${self.total_cost:.4f}",
            f"   Budget remaining:  ${self.remaining_budget:.4f}",
            f"   Budget exceeded:   {self.budget_exceeded}",
        ]
        if self.step_log:
            lines.append("   Step-by-step costs:")
            for entry in self.step_log:
                lines.append(f"     {entry['step']}: ${entry['cost']:.4f} (cumulative: ${entry['cumulative_cost']:.4f})")
        return "\n".join(lines)


class BudgetTrackingCallback(BaseCallbackHandler):
    """LangChain callback that feeds token usage into the BudgetManager.

    Tries multiple paths to extract token usage since streaming responses
    may not populate llm_output.token_usage consistently.
    """

    def __init__(self, budget: BudgetManager):
        self.budget = budget

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Capture token usage from each LLM call.

        Some providers put usage in message.usage_metadata (not llm_output).
        """
        input_tokens = 0
        output_tokens = 0

        # Path 1: message.usage_metadata (streaming providers)
        if hasattr(response, "generations"):
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg:
                        usage = getattr(msg, "usage_metadata", None)
                        if usage:
                            input_tokens = usage.get("input_tokens", 0)
                            output_tokens = usage.get("output_tokens", 0)

        # Path 2: llm_output.token_usage (standard OpenAI non-streaming)
        if input_tokens == 0 and hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

        if input_tokens > 0 or output_tokens > 0:
            self.budget.record_llm_usage(input_tokens, output_tokens, "llm_call")

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs: Any) -> None:
        """Record tool cost when a tool is invoked."""
        tool_name = serialized.get("name", "unknown")
        self.budget.record_tool_usage(tool_name)


# ===========================================================================
# Simulated web search & reading (no real API calls needed for testing)
# ===========================================================================

_SIMULATED_SEARCH_RESULTS = {
    "default": [
        {"title": "Comprehensive Guide to {topic}", "url": "https://example.com/guide", "snippet": "An in-depth look at {topic} covering all major aspects..."},
        {"title": "{topic}: What Experts Say", "url": "https://example.com/experts", "snippet": "Leading researchers weigh in on {topic}..."},
        {"title": "The History and Future of {topic}", "url": "https://example.com/history", "snippet": "From its origins to modern developments, {topic} has evolved..."},
        {"title": "10 Things You Didn't Know About {topic}", "url": "https://example.com/10things", "snippet": "Surprising facts and lesser-known aspects of {topic}..."},
        {"title": "{topic} vs Alternatives: A Deep Dive", "url": "https://example.com/comparison", "snippet": "How does {topic} compare to competing approaches..."},
    ]
}

_SEARCH_CALL_COUNT = 0
_READ_CALL_COUNT = 0


@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic.

    Args:
        query: Search query string.
    """
    global _SEARCH_CALL_COUNT
    _SEARCH_CALL_COUNT += 1

    results = []
    for r in _SIMULATED_SEARCH_RESULTS["default"]:
        results.append(
            f"  - [{r['title'].format(topic=query)}]({r['url']})\n"
            f"    {r['snippet'].format(topic=query)}"
        )

    # Every search returns results that hint at MORE things to explore
    return (
        f"Search results for '{query}' ({len(results)} results):\n\n"
        + "\n".join(results)
        + f"\n\nRelated searches: '{query} advanced techniques', "
        f"'{query} common pitfalls', '{query} case studies', "
        f"'{query} latest research 2026', '{query} expert opinions', "
        f"'{query} implementation details', '{query} theoretical foundations'"
    )


@tool
def read_webpage(url: str) -> str:
    """Read and extract content from a webpage.

    Args:
        url: The URL to read.
    """
    global _READ_CALL_COUNT
    _READ_CALL_COUNT += 1

    # Simulated page content that always suggests more avenues to explore
    return (
        f"Content from {url}:\n\n"
        f"This is a comprehensive article covering the topic in detail. "
        f"Key points include:\n"
        f"1. The fundamental principles are more nuanced than commonly understood\n"
        f"2. Recent studies (see references below) challenge earlier assumptions\n"
        f"3. Practical applications vary significantly across industries\n"
        f"4. Expert consensus is still evolving on several sub-topics\n"
        f"5. There are important connections to adjacent fields worth exploring\n\n"
        f"References cited in this article:\n"
        f"- 'Advanced Analysis of Sub-Topic A' (https://example.com/subtopic-a)\n"
        f"- 'Contrarian View on Sub-Topic B' (https://example.com/subtopic-b)\n"
        f"- 'Meta-Analysis of Related Research' (https://example.com/meta)\n"
        f"- 'Industry Report 2026' (https://example.com/report)\n\n"
        f"The author recommends further reading on at least 3 of these references "
        f"for a complete understanding."
    )


@tool
def save_notes(content: str) -> str:
    """Save research notes for later compilation into the final report.

    Args:
        content: The research notes to save.
    """
    # In the real version this writes to a file; here we just acknowledge
    return f"✅ Notes saved ({len(content)} chars). Continue researching for a comprehensive report."


# ---------------------------------------------------------------------------
# FIXED: Wrapper tool that removes the "Continue researching" nudge
# + budget-aware messaging
# ---------------------------------------------------------------------------

# Module-level budget reference so the tool can check it
_active_budget: Optional[BudgetManager] = None


@tool
def save_research_notes(content: str) -> str:
    """Save research notes for the final report. Call this when you have
    key findings to record.

    Args:
        content: The research notes to save.
    """
    result = save_notes.invoke({"content": content})
    result = result.replace(
        "Continue researching for a comprehensive report.",
        "Notes saved."
    )

    # Budget-aware feedback
    if _active_budget and _active_budget.budget_exceeded:
        return (
            f"{result} ⚠️ BUDGET LIMIT REACHED (${_active_budget.total_cost:.4f} of "
            f"${_active_budget.max_cost:.2f}). You MUST stop researching now. "
            f"Compile your saved notes into a final answer immediately."
        )
    elif _active_budget and _active_budget.remaining_budget < 0.30:
        return (
            f"{result} ⚠️ Budget running low (${_active_budget.remaining_budget:.4f} remaining). "
            f"Wrap up your research and write your final answer."
        )
    else:
        return f"{result} Review your notes and decide if you have enough to write your final answer."


# ---------------------------------------------------------------------------
# FIXED AGENT — Convergent prompt + hard limits + budget tracking
# ---------------------------------------------------------------------------

def create_research_agent(budget: Optional[BudgetManager] = None):
    global _active_budget
    if budget is None:
        budget = BudgetManager(max_cost=1.00, max_tokens=50_000)
    _active_budget = budget

    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        # base_url=os.getenv("OPENAI_BASE_URL"),  # Optional: set for custom proxy
        streaming=True,
    )

    # FIXED: Convergent prompt with explicit stopping criteria + budget awareness
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a focused research agent. Your goal is to produce a clear,
well-sourced report on the user's topic.

Research methodology:
1. Do 1-2 broad searches to understand the landscape
2. Read the 2-3 most relevant sources
3. Save your key findings as notes
4. If critical gaps remain, do ONE more targeted search
5. Compile your findings into a final answer

STOPPING CRITERIA — you MUST stop and deliver your report when ANY of these are true:
- You have saved notes covering the main aspects of the topic
- You have read 3 or more sources
- You have done 3 or more searches
- You have enough information to answer the user's question
- The save_research_notes tool warns about budget limits

BUDGET: You have a limited token/cost budget per query. If a tool response
warns about budget limits, you MUST immediately compile your notes into a
final answer. Do NOT make any more search or read calls after a budget warning.

Quality over quantity: A focused 300-word report from 3 good sources is better
than a rambling summary from 20 sources. The user wants a useful answer, not
an exhaustive literature review.

When ready, write your final answer directly — do NOT search or read further."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # FIXED: Use save_research_notes (wrapped) instead of save_notes
    tools = [web_search, read_webpage, save_research_notes]
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Budget callback tracks LLM token usage + tool costs per step
    # Must be on BOTH the LLM and the executor for full coverage
    budget_callback = BudgetTrackingCallback(budget)
    llm.callbacks = [budget_callback]

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,          # FIXED: Hard cap on iterations
        max_execution_time=120.0,  # FIXED: 2-minute hard timeout
        callbacks=[budget_callback],
    )


# ---------------------------------------------------------------------------
# Test queries — DO NOT MODIFY
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    "What are the pros and cons of microservices architecture?",
    "Summarize the current state of quantum computing in 2026.",
]


def main():
    global _SEARCH_CALL_COUNT, _READ_CALL_COUNT

    for query in TEST_QUERIES:
        _SEARCH_CALL_COUNT = 0
        _READ_CALL_COUNT = 0

        budget = BudgetManager(max_cost=1.00, max_tokens=50_000)
        agent = create_research_agent(budget=budget)
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

        try:
            result = agent.invoke({"input": query})
            print(f"\nRESPONSE: {result['output'][:500]}...")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")

        elapsed = time.time() - start_time
        print(f"\n📊 Stats:")
        print(f"   Search calls: {_SEARCH_CALL_COUNT}")
        print(f"   Page reads:   {_READ_CALL_COUNT}")
        print(f"   Time:         {elapsed:.1f}s")
        print(f"\n💰 Budget Report:")
        print(budget.report())


if __name__ == "__main__":
    main()
