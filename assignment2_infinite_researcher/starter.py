"""
Assignment 2: The Infinite Researcher
======================================
This research agent searches the web and compiles reports, but it NEVER STOPS.
It keeps exploring deeper and deeper until it crashes or drains your API budget.

Run this and observe the runaway behavior:
    python starter.py

Your job: Add stopping conditions, cost controls, and fix the prompt.
"""

import os
import time
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Simulated web search & reading (no real API calls needed for testing)
# ---------------------------------------------------------------------------

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
# 🐛 THE BROKEN AGENT — This is what you need to fix
# ---------------------------------------------------------------------------

def create_research_agent():
    llm = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        # base_url=os.getenv("OPENAI_BASE_URL"),  # Optional: set for custom proxy
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an elite deep research agent. Your goal is to produce the most
         comprehensive, authoritative, and exhaustive report possible.

         Your research methodology:
         - Search for information from multiple angles and perspectives
         - Read EVERY source you find — leave no stone unturned
         - Follow ALL references and citations to their source material
         - Cross-reference claims across multiple sources
         - When you find a new sub-topic, explore it fully before moving on
         - Check for contrarian viewpoints and edge cases
         - Verify facts by finding at least 3 independent sources
         - If a source mentions related topics, research those too
         - NEVER settle for surface-level understanding
         - Keep researching until you have COMPLETE knowledge of the topic
         - The user is counting on you to be thorough — missing information
           is worse than taking extra time

         Remember: A truly great researcher never stops asking questions.
         Every answer should lead to deeper inquiry."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    tools = [web_search, read_webpage, save_notes]
    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
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
    agent = create_research_agent()

    for query in TEST_QUERIES:
        _SEARCH_CALL_COUNT = 0
        _READ_CALL_COUNT = 0
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
        print(f"   (Imagine each search = $0.01, each LLM call = $0.05)")
        print(f"   Estimated cost: ${_SEARCH_CALL_COUNT * 0.01 + (_SEARCH_CALL_COUNT + _READ_CALL_COUNT) * 0.05:.2f}")


if __name__ == "__main__":
    main()
