import structlog
from langgraph.graph import StateGraph, END
from app.agents.state import ResearchState
from app.agents.scholar import scholar_node
from app.agents.critic import critic_node
from app.services.retrieval import retrieve_chunks
from app.services.graph import get_graph_context
from app.utils.token_counter import count_tokens
from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


async def retrieval_node(state: ResearchState) -> ResearchState:
    """Retrieve relevant chunks from Pinecone and graph context from Neo4j."""
    log = logger.bind(query=state["query"][:50])
    log.info("retrieving_context")

    try:
        # Vector retrieval from Pinecone
        citations = await retrieve_chunks(
            query=state["query"],
            jurisdiction=state.get("jurisdiction"),
            document_type=state.get("document_type"),
        )

        # Extract concepts from query for graph lookup
        query_concepts = [
            word.lower() for word in state["query"].split()
            if len(word) > 5
        ][:5]

        # Graph context from Neo4j
        graph_context = await get_graph_context(query_concepts)

        log.info("retrieval_complete", citations=len(citations))

        return {
            **state,
            "citations": citations,
            "graph_context": graph_context,
        }

    except Exception as e:
        log.error("retrieval_failed", error=str(e))
        return {
            **state,
            "citations": [],
            "graph_context": "",
            "error": str(e),
        }


def should_continue_debate(state: ResearchState) -> str:
    """Router — continue debate or end."""
    if not state.get("enable_debate"):
        return "end"
    if state.get("debate_complete"):
        return "end"
    if state.get("current_round", 0) >= settings.max_debate_rounds:
        return "end"
    return "critic"


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("retrieval", retrieval_node)
    graph.add_node("scholar", scholar_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("retrieval")

    graph.add_edge("retrieval", "scholar")

    graph.add_conditional_edges(
        "scholar",
        should_continue_debate,
        {
            "critic": "critic",
            "end": END,
        }
    )

    graph.add_conditional_edges(
        "critic",
        should_continue_debate,
        {
            "critic": "scholar",
            "end": END,
        }
    )

    return graph.compile()


agent_graph = build_graph()


async def run_research(
    query: str,
    jurisdiction: str = None,
    document_type: str = None,
    enable_debate: bool = True,
) -> ResearchState:
    log = logger.bind(query=query[:50])
    log.info("research_started")

    initial_state: ResearchState = {
        "query": query,
        "jurisdiction": jurisdiction,
        "document_type": document_type,
        "enable_debate": enable_debate,
        "citations": [],
        "graph_context": "",
        "scholarly_analysis": "",
        "reflection_trace": "",
        "debate_rounds": [],
        "current_round": 1,
        "debate_complete": False,
        "final_answer": "",
        "confidence_score": 0.0,
        "error": None,
    }

    result = await agent_graph.ainvoke(initial_state)

    log.info(
        "research_complete",
        citations=len(result.get("citations", [])),
        rounds=len(result.get("debate_rounds", [])),
        confidence=result.get("confidence_score"),
    )

    return result