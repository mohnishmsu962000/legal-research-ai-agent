import structlog
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import get_settings
from app.agents.state import ResearchState

logger = structlog.get_logger()
settings = get_settings()

SCHOLAR_SYSTEM_PROMPT = """
You are an expert legal scholar with deep knowledge of case law, statutes, and legal precedents.

Your role is to provide rigorous, well-grounded legal analysis based strictly on the provided source documents.

Rules:
- Only make claims that are directly supported by the provided citations
- Always reference specific documents when making legal arguments
- Identify key legal principles, precedents, and statutory interpretations
- Note any conflicting authorities or jurisdictional differences
- Be precise about legal terminology
- If the sources are insufficient to answer the question, say so explicitly

Never hallucinate cases, statutes, or legal principles not present in the sources.
"""

REFLECTION_PROMPT = """
You are a rigorous legal scholar reviewing your own analysis.

Review the analysis below and generate a reflection trace addressing:
1. Are all claims supported by the cited sources?
2. Have I represented the law accurately and without oversimplification?
3. Are there gaps in the sources that limit the analysis?
4. Have I considered conflicting authorities?
5. What are the limitations of this analysis?

Be honest and precise. This reflection will be shown to the user.

Respond in this exact JSON format:
{
  "reflection": "your detailed reflection here",
  "confidence_score": 0.0 to 1.0,
  "gaps": ["gap 1", "gap 2"],
  "limitations": ["limitation 1", "limitation 2"]
}

Return only valid JSON, no explanation, no markdown.
"""


def get_llm():
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )


async def scholar_node(state: ResearchState) -> ResearchState:
    log = logger.bind(query=state["query"][:50])
    log.info("scholar_analyzing")

    try:
        llm = get_llm()

        # Build context from citations and graph
        citations_text = "\n\n".join([
            f"[{i+1}] {c.title} ({c.document_type}, {c.jurisdiction or 'N/A'}, {c.year or 'N/A'}):\n{c.chunk_text}"
            for i, c in enumerate(state["citations"])
        ])

        graph_context = state.get("graph_context", "")

        user_message = f"""
Legal Research Query: {state["query"]}

Knowledge Graph Context:
{graph_context}

Source Documents:
{citations_text}

Provide a comprehensive legal analysis based strictly on these sources.
"""

        messages = [
            SystemMessage(content=SCHOLAR_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = await llm.ainvoke(messages)
        scholarly_analysis = response.content

        # Generate reflection trace
        reflection_messages = [
            SystemMessage(content=REFLECTION_PROMPT),
            HumanMessage(content=f"Analysis to review:\n\n{scholarly_analysis}"),
        ]

        reflection_response = await llm.ainvoke(reflection_messages)
        reflection_data = json.loads(reflection_response.content)

        log.info(
            "scholar_complete",
            confidence=reflection_data.get("confidence_score"),
        )

        return {
            **state,
            "scholarly_analysis": scholarly_analysis,
            "reflection_trace": reflection_data.get("reflection", ""),
            "confidence_score": reflection_data.get("confidence_score", 0.5),
        }

    except Exception as e:
        log.error("scholar_failed", error=str(e))
        return {
            **state,
            "scholarly_analysis": "",
            "reflection_trace": "",
            "confidence_score": 0.0,
            "error": str(e),
        }