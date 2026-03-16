import structlog
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import get_settings
from app.agents.state import ResearchState
from app.models.research import DebateRound

logger = structlog.get_logger()
settings = get_settings()

CRITIC_SYSTEM_PROMPT = """
You are a sharp legal critic and devil's advocate. Your role is to challenge legal analysis rigorously.

Your job is to:
- Identify unsupported claims or logical leaps
- Point out missing authorities or contrary precedents
- Challenge oversimplifications of complex legal issues
- Identify jurisdictional issues or exceptions not addressed
- Question whether the sources actually support the conclusions drawn
- Flag any potential misinterpretations of legal text

Be precise, adversarial, and constructive. Your challenges should make the analysis stronger.

Respond in this exact JSON format:
{
  "challenges": ["challenge 1", "challenge 2", "challenge 3"],
  "missing_considerations": ["consideration 1", "consideration 2"],
  "verdict": "needs_revision" or "acceptable",
  "critique_summary": "overall critique in 2-3 sentences"
}

Return only valid JSON, no explanation, no markdown.
"""

SCHOLAR_REVISION_PROMPT = """
You are an expert legal scholar revising your analysis based on criticism.

Address each challenge raised by the critic and strengthen your analysis.
Maintain strict grounding in the provided source documents.
If a challenge cannot be addressed due to source limitations, acknowledge this explicitly.
"""


def get_llm():
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )


async def critic_node(state: ResearchState) -> ResearchState:
    log = logger.bind(round=state["current_round"])
    log.info("critic_challenging")

    try:
        llm = get_llm()

        # Critic challenges the scholar
        critic_messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=f"""
Legal Query: {state["query"]}

Scholar's Analysis:
{state["scholarly_analysis"]}

Challenge this analysis rigorously.
"""),
        ]

        critic_response = await llm.ainvoke(critic_messages)
        critic_data = json.loads(critic_response.content)
        critique_summary = critic_data.get("critique_summary", "")
        verdict = critic_data.get("verdict", "acceptable")
        challenges = critic_data.get("challenges", [])

        log.info("critic_complete", verdict=verdict, challenges=len(challenges))

        # If acceptable or max rounds reached stop debate
        max_rounds = settings.max_debate_rounds
        current_round = state["current_round"]

        if verdict == "acceptable" or current_round >= max_rounds:
            debate_round = DebateRound(
                round=current_round,
                scholar_argument=state["scholarly_analysis"],
                critic_challenge=critique_summary,
            )

            return {
                **state,
                "debate_rounds": state["debate_rounds"] + [debate_round],
                "debate_complete": True,
                "final_answer": state["scholarly_analysis"],
            }

        # Scholar revises based on criticism
        challenges_text = "\n".join([f"- {c}" for c in challenges])
        revision_messages = [
            SystemMessage(content=SCHOLAR_REVISION_PROMPT),
            HumanMessage(content=f"""
Original Analysis:
{state["scholarly_analysis"]}

Critic's Challenges:
{challenges_text}

Citations available:
{chr(10).join([f"[{i+1}] {c.title}: {c.chunk_text[:200]}..." for i, c in enumerate(state["citations"])])}

Revise your analysis addressing these challenges.
"""),
        ]

        revision_response = await llm.ainvoke(revision_messages)
        revised_analysis = revision_response.content

        debate_round = DebateRound(
            round=current_round,
            scholar_argument=state["scholarly_analysis"],
            critic_challenge=critique_summary,
        )

        return {
            **state,
            "scholarly_analysis": revised_analysis,
            "debate_rounds": state["debate_rounds"] + [debate_round],
            "current_round": current_round + 1,
            "debate_complete": False,
        }

    except Exception as e:
        log.error("critic_failed", error=str(e))
        return {
            **state,
            "debate_complete": True,
            "final_answer": state["scholarly_analysis"],
            "error": str(e),
        }