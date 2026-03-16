from typing import TypedDict, Optional
from app.models.research import Citation, DebateRound


class ResearchState(TypedDict):
    # Input
    query: str
    jurisdiction: Optional[str]
    document_type: Optional[str]
    enable_debate: bool

    # Retrieval
    citations: list[Citation]
    graph_context: str

    # Scholar
    scholarly_analysis: str
    reflection_trace: str

    # Debate
    debate_rounds: list[DebateRound]
    current_round: int
    debate_complete: bool

    # Output
    final_answer: str
    confidence_score: float
    error: Optional[str]