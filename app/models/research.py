from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DocumentType(str, Enum):
    CASE_LAW = "case_law"
    STATUTE = "statute"
    REGULATION = "regulation"
    LEGAL_BRIEF = "legal_brief"
    CONTRACT = "contract"


class IngestRequest(BaseModel):
    title: str = Field(..., description="Title of the legal document")
    document_type: DocumentType = Field(..., description="Type of legal document")
    jurisdiction: Optional[str] = Field(None, description="Jurisdiction e.g. US Federal, California")
    year: Optional[int] = Field(None, description="Year of the document")
    source_url: Optional[str] = Field(None, description="Source URL if available")


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    text: str
    chunk_index: int
    document_type: DocumentType
    jurisdiction: Optional[str] = None
    year: Optional[int] = None


class ResearchQuery(BaseModel):
    query: str = Field(..., description="Legal research question")
    jurisdiction: Optional[str] = Field(None, description="Filter by jurisdiction")
    document_type: Optional[DocumentType] = Field(None, description="Filter by document type")
    year_from: Optional[int] = Field(None, description="Filter from year")
    year_to: Optional[int] = Field(None, description="Filter to year")
    enable_debate: bool = Field(True, description="Enable Scholar-Critic debate loop")


class Citation(BaseModel):
    document_id: str
    title: str
    chunk_text: str
    relevance_score: float
    document_type: DocumentType
    jurisdiction: Optional[str] = None
    year: Optional[int] = None


class DebateRound(BaseModel):
    round: int
    scholar_argument: str
    critic_challenge: str


class ResearchResponse(BaseModel):
    query: str
    answer: str
    scholarly_analysis: str
    reflection_trace: str
    citations: list[Citation]
    debate_rounds: list[DebateRound]
    confidence_score: float
    cached: bool = False