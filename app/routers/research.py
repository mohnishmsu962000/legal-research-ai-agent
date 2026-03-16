import structlog
from fastapi import APIRouter, HTTPException
from app.models.research import ResearchQuery, ResearchResponse
from app.agents.graph_agent import run_research
from app.services.cache import make_cache_key, get_cached_response, set_cached_response

logger = structlog.get_logger()

router = APIRouter(prefix="/research", tags=["research"])


@router.post("/query", response_model=ResearchResponse)
async def research_query(payload: ResearchQuery):
    """Run a legal research query through the full agent pipeline."""
    log = logger.bind(query=payload.query[:50])
    log.info("research_query_received")

    # Check cache
    cache_key = make_cache_key(
        payload.query,
        payload.jurisdiction,
        payload.document_type,
    )

    cached = await get_cached_response(cache_key)
    if cached:
        return ResearchResponse(**cached, cached=True)

    try:
        result = await run_research(
            query=payload.query,
            jurisdiction=payload.jurisdiction,
            document_type=payload.document_type.value if payload.document_type else None,
            enable_debate=payload.enable_debate,
        )

        if result.get("error") and not result.get("final_answer"):
            raise HTTPException(status_code=500, detail=result["error"])

        response = ResearchResponse(
            query=payload.query,
            answer=result.get("final_answer") or result.get("scholarly_analysis", ""),
            scholarly_analysis=result.get("scholarly_analysis", ""),
            reflection_trace=result.get("reflection_trace", ""),
            citations=result.get("citations", []),
            debate_rounds=result.get("debate_rounds", []),
            confidence_score=result.get("confidence_score", 0.0),
            cached=False,
        )

        # Cache the response
        await set_cached_response(cache_key, response.model_dump())

        log.info(
            "research_complete",
            confidence=response.confidence_score,
            citations=len(response.citations),
            debate_rounds=len(response.debate_rounds),
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        log.error("research_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def research_health():
    return {"status": "ok"}