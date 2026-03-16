import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers.research import router as research_router
from app.routers.ingest import router as ingest_router
from app.services.graph import close_driver

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting_up")
    yield
    await close_driver()
    logger.info("shut_down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered legal research agent with GraphRAG, multi-agent debate loop, and hallucination filtering for accurate answers across large legal document corpora.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research_router)
app.include_router(ingest_router)


@app.get("/", tags=["health"])
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}