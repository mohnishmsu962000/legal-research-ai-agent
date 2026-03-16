import structlog
import hashlib
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.research import DocumentType, IngestRequest
from app.utils.chunker import extract_text_from_pdf, chunk_text
from app.services.retrieval import ingest_chunks
from app.services.graph import ingest_document_to_graph, create_indexes
from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

router = APIRouter(prefix="/ingest", tags=["ingest"])


def generate_document_id(title: str, document_type: str) -> str:
    raw = f"{title}_{document_type}"
    return hashlib.md5(raw.encode()).hexdigest()


def extract_concepts_from_text(text: str) -> list[str]:
    """Simple concept extraction — key legal terms."""
    legal_terms = [
        "contract", "liability", "negligence", "damages", "breach",
        "jurisdiction", "precedent", "statute", "regulation", "constitution",
        "amendment", "rights", "duty", "tort", "property", "evidence",
        "criminal", "civil", "appeal", "court", "judge", "jury",
        "plaintiff", "defendant", "motion", "discovery", "settlement",
        "injunction", "remedy", "standing", "sovereignty", "due process",
    ]
    text_lower = text.lower()
    return [term for term in legal_terms if term in text_lower]


@router.post("/document")
async def ingest_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    document_type: DocumentType = Form(...),
    jurisdiction: str = Form(None),
    year: int = Form(None),
):
    """Ingest a PDF legal document into the system."""
    log = logger.bind(title=title, document_type=document_type)

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save temp file
        temp_path = f"/tmp/{file.filename}"
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Extract text
        log.info("extracting_text")
        text = extract_text_from_pdf(temp_path)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Generate document ID
        document_id = generate_document_id(title, document_type)

        # Chunk text
        chunks = chunk_text(
            text=text,
            document_id=document_id,
            title=title,
            document_type=document_type,
            jurisdiction=jurisdiction,
            year=year,
        )

        # Extract concepts for graph
        concepts = extract_concepts_from_text(text)

        # Ingest to Pinecone
        log.info("ingesting_to_pinecone", chunks=len(chunks))
        await ingest_chunks(chunks)

        # Ingest to Neo4j
        log.info("ingesting_to_neo4j")
        await ingest_document_to_graph(
            document_id=document_id,
            title=title,
            document_type=document_type,
            jurisdiction=jurisdiction,
            year=year,
            concepts=concepts,
        )

        # Cleanup
        os.remove(temp_path)

        log.info("document_ingested", document_id=document_id, chunks=len(chunks))

        return {
            "success": True,
            "document_id": document_id,
            "title": title,
            "chunks": len(chunks),
            "concepts_extracted": concepts,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error("ingest_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/setup")
async def setup_indexes():
    """Create Neo4j indexes — run once on setup."""
    await create_indexes()
    return {"success": True, "message": "Indexes created"}