import structlog
from pinecone import Pinecone
from app.config import get_settings
from app.models.research import DocumentChunk, Citation
from app.services.embeddings import embed_text, embed_batch

logger = structlog.get_logger()
settings = get_settings()

_index = None


def get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        _index = pc.Index(settings.pinecone_index_name)
    return _index


async def ingest_chunks(chunks: list[DocumentChunk]) -> None:
    """Embed and upsert document chunks into Pinecone."""
    log = logger.bind(chunks=len(chunks))
    log.info("ingesting_chunks")

    index = get_index()
    texts = [chunk.text for chunk in chunks]
    embeddings = await embed_batch(texts)

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk.chunk_id,
            "values": embedding,
            "metadata": {
                "document_id": chunk.document_id,
                "title": chunk.title,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "document_type": chunk.document_type,
                "jurisdiction": chunk.jurisdiction or "",
                "year": chunk.year or 0,
            }
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    log.info("chunks_ingested")


async def retrieve_chunks(
    query: str,
    top_k: int = None,
    jurisdiction: str = None,
    document_type: str = None,
    year_from: int = None,
    year_to: int = None,
) -> list[Citation]:
    """Retrieve relevant chunks from Pinecone with optional metadata filtering."""
    top_k = top_k or settings.retrieval_top_k
    query_embedding = await embed_text(query)
    index = get_index()

    # Build metadata filter
    filter_dict = {}
    if jurisdiction:
        filter_dict["jurisdiction"] = {"$eq": jurisdiction}
    if document_type:
        filter_dict["document_type"] = {"$eq": document_type}
    if year_from and year_to:
        filter_dict["year"] = {"$gte": year_from, "$lte": year_to}
    elif year_from:
        filter_dict["year"] = {"$gte": year_from}
    elif year_to:
        filter_dict["year"] = {"$lte": year_to}

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None,
    )

    citations = []
    for match in results.matches:
        meta = match.metadata
        citations.append(Citation(
            document_id=meta["document_id"],
            title=meta["title"],
            chunk_text=meta["text"],
            relevance_score=match.score,
            document_type=meta["document_type"],
            jurisdiction=meta.get("jurisdiction") or None,
            year=meta.get("year") or None,
        ))

    logger.info("chunks_retrieved", query=query[:50], results=len(citations))
    return citations