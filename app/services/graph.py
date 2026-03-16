import structlog
from neo4j import AsyncGraphDatabase
from app.config import get_settings
from app.models.research import DocumentChunk

logger = structlog.get_logger()
settings = get_settings()

_driver = None


async def get_driver():
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
    return _driver


async def close_driver():
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


async def create_indexes():
    """Create Neo4j indexes for performance."""
    driver = await get_driver()
    async with driver.session() as session:
        await session.run("CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.document_id)")
        await session.run("CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)")
        await session.run("CREATE INDEX jurisdiction_name IF NOT EXISTS FOR (j:Jurisdiction) ON (j.name)")
    logger.info("neo4j_indexes_created")


async def ingest_document_to_graph(
    document_id: str,
    title: str,
    document_type: str,
    jurisdiction: str = None,
    year: int = None,
    concepts: list[str] = None,
):
    """Create document node and relationships in Neo4j."""
    driver = await get_driver()
    async with driver.session() as session:
        # Create document node
        await session.run(
            """
            MERGE (d:Document {document_id: $document_id})
            SET d.title = $title,
                d.document_type = $document_type,
                d.year = $year
            """,
            document_id=document_id,
            title=title,
            document_type=document_type,
            year=year,
        )

        # Create jurisdiction node and relationship
        if jurisdiction:
            await session.run(
                """
                MERGE (j:Jurisdiction {name: $jurisdiction})
                WITH j
                MATCH (d:Document {document_id: $document_id})
                MERGE (d)-[:BELONGS_TO]->(j)
                """,
                jurisdiction=jurisdiction,
                document_id=document_id,
            )

        # Create concept nodes and relationships
        if concepts:
            for concept in concepts:
                await session.run(
                    """
                    MERGE (c:Concept {name: $concept})
                    WITH c
                    MATCH (d:Document {document_id: $document_id})
                    MERGE (d)-[:CONTAINS_CONCEPT]->(c)
                    """,
                    concept=concept,
                    document_id=document_id,
                )

    logger.info("document_ingested_to_graph", document_id=document_id)


async def get_related_documents(document_id: str, limit: int = 5) -> list[dict]:
    """Find documents related through shared concepts or jurisdiction."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document {document_id: $document_id})-[:CONTAINS_CONCEPT]->(c:Concept)
            <-[:CONTAINS_CONCEPT]-(related:Document)
            WHERE related.document_id <> $document_id
            WITH related, count(c) as shared_concepts
            ORDER BY shared_concepts DESC
            LIMIT $limit
            RETURN related.document_id as document_id,
                   related.title as title,
                   related.document_type as document_type,
                   shared_concepts
            """,
            document_id=document_id,
            limit=limit,
        )
        return [dict(record) async for record in result]


async def get_concept_community(concept: str, limit: int = 10) -> list[dict]:
    """Get all documents related to a legal concept."""
    driver = await get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (c:Concept {name: $concept})<-[:CONTAINS_CONCEPT]-(d:Document)
            OPTIONAL MATCH (d)-[:BELONGS_TO]->(j:Jurisdiction)
            RETURN d.document_id as document_id,
                   d.title as title,
                   d.document_type as document_type,
                   d.year as year,
                   j.name as jurisdiction
            LIMIT $limit
            """,
            concept=concept,
            limit=limit,
        )
        return [dict(record) async for record in result]


async def get_graph_context(query_concepts: list[str]) -> str:
    """Build a global context summary from the knowledge graph."""
    driver = await get_driver()
    context_parts = []

    async with driver.session() as session:
        for concept in query_concepts[:5]:
            result = await session.run(
                """
                MATCH (c:Concept {name: $concept})<-[:CONTAINS_CONCEPT]-(d:Document)
                RETURN d.title as title, d.document_type as type, d.year as year
                LIMIT 5
                """,
                concept=concept,
            )
            docs = [dict(record) async for record in result]
            if docs:
                doc_list = ", ".join([f"{d['title']} ({d['type']}, {d['year']})" for d in docs])
                context_parts.append(f"Concept '{concept}' appears in: {doc_list}")

    return "\n".join(context_parts)