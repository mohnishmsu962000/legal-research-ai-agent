# Legal Research AI Agent

AI-powered legal research agent with GraphRAG, multi-agent Scholar-Critic debate loop, and hallucination filtering for accurate answers across large legal document corpora.

## What it does

Upload legal documents (case law, statutes, regulations) and ask complex legal research questions. The agent retrieves relevant passages, builds graph-based context, runs a Scholar-Critic debate loop to stress-test the analysis, and returns a grounded, cited answer with a full reasoning trace.

## Architecture
```
Legal research query
        ↓
Retrieval node
├── Pinecone (vector search — top K relevant chunks)
└── Neo4j (graph context — related concepts and documents)
        ↓
Scholar node (GPT-4o — grounded legal analysis + reflection trace)
        ↓
Critic node (GPT-4o — challenges the analysis)
        ↓
Scholar revises (if critic unsatisfied and rounds remaining)
        ↓
Repeat up to N debate rounds
        ↓
Final answer + citations + debate transcript + confidence score
        ↓
Redis cache (identical queries served instantly)
```

## Agent Graph
```
retrieval → scholar → [critic → scholar]* → END
                           ↑___________|
                        (debate loop)
```

## Tech Stack

- **FastAPI** — API server
- **LangGraph** — agent orchestration and debate loop
- **LangChain + GPT-4o** — Scholar and Critic agents
- **Pinecone** — vector database for semantic chunk retrieval
- **Neo4j AuraDB** — knowledge graph for concept relationships
- **Redis** — semantic response caching
- **OpenAI text-embedding-3-small** — document embeddings
- **PyPDF** — PDF text extraction
- **Tiktoken** — token counting and cost estimation
- **LangSmith** — observability and tracing
- **Structlog** — structured JSON logging
- **Docker** — containerization

## Key Features

### GraphRAG
Documents are indexed in both Pinecone (vector search) and Neo4j (knowledge graph). Queries use both — vector similarity for local passage retrieval and graph traversal for global concept context.

### Scholar-Critic Debate Loop
Two LangGraph agents argue with each other before returning an answer. The Scholar grounds its analysis in retrieved documents. The Critic challenges unsupported claims, missing authorities, and oversimplifications. The Scholar revises. This loop runs up to 3 rounds.

### Hallucination Filtering
The Scholar generates a reflection trace critiquing its own analysis — identifying gaps, limitations, and unsupported claims. A confidence score is returned with every response.

### Semantic Caching
Identical or semantically similar queries are served from Redis cache, reducing latency and API costs.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest/document` | Upload and ingest a PDF legal document |
| POST | `/ingest/setup` | Create Neo4j indexes (run once) |
| POST | `/research/query` | Run a legal research query |
| GET | `/health` | Health check |

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/legal-research-ai-agent.git
cd legal-research-ai-agent
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### 3. Configure environment
```bash
cp .env.example .env
```

Fill in all values in `.env`:
- `OPENAI_API_KEY` — from platform.openai.com
- `PINECONE_API_KEY` — from app.pinecone.io
- `PINECONE_INDEX_NAME` — create index with 1536 dimensions, cosine metric
- `NEO4J_URI` — from neo4j.com/cloud/aura-free
- `NEO4J_PASSWORD` — from AuraDB instance creation
- `REDIS_URL` — from redis.io/try-free

### 4. Run the server
```bash
uvicorn app.main:app --reload
```

### 5. Setup indexes (run once)
```
POST /ingest/setup
```

### 6. Ingest a document
```
POST /ingest/document
```

Upload a PDF with title, document_type, jurisdiction, year.

### 7. Run a research query
```json
{
  "query": "What was the basis for the negligence claim and how did the court rule on liability?",
  "jurisdiction": "US Federal",
  "document_type": "case_law",
  "enable_debate": true
}
```

## Sample Response Structure
```json
{
  "query": "...",
  "answer": "Revised scholarly analysis...",
  "scholarly_analysis": "...",
  "reflection_trace": "Agent's self-critique...",
  "citations": [...],
  "debate_rounds": [
    {
      "round": 1,
      "scholar_argument": "...",
      "critic_challenge": "..."
    }
  ],
  "confidence_score": 0.8,
  "cached": false
}
```

## Project Structure
```
app/
├── config.py                  # Settings and environment config
├── main.py                    # FastAPI app entry point
├── models/
│   └── research.py            # Pydantic models
├── agents/
│   ├── state.py               # Typed LangGraph state
│   ├── scholar.py             # Scholar agent — legal analysis
│   ├── critic.py              # Critic agent — challenges analysis
│   └── graph_agent.py         # LangGraph graph — debate loop
├── services/
│   ├── embeddings.py          # OpenAI embeddings
│   ├── retrieval.py           # Pinecone vector search
│   ├── graph.py               # Neo4j knowledge graph
│   └── cache.py               # Redis semantic cache
├── routers/
│   ├── research.py            # Research query endpoint
│   └── ingest.py              # Document ingestion endpoint
└── utils/
    ├── chunker.py             # PDF extraction and text chunking
    └── token_counter.py       # Token counting and cost estimation
```

## Running with Docker
```bash
docker-compose up --build
```