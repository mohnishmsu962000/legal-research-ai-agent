import structlog
import hashlib
from pypdf import PdfReader
from app.models.research import DocumentChunk, DocumentType
from app.utils.token_counter import count_tokens
from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    raw = f"{document_id}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(
    text: str,
    document_id: str,
    title: str,
    document_type: DocumentType,
    jurisdiction: str = None,
    year: int = None,
) -> list[DocumentChunk]:
    chunks = []
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    # Split into words
    words = text.split()
    current_chunk_words = []
    current_tokens = 0
    chunk_index = 0

    for word in words:
        word_tokens = count_tokens(word)

        if current_tokens + word_tokens > chunk_size and current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            chunk_id = generate_chunk_id(document_id, chunk_index)

            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                title=title,
                text=chunk_text,
                chunk_index=chunk_index,
                document_type=document_type,
                jurisdiction=jurisdiction,
                year=year,
            ))

            # Overlap — keep last N words
            overlap_words = current_chunk_words[-chunk_overlap:]
            current_chunk_words = overlap_words + [word]
            current_tokens = count_tokens(" ".join(current_chunk_words))
            chunk_index += 1
        else:
            current_chunk_words.append(word)
            current_tokens += word_tokens

    # Last chunk
    if current_chunk_words:
        chunk_text = " ".join(current_chunk_words)
        chunk_id = generate_chunk_id(document_id, chunk_index)
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            title=title,
            text=chunk_text,
            chunk_index=chunk_index,
            document_type=document_type,
            jurisdiction=jurisdiction,
            year=year,
        ))

    logger.info("document_chunked", document_id=document_id, chunks=len(chunks))
    return chunks