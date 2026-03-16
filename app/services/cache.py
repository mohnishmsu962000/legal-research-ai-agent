import structlog
import hashlib
import json
import redis.asyncio as redis
from app.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

_client = None


async def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(settings.redis_url, decode_responses=True)
    return _client


def make_cache_key(query: str, jurisdiction: str = None, document_type: str = None) -> str:
    raw = f"{query}_{jurisdiction}_{document_type}"
    return f"legal_research:{hashlib.md5(raw.encode()).hexdigest()}"


async def get_cached_response(key: str) -> dict | None:
    try:
        client = await get_redis()
        value = await client.get(key)
        if value:
            logger.info("cache_hit", key=key)
            return json.loads(value)
        return None
    except Exception as e:
        logger.error("cache_get_failed", error=str(e))
        return None


async def set_cached_response(key: str, value: dict) -> None:
    try:
        client = await get_redis()
        await client.setex(key, settings.cache_ttl, json.dumps(value))
        logger.info("cache_set", key=key, ttl=settings.cache_ttl)
    except Exception as e:
        logger.error("cache_set_failed", error=str(e))


async def invalidate_cache(key: str) -> None:
    try:
        client = await get_redis()
        await client.delete(key)
        logger.info("cache_invalidated", key=key)
    except Exception as e:
        logger.error("cache_invalidate_failed", error=str(e))