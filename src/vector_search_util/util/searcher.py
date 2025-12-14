from typing import Any
from vector_search_util.llm.embedding_config import EmbeddingConfig
from vector_search_util.util.client import EmbeddingClient

async def vector_search(query: str, category: str = "", tags: dict[str, list[str]]={}, num_results: int =5) -> list:
    
    # ベクトルDBの初期化
    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)

    # 検索実行
    results = await embedding_client.vector_search(query, category, tags, num_results)
    return results
