
from vector_search_util.llm.embedding_config import EmbeddingConfig
from vector_search_util.util.client import EmbeddingClient, EmbeddingBatchClient

async def load_documents_from_excel(file_path: str, content_column: str, source_id_column: str, category_column: str, metadata_columns: list[str]):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = EmbeddingBatchClient(embedding_client)

    data_list = batch_client.load_documents_from_excel(
        file_path, content_column, source_id_column, category_column, metadata_columns
    )
    await batch_client.update(data_list)
    print(f"Loaded {len(data_list)} records from {file_path}.")

async def unload_documents_to_excel(output_file: str, tags: dict[str, str] = {}):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = EmbeddingBatchClient(embedding_client)
    await batch_client.unload_documents_to_excel(output_file, tags)
    print(f"Unloaded embeddings to {output_file}.")

async def delete_documents_from_excel(input_file_path: str, source_id_column: str, tags: dict[str, str] = {}):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = EmbeddingBatchClient(embedding_client)
    await batch_client.delete_documents_from_excel(input_file_path, source_id_column, tags)
    print(f"Deleted records based on source IDs from {input_file_path}.")
