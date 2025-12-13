from vector_search_util.llm.embedding_config import EmbeddingConfig
from vector_search_util.util.client import EmbeddingClient, CategoryBatchClient, TagData, TagBatchClient

async def delete_tags(input_file_path: str, name_column: str):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = TagBatchClient(embedding_client)
    await batch_client.delete_tag_data_from_excel(input_file_path, name_column)
    print(f"Deleted tags based on names from {input_file_path}.")

async def unload_tags(output_file: str):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = TagBatchClient(embedding_client)
    await batch_client.unload_tag_data_to_excel(output_file)
    print(f"Unloaded tags to {output_file}.")


async def load_tags(input_file_path: str, name_column: str, description_column: str):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = TagBatchClient(embedding_client)
    
    await batch_client.load_tag_data_from_excel(
        input_file_path, name_column, description_column
    )

async def list_tags() -> list[TagData]:

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    
    tags = await embedding_client.list_tags()

    return tags