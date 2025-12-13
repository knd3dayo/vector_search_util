from vector_search_util.llm.embedding_config import EmbeddingConfig
from vector_search_util.util.client import EmbeddingClient, CategoryBatchClient, CategoryData

async def delete_categories(input_file_path: str, name_column: str):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = CategoryBatchClient(embedding_client)
    await batch_client.delete_category_data_from_excel(input_file_path, name_column)
    print(f"Deleted categories based on names from {input_file_path}.")

async def unload_categories(output_file: str):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = CategoryBatchClient(embedding_client)
    await batch_client.unload_category_data_to_excel(output_file)
    print(f"Unloaded categories to {output_file}.")


async def load_categories(input_file_path: str, name_column: str, description_column: str):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = CategoryBatchClient(embedding_client)
    
    await batch_client.load_category_data_from_excel(
        input_file_path, name_column, description_column
    )

async def list_categories() -> list[CategoryData]:

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    
    categories = await embedding_client.list_categories()

    return categories