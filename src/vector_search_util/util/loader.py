
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

from vector_search_util.llm.embedding_config import EmbeddingConfig
from vector_search_util.util.client import EmbeddingClient, RelationBatchClient, RelationData

async def delete_relations(input_file_path: str, from_node_column: str, to_node_column: str, edge_type_column: str):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = RelationBatchClient(embedding_client)
    await batch_client.delete_relation_data_from_excel(input_file_path, from_node_column, to_node_column, edge_type_column)
    print(f"Deleted categories based on names from {input_file_path}.")

async def unload_relations(output_file: str):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = RelationBatchClient(embedding_client)
    await batch_client.unload_relation_data_to_excel(output_file)
    print(f"Unloaded categories to {output_file}.")


async def load_relations(input_file_path: str, from_node_column: str, to_node_column: str, edge_type_column: str):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = RelationBatchClient(embedding_client)
    
    await batch_client.load_relation_data_from_excel(
        input_file_path, from_node_column, to_node_column, edge_type_column
    )

async def list_relations() -> list[RelationData]:

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    
    relations = await embedding_client.get_relations()

    return relations

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
    
    categories = await embedding_client.get_categories()

    return categories

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
    
    tags = await embedding_client.get_tags()

    return tags