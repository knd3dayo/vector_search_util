from typing import Annotated
from fastapi import FastAPI
from langchain_core.documents import Document
from vector_search_util.core.client import (
    EmbeddingClient, EmbeddingBatchClient, RelationBatchClient, CategoryBatchClient, TagBatchClient
)   
from vector_search_util.model import EmbeddingConfig, ConditionContainer, SourceDocumentData, CategoryData, RelationData, TagData

app = FastAPI()

# vector searchでLangChainのDocumentsを返すAPI
@app.get("/vector_search_langchain_documents", response_model=list)
async def vector_search_langchain_documents(
    query: Annotated[str, "The search query string."],
    category: Annotated[str, "The category to filter the search by."] = "",
    filter: Annotated[ConditionContainer, "A dictionary of tags to filter the search by. "] = ConditionContainer(),
    num_results: Annotated[int, "The number of results to return."] = 5,
) -> list[Document]:
    
    """Perform a vector search in the vector database and return Langchain Documents.

    Args:
        query (str): The search query string.
        category (str | None): The category to filter the search by.
        filter (ConditionContainer): A dictionary of tags to filter the search by.
        num_results (int): The number of results to return.

    Returns:
        list: A list of Langchain Documents as search results.
    """
    embedding_client = EmbeddingClient()
    results = await embedding_client.vector_search_langchain_documents(query, category, filter, num_results)
    return results

@app.get("/get_langchain_documents", response_model=list)
async def get_langchain_documents(
    source_ids: Annotated[list[str], "A list of source IDs of documents to retrieve."] = [],
    category_ids: Annotated[list[str], "A list of category IDs to filter documents by."] = [],
    filter: Annotated[ConditionContainer, "A dictionary of tags to filter documents by. "] = ConditionContainer(),
) -> list[Document]:
    embedding_client = EmbeddingClient()
    _, documents = await embedding_client.get_langchain_documents(source_ids, category_ids, filter)
    return documents


@app.get("/vector_search", response_model=list)
async def vector_search(
    query: Annotated[str, "The search query string."],
    category: Annotated[str, "The category to filter the search by."] = "",
    filter: Annotated[ConditionContainer, "A dictionary of tags to filter the search by. "] = ConditionContainer(),
    num_results: Annotated[int, "The number of results to return."] = 5,
) -> list[SourceDocumentData]:
    
    """Perform a vector search in the vector database.

    Args:
        query (str): The search query string.
        category (str | None): The category to filter the search by.
        filter (ConditionContainer): A dictionary of tags to filter the search by.
        num_results (int): The number of results to return.

    Returns:
        list: A list of search results.
    """
    embedding_client = EmbeddingClient()
    results = await embedding_client.vector_search(query, category, filter, num_results)
    return results

# get documents
@app.get("/get_documents", response_model=list)
async def get_documents(
    source_ids: Annotated[list[str], "A list of source IDs of documents to retrieve."] = [],
    category_ids: Annotated[list[str], "A list of category IDs to filter documents by."] = [],
    filter: Annotated[ConditionContainer, "A dictionary of tags to filter documents by. "] = ConditionContainer(),
) -> list[SourceDocumentData]:
    """Retrieve documents from the vector database based on a list of source IDs.

    Args:
        source_ids (list[str]): A list of source IDs of documents to retrieve.
        category_ids (list[str]): A list of category IDs to filter documents by.
        filter (ConditionContainer): A dictionary of tags to filter documents by.
    Returns:
        list[EmbeddingData]: A list of documents retrieved from the vector database.
    """
    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    _, documents = await embedding_client.get_documents(source_ids, category_ids, filter)
    return documents

# upsert documents
@app.post("/upsert_documents")
async def upsert_documents(
    data_list: Annotated[list[SourceDocumentData], "A list of documents to update embeddings for."]
):
    """Update embeddings for a list of documents in the vector database.

    Args:
        data_list (list[SourceDocumentData]): A list of documents to update embeddings for.
    """

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    await embedding_client.upsert_documents(data_list)

# delete documents
@app.delete("/delete_documents")
async def delete_documents(
    source_id_list: Annotated[list[str], "A list of source IDs of documents to delete."],
    filter: Annotated[ConditionContainer, "A dictionary of tags to filter documents by. "] = ConditionContainer(),
):
    """Delete documents from the vector database based on a list of source IDs.

    Args:
        source_id_list (list[str]): A list of source IDs of documents to delete.
        filter (ConditionContainer): A dictionary of tags to filter documents by.
    """

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    await embedding_client.delete_documents_by_source_ids(source_id_list, filter)

# get categories
@app.get("/get_categories", response_model=list)
async def get_categories(
    name_list: Annotated[list[str], "A list of category names to retrieve."] = [],
) -> list[CategoryData]:
    """Retrieve categories from the vector database.

    Args:
        name_list (list[str]): A list of category names to retrieve.
    Returns:
        list[CategoryData]: A list of categories retrieved from the vector database.
    """
    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    categories = await embedding_client.get_categories(name_list)
    return categories

# upsert categories
@app.post("/upsert_categories")
async def upsert_categories(
    categories: Annotated[list[CategoryData], "The list of categories to update."],
):
    """Update a category in the vector database.

    Args:
        categories (list[CategoryData]): The list of categories to update.
    """

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    await embedding_client.upsert_categories(categories)    

# delete category
@app.delete("/delete_categories")
async def delete_categories(
    name_list: Annotated[list[str], "The list of category names to delete."],
):
    """Delete categories from the vector database based on a list of category names.

    Args:
        name_list (list[str]): The list of category names to delete.
    """

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    await embedding_client.delete_categories(name_list)

# get relations
@app.get("/get_relations")
async def get_relations() -> list[RelationData]:
    """Retrieve relations from the vector database.

    Returns:
        list[RelationData]: A list of relations retrieved from the vector database.
    """
    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    relations = await embedding_client.get_relations()
    return relations

# upsert relations
@app.post("/upsert_relations")
async def upsert_relations(
    relations: Annotated[list[RelationData], "The list of relations to upsert."],
):
    """Upsert relations in the vector database.

    Args:
        relations (list[RelationData]): The list of relations to upsert.
    """

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    await embedding_client.upsert_relations(relations)

# delete relations
@app.delete("/delete_relations")
async def delete_relations(
    relations: Annotated[list[RelationData], "The list of relations to delete."],
):
    """Delete relations from the vector database based on a list of relation IDs.

    Args:
        relation_ids (list[str]): The list of relation IDs to delete.
    """

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    await embedding_client.delete_relations(relations)

# get tags
@app.get("/get_tags")
async def get_tags() -> list[TagData]:
    """Retrieve tags from the vector database.

    Returns:
        list[TagData]: A list of tags retrieved from the vector database.
    """
    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    tags = await embedding_client.get_tags()
    return tags

# upsert tags
@app.post("/upsert_tags")
async def upsert_tags(
    tags: Annotated[list[TagData], "The list of tags to upsert."],
):
    """Upsert tags in the vector database.

    Args:
        tags (list[TagData]): The list of tags to upsert.
    """

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    await embedding_client.upsert_tags(tags)

# delete tags
@app.delete("/delete_tags")
async def delete_tags(
    name_list: Annotated[list[str], "The list of tag names to delete."],
):
    """Delete tags from the vector database based on a list of tag names.

    Args:
        name_list (list[str]): The list of tag names to delete.
    """

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    await embedding_client.delete_tags(name_list)

@app.post("/load_documents_from_excel")
async def load_documents_from_excel(
        file_path: Annotated[str, "The path to the Excel file."],
        content_column: Annotated[str, "The name of the column containing document content."] = "content",
        source_id_column: Annotated[str, "The name of the column containing source IDs."] = "source_id",
        category_column: Annotated[str, "The name of the column containing categories."] = "category",
        metadata_columns: Annotated[list[str], "A list of column names to include as metadata."] = [],
        append_vectors: Annotated[bool, """
                                  If True, add vectors for existing source document search. 
                                  If the vector DB has existing documents, vectors for existing document search are added. 
                                  If the vector DB has no existing documents, new ones are created."""] = False
    ):
    """Load documents from an Excel file into the vector database.

    Args:
        file_path (str): The path to the Excel file.
        content_column (str): The name of the column containing document content.
        source_id_column (str): The name of the column containing source IDs.
        category_column (str): The name of the column containing categories.
        metadata_columns (list[str]): A list of column names to include as metadata.
        append_vectors (bool): If true, add vectors for existing source document search.
    """

    embedding_client = EmbeddingClient()
    batch_client = EmbeddingBatchClient(embedding_client)
    await batch_client.load_documents_from_excel(
        file_path, content_column, source_id_column, category_column, metadata_columns, append_vectors
    )

@app.get("/unload_documents_to_excel")
async def unload_documents_to_excel(
        file_path: Annotated[str, "The path to the output Excel file."]
    ):
    """Unload documents from the vector database to an Excel file.

    Args:
        file_path (str): The path to the output Excel file.
    """

    embedding_client = EmbeddingClient()
    batch_client = EmbeddingBatchClient(embedding_client)
    await batch_client.unload_documents_to_excel(file_path)

@app.delete("/delete_documents_from_excel")
async def delete_documents_from_excel(
        file_path: Annotated[str, "The path to the Excel file."],
        source_id_column: Annotated[str, "The name of the column containing source IDs."] = "source_id",
        category_column: Annotated[str, "The name of the column containing categories."] = "category",
        metadata_columns: Annotated[dict[str, list[str]], "A list of column names to include as metadata."] = {}
    ):
    """Delete documents from the vector database based on an Excel file.

    Args:
        file_path (str): The path to the Excel file.
        source_id_column (str): The name of the column containing source IDs.
        category_column (str): The name of the column containing categories.
        metadata_columns (list[str]): A list of column names to include as metadata.
    """

    embedding_client = EmbeddingClient()
    batch_client = EmbeddingBatchClient(embedding_client)
    await batch_client.delete_documents_from_excel(
        file_path, source_id_column, category_column, metadata_columns
    )

@app.post("/load_categories_from_excel")
async def load_categories(
        input_file_path: str, name_column: str, description_column: str
    ):
    embedding_client = EmbeddingClient()
    batch_client = CategoryBatchClient(embedding_client)
    
    await batch_client.load_category_data_from_excel(
        input_file_path, name_column, description_column
    )

@app.get("/unload_categories_to_excel")
async def unload_categories(output_file: str):

    embedding_client = EmbeddingClient()
    batch_client = CategoryBatchClient(embedding_client)
    await batch_client.unload_category_data_to_excel(output_file)

@app.post("/load_relations_from_excel")
async def load_relations_from_excel(input_file_path: str, from_node_column: str, to_node_column: str, edge_type_column: str):
    embedding_client = EmbeddingClient()
    batch_client = RelationBatchClient(embedding_client)
    
    await batch_client.load_relation_data_from_excel(
        input_file_path, from_node_column, to_node_column, edge_type_column
    )
@app.get("/unload_relations_to_excel")
async def unload_relations_to_excel(output_file: str):

    embedding_client = EmbeddingClient()
    batch_client = RelationBatchClient(embedding_client)
    await batch_client.unload_relation_data_to_excel(output_file)

@app.delete("/delete_relations_from_excel")
async def delete_relations_from_excel(input_file_path: str, from_node_column: str, to_node_column: str, edge_type_column: str):
    embedding_client = EmbeddingClient()
    batch_client = RelationBatchClient(embedding_client)
    await batch_client.delete_relation_data_from_excel(input_file_path, from_node_column, to_node_column, edge_type_column)

@app.post("/load_tags_from_excel")
async def load_tags_from_excel(input_file_path, name_column, description_column):
    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = TagBatchClient(embedding_client)
    
    await batch_client.load_tag_data_from_excel(
        input_file_path, name_column, description_column
    )

@app.get("/unload_tags_to_excel")
async def unload_tags_from_excel(output_file: str):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = TagBatchClient(embedding_client)
    await batch_client.unload_tag_data_to_excel(output_file)

@app.delete("/delete_tags_from_excel")
async def delete_tags_from_excel(input_file_path: str, name_column: str):

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    batch_client = TagBatchClient(embedding_client)
    await batch_client.delete_tag_data_from_excel(input_file_path, name_column)




# ping endpoint
@app.get("/ping")
async def ping():
    """Ping the API server to check if it's alive.

    Returns:
        str: A message indicating that the server is alive.
    """
    return "Pong! The API server is alive."

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)
