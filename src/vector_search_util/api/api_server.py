from typing import Annotated
from fastapi import FastAPI
from vector_search_util.util.client import EmbeddingClient, EmbeddingData, CategoryData, RelationData, TagData
from vector_search_util.llm.embedding_config import EmbeddingConfig

app = FastAPI()

@app.get("/vector_search", response_model=list)
async def vector_search(
    query: Annotated[str, "The search query string."],
    category: Annotated[str, "The category to filter the search by."] = "",
    filter: Annotated[dict[str, list[str]], "A dictionary of tags to filter the search by. eg { \"key\": [\"value1\", \"value2\"] }"] = {},
    num_results: Annotated[int, "The number of results to return."] = 5,
) -> list[EmbeddingData]:
    
    """Perform a vector search in the vector database.

    Args:
        query (str): The search query string.
        category (str | None): The category to filter the search by.
        filter (dict[str, list[str]] | None): A dictionary of tags to filter the search by. eg { "key": ["value1", "value2"] }
        num_results (int): The number of results to return.

    Returns:
        list: A list of search results.
    """
    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    results = await embedding_client.vector_search(query, category, filter, num_results)
    return results

# get documents
@app.get("/get_documents", response_model=list)
async def get_documents(
    source_ids: Annotated[list[str], "A list of source IDs of documents to retrieve."] = [],
    category_ids: Annotated[list[str], "A list of category IDs to filter documents by."] = [],
    tags: Annotated[dict[str, list[str]], "A dictionary of tags to filter documents by. eg { \"key\": [\"value1\", \"value2\"] }"] = {},
) -> list[EmbeddingData]:
    """Retrieve documents from the vector database based on a list of source IDs.

    Args:
        source_ids (list[str]): A list of source IDs of documents to retrieve.
        category_ids (list[str]): A list of category IDs to filter documents by.
        tags (dict[str, list[str]]): A dictionary of tags to filter documents by. eg { "key": ["value1", "value2"] }
    Returns:
        list[EmbeddingData]: A list of documents retrieved from the vector database.
    """
    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    _, documents = await embedding_client.get_documents(source_ids, category_ids, tags)
    return documents

# upsert documents
@app.post("/upsert_documents")
async def upsert_documents(
    data_list: Annotated[list[EmbeddingData], "A list of documents to update embeddings for."]
):
    """Update embeddings for a list of documents in the vector database.

    Args:
        data_list (list[EmbeddingData]): A list of documents to update embeddings for.
    """

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    await embedding_client.upsert_documents(data_list)

# delete documents
@app.delete("/delete_documents")
async def delete_documents(
    source_id_list: Annotated[list[str], "A list of source IDs of documents to delete."],
    tags: Annotated[dict[str, list[str]], "A dictionary of tags to filter documents by. eg { \"key\": [\"value1\", \"value2\"] }"] = {},
):
    """Delete documents from the vector database based on a list of source IDs.

    Args:
        source_id_list (list[str]): A list of source IDs of documents to delete.
        tags (dict[str, list[str]]): A dictionary of tags to filter documents by. eg { "key": ["value1", "value2"] }
    """

    config = EmbeddingConfig()
    embedding_client = EmbeddingClient(config)
    await embedding_client.delete_documents_by_source_ids(source_id_list, tags)

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
    