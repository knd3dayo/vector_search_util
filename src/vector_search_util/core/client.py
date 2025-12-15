import asyncio
import os
from typing import Any, Optional
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
from pandas import DataFrame
from langchain_core.documents import Document
from vector_search_util.model import (
    CategoryData, RelationData, TagData, ConditionContainer, EmbeddingConfig, SourceDocumentData, SourceDocumentData
)
from vector_search_util._internal.db import SQLiteClient

from vector_search_util._internal.langchain.langchain_vector_db import LangChainVectorDB
from vector_search_util._internal.langchain.langchain_client import LangchainClient

import vector_search_util._internal.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig = EmbeddingConfig()):
        if config is None:
            config = EmbeddingConfig()
        self.config = config

        self.client = LangchainClient.create_client(config)
        self.vector_db = LangChainVectorDB.create_vector_db(self.client)
        self.category_db_path: str = os.path.join(self.config.app_data_path, "vector_db_search_app.db")
        self.sqlite_client = SQLiteClient(self.category_db_path)

    async def vector_search_langchain_documents(self, query: str, category: str = "", condition: ConditionContainer = ConditionContainer(), top_k: int = 5) -> list[Document]:
        results = await self.vector_db.vector_search(query, category, condition, top_k)
        return results
    
    async def vector_search(self, query: str, category: str = "", condition: ConditionContainer = ConditionContainer(), top_k: int = 5) -> list[SourceDocumentData]:
        results = await self.vector_db.vector_search(query, category, condition, top_k)
        return SourceDocumentData.from_langchain_documents(results, self.sqlite_client.get_content_by_source_id)

    async def get_langchain_documents(
            self,
            source_ids: list[str] = [],
            category_ids: list[str] = [],
            condition: ConditionContainer = ConditionContainer()
            ) -> tuple[list[str], list[Document]]:

        if source_ids:
            condition.add_in_condition(self.config.source_id_key, source_ids)
        if category_ids:
            condition.add_in_condition(self.config.category_key, category_ids)

        ids, results = await self.vector_db.get_documents(condition)
        return ids, results

    async def get_documents(
            self, 
            source_ids: list[str] = [],
            category_ids: list[str] = [],
            condition: ConditionContainer = ConditionContainer()
            ) -> tuple[list[str], list[SourceDocumentData]]:

        ids, results = await self.get_langchain_documents(source_ids, category_ids, condition)
        return ids, SourceDocumentData.from_langchain_documents(results, self.sqlite_client.get_content_by_source_id)
    
    async def upsert_documents(self, data_list: list[SourceDocumentData]):
        # source_documentsに新規ドキュメントがあれば追加
        await self.sqlite_client.upsert_source_documents(data_list)
        # data_listのcategoryのsetを取得して、カテゴリDBに存在しない場合は追加する
        data_list_category_names_set = set([data.category for data in data_list if data.category is not None])
        # data_listのmetadataのkeyのsetを取得して、タグDBに存在しない場合は追加する
        data_list_metadata_keys_set = set([key for data in data_list for key in data.metadata.keys() if key is not None])

        await self.vector_db.upsert_documents(SourceDocumentData.to_langchain_documents(data_list))

        # metadataに新規カテゴリがあれば追加
        await self.sqlite_client.upsert_new_categories(data_list_category_names_set)

        # metadataに新規タグがあれば追加
        await self.sqlite_client.upsert_new_tags(data_list_metadata_keys_set)

    async def delete_documents_by_source_ids(self, source_id_list: list[str], condition: ConditionContainer = ConditionContainer()):
        condition.add_in_condition(self.config.source_id_key, source_id_list)
        await self.sqlite_client.delete_source_documents(source_id_list)
        await self.vector_db.delete_documents_by_tags(condition)

    async def delete_all_documents(self):
        ids, _ = await self.vector_db.get_documents()
        await self.sqlite_client.delete_all_source_documents()
        await self.vector_db.delete_documents_by_ids(ids)
    
    async def upsert_categories(self, categories: list[CategoryData]):
        await self.sqlite_client.upsert_categories(categories)    

    async def get_categories(self, name_list: list[str] = []) -> list[CategoryData]:
        return await self.sqlite_client.get_categories(name_list)
    
    async def delete_categories(self, name_list: list[str]):
        await self.sqlite_client.delete_categories(name_list)

    async def delete_all_categories(self):
        await self.sqlite_client.delete_all_categories()

    # relations
    async def get_relations(self, from_nodes: list[str] = [], to_nodes: list[str] = [], edge_types: list[str] = []) -> list[RelationData]:
        return await self.sqlite_client.get_relations(from_nodes, to_nodes, edge_types)

    async def upsert_relations(self, relations: list[RelationData]):
        await self.sqlite_client.upsert_relations(relations)
    
    async def delete_relations(self, relations: list[RelationData]):
        await self.sqlite_client.delete_relations(relations)
    
    async def delete_all_relations(self):
        await self.sqlite_client.delete_all_relations()

    async def cleanup_categories(self):
        # Sqliteに登録されたカテゴリごとにタグ検索して、データが存在しない場合はそのタグを削除する。
        categories = await self.sqlite_client.get_categories()
    
        for category in categories:
            condition = ConditionContainer().add_in_condition(self.config.category_key, [category.name])
            vector_ids, _ = await self.vector_db.get_documents(condition)
            if len(vector_ids) == 0:
                await self.sqlite_client.delete_categories([category.name])
    
    async def get_tags(self, name_list: list[str] = []) -> list[TagData]:
        return await self.sqlite_client.get_tags(name_list)

    async def delete_tags(self, name_list: list[str]):
        await self.sqlite_client.delete_tags(name_list)

    async def upsert_tags(self, tags: list[TagData]):
        await self.sqlite_client.upsert_tags(tags)
    
    async def delete_all_tags(self):
        await self.sqlite_client.delete_all_tags()

class EmbeddingBatchClient:
    def __init__(self, embedding_client: EmbeddingClient):
        self.embedding_client = embedding_client

    async def _process_row_(self, row_num: int, data: SourceDocumentData, progress: tqdm_asyncio) -> int:
        await self.embedding_client.upsert_documents([data])
        progress.update(1)
        return row_num

    async def update(self, data_list: list[SourceDocumentData]):
        progress = tqdm_asyncio(total=len(data_list), desc="progress")
        progress.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        concurrency  = int(self.embedding_client.config.concurrency)
        async with asyncio.Semaphore(concurrency):
            tasks = [self._process_row_(i, data, progress) for i, data in enumerate(data_list)]
            await asyncio.gather(*tasks)
            progress.close()

    def __create_documents_from_dataframe__(
        self, df: DataFrame, content_column: str, source_id_column: str, category_column: str, metadata_columns: list[str]
    ) -> list[SourceDocumentData]:
        data_list: list[SourceDocumentData] = []
        for _, row in df.iterrows():
            content = row.get(content_column, "")
            source_id = row.get(source_id_column, "")
            category = row.get(category_column, "") if category_column in df.columns else ""
            if not content or not source_id:
                continue
            metadata = {key: row.get(key, "") for key in metadata_columns}
            data = SourceDocumentData(source_content=str(content), source_id=str(source_id), category=str(category), metadata=metadata)
            data_list.append(data)
        return data_list

    async def delete_documents_from_excel(
        self, file_path: str, source_id_column: str, category_column: str, tags: dict[str, list[str]] ={}
    ):
        df = pd.read_excel(file_path)
        df.replace(to_replace=r"_x000D_", value="", regex=True, inplace=True)

        source_id_list: list[str] = []
        if source_id_column in df.columns:
            source_id_list = df[source_id_column].astype(str).tolist()
        category_list: list[str] = []
        if category_column in df.columns:
            category_list = df[category_column].astype(str).tolist()
        # tagからConditionContainerを作成
        condition = ConditionContainer()
        if category_list:
            condition.add_in_condition(self.embedding_client.config.category_key, category_list)
        for key, values in tags.items():
            condition.add_in_condition(key, values)
        await self.embedding_client.delete_documents_by_source_ids(source_id_list, condition)
    
    async def load_documents_from_excel(
        self, file_path: str, content_column: str, source_id_column: str, category_column: str, metadata_columns: list[str]
    ):
        df = pd.read_excel(file_path)
        df.replace(to_replace=r"_x000D_", value="", regex=True, inplace=True)
        df.fillna("", inplace=True)
        data_list = self.__create_documents_from_dataframe__(df, content_column, source_id_column, category_column, metadata_columns)
        await self.update(data_list)
    
    async def unload_documents_to_excel(
        self, file_path: str,
        tags: dict[str, Any] ={}
    ):
        # tagからConditionContainerを作成
        condition = ConditionContainer()
        for key, values in tags.items():
            condition.add_in_condition(key, values)
        _, documents = await self.embedding_client.vector_db.get_documents(condition)
        keys = set()
        keys.add("page_content")
        data_list = []
        for document in documents:
            data = {
                "page_content": document.page_content,
            }
            keys.update(document.metadata.keys())
            data.update(document.metadata)
            data_list.append(data)

        df = DataFrame()
        # 全てのキーをDataFrameのカラムとして追加
        for key in keys:
            df[key] = [data.get(key, "") for data in data_list]

        df.to_excel(file_path, index=False)


class CategoryBatchClient:
    def __init__(self, embedding_client: EmbeddingClient):
        self.embedding_client = embedding_client

    async def _process_row_(self, row_num: int, data: CategoryData, progress: tqdm_asyncio) -> int:
        await self.embedding_client.upsert_categories([data])
        progress.update(1)
        return row_num

    async def run(self, data_list: list[CategoryData]):
        progress = tqdm_asyncio(total=len(data_list), desc="progress")
        progress.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        concurrency  = int(self.embedding_client.config.concurrency)
        async with asyncio.Semaphore(concurrency):
            tasks = [self._process_row_(i, data, progress) for i, data in enumerate(data_list)]
            await asyncio.gather(*tasks)
            progress.close()

    def create_category_data_from_dataframe(
        self, df: DataFrame, name_column: str, description_column: str
    ) -> list[CategoryData]:
        category_list: list[CategoryData] = []
        for _, row in df.iterrows():
            name = row.get(name_column, "")
            description = row.get(description_column, "")
            if not name:
                continue
            category = CategoryData(name=str(name), description=str(description))
            category_list.append(category)
        return category_list

    async def load_category_data_from_excel(
        self, file_path: str, name_column: str, description_column: str
    ):
        df = pd.read_excel(file_path)
        df.replace(to_replace=r"_x000D_", value="", regex=True, inplace=True)
        df.fillna("", inplace=True)
        category_list: list[CategoryData] = []
        for _, row in df.iterrows():
            name = row.get(name_column, "")
            description = row.get(description_column, "")
            if not name:
                continue
            category = CategoryData(name=str(name), description=str(description))
            category_list.append(category)
    
        await self.embedding_client.upsert_categories(category_list)

    async def unload_category_data_to_excel(
        self, file_path: str
    ):
        # 全カテゴリを取得してExcelに保存する
        category_list = await self.embedding_client.get_categories()
        data = {
            "name": [category.name for category in category_list],
            "description": [category.description for category in category_list],
        }
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)

    async def delete_category_data_from_excel(
        self, file_path: str, name_column: str
    ):
        df = pd.read_excel(file_path)
        df.replace(to_replace=r"_x000D_", value="", regex=True, inplace=True)
        name_list: list[str] = []
        if name_column in df.columns:
            name_list = df[name_column].astype(str).tolist()
     
        await self.embedding_client.delete_categories(name_list)

class RelationBatchClient:
    def __init__(self, embedding_client: EmbeddingClient):
        self.embedding_client = embedding_client

    async def _process_row_(self, row_num: int, data: RelationData, progress: tqdm_asyncio) -> int:
        await self.embedding_client.upsert_relations([data])
        progress.update(1)
        return row_num

    async def run(self, data_list: list[RelationData]):
        progress = tqdm_asyncio(total=len(data_list), desc="progress")
        progress.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        concurrency  = int(self.embedding_client.config.concurrency)
        async with asyncio.Semaphore(concurrency):
            tasks = [self._process_row_(i, data, progress) for i, data in enumerate(data_list)]
            await asyncio.gather(*tasks)
            progress.close()

    def create_relation_data_from_dataframe(
        self, df: DataFrame, from_node_column: str, to_node_column: str, edge_type_column: str
    ) -> list[RelationData]:
        relation_list: list[RelationData] = []
        for _, row in df.iterrows():
            from_node = row.get(from_node_column, "")
            to_node = row.get(to_node_column, "")
            edge_type = row.get(edge_type_column, "")
            if not from_node or not to_node or not edge_type:
                continue
            relation = RelationData(from_node=str(from_node), to_node=str(to_node), edge_type=str(edge_type))
            relation_list.append(relation)
        return relation_list

    async def load_relation_data_from_excel(
        self, file_path: str, from_node_column: str, to_node_column: str, edge_type_column: str
    ):
        df = pd.read_excel(file_path)
        df.replace(to_replace=r"_x000D_", value="", regex=True, inplace=True)
        df.fillna("", inplace=True)
        relation_list: list[RelationData] = []
        for _, row in df.iterrows():
            from_node = row.get(from_node_column, "")
            to_node = row.get(to_node_column, "")
            edge_type = row.get(edge_type_column, "")
            relation = RelationData(from_node=str(from_node), to_node=str(to_node), edge_type=str(edge_type))
            relation_list.append(relation)
    
        await self.embedding_client.upsert_relations(relation_list)

    async def unload_relation_data_to_excel(
        self, file_path: str
    ):
        # 全カテゴリを取得してExcelに保存する
        relation_list = await self.embedding_client.get_relations()
        data = {
            "from_node": [relation.from_node for relation in relation_list],
            "to_node": [relation.to_node for relation in relation_list],
            "edge_type": [relation.edge_type for relation in relation_list],
        }
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)

    async def delete_relation_data_from_excel(
        self, file_path: str, from_node_column: str, to_node_column: str, edge_type_column: str
    ):
        df = pd.read_excel(file_path)
        df.replace(to_replace=r"_x000D_", value="", regex=True, inplace=True)
        relation_list: list[RelationData] = []
        for _, row in df.iterrows():
            from_node = row.get(from_node_column, "")
            to_node = row.get(to_node_column, "")
            edge_type = row.get(edge_type_column, "")
            relation = RelationData(from_node=str(from_node), to_node=str(to_node), edge_type=str(edge_type))
            relation_list.append(relation)
     
        await self.embedding_client.delete_relations(relation_list)

class TagBatchClient:
    def __init__(self, embedding_client: EmbeddingClient):
        self.embedding_client = embedding_client

    async def delete_tag_data_from_excel(
        self, file_path: str, name_column: str
    ):
        df = pd.read_excel(file_path)
        df.replace(to_replace=r"_x000D_", value="", regex=True, inplace=True)
        name_list: list[str] = []
        if name_column in df.columns:
            name_list = df[name_column].astype(str).tolist()
     
        await self.embedding_client.delete_tags(name_list)

    async def unload_tag_data_to_excel(
        self, file_path: str
    ):
        # 全タグを取得してExcelに保存する
        tag_list = await self.embedding_client.get_tags()
        data = {
            "name": [tag.name for tag in tag_list],
            "description": [tag.description for tag in tag_list],
        }
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)

    async def load_tag_data_from_excel(
        self, file_path: str, name_column: str, description_column: str
    ):
        df = pd.read_excel(file_path)
        df.replace(to_replace=r"_x000D_", value="", regex=True, inplace=True)
        df.fillna("", inplace=True)
        tag_list: list[TagData] = []
        for _, row in df.iterrows():
            name = row.get(name_column, "")
            description = row.get(description_column, "")
            if not name:
                continue
            tag = TagData(name=str(name), description=str(description))
            tag_list.append(tag)
    
        await self.embedding_client.upsert_tags(tag_list)
