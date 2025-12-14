import asyncio
import aiosqlite
import sqlite3
import os
from typing import Any
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
from pandas import DataFrame


from langchain_core.documents import Document

from vector_search_util.langchain.langchain_vector_db import LangChainVectorDB
from vector_search_util.langchain.models import EmbeddingData
from vector_search_util.llm.embedding_config import EmbeddingConfig
from vector_search_util.langchain.langchain_client import LangchainClient

# category_data
class CategoryData(BaseModel):
    name: str
    description: str

# relation_data
class RelationData(BaseModel):
    # Relationの各フィールド。空文字は禁止
    from_node: str
    to_node: str
    edge_type: str

    def is_valid(self) -> bool:
        # すべてのフィールドが非空文字列であることを確認
        return all([self.from_node, self.to_node, self.edge_type])

# tag_data
class TagData(BaseModel):
    name: str
    description: str

# sqlite3
class SQLiteClient:
    initialized: bool = False
    def __init__(self, db_path: str):
        self.db_path = db_path
        if not SQLiteClient.initialized:
            dirname = os.path.dirname(self.db_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            SQLiteClient.initialized = True
            self.create_categories_table()
            self.create_tags_table()
            self.create_relations_table()

    def create_categories_table(self):
        # DBPropertiesテーブルが存在しない場合は作成する
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS categories (
                    name TEXT NOT NULL PRIMARY KEY,
                    description TEXT NOT NULL
                )
            ''')
            conn.commit()

    # Category間のリレーションを管理するテーブル
    def create_relations_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS relations (
                    from_node TEXT NOT NULL,
                    to_node TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    PRIMARY KEY (from_node, to_node, edge_type),
                    FOREIGN KEY (from_node) REFERENCES categories(name),
                    FOREIGN KEY (to_node) REFERENCES categories(name)
                )
            ''')
            conn.commit()

    def create_tags_table(self):
        # DBPropertiesテーブルが存在しない場合は作成する
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS tags (
                    name TEXT NOT NULL,
                    description TEXT,
                    PRIMARY KEY (name)
                )
            ''')
            conn.commit()

    async def get_categories(self, names: list[str] = []) -> list[CategoryData]:
        conditions = []
        if names:
            conditions.append("name IN ({})".format(",".join("?" * len(names))))
        query = "SELECT name, description FROM categories"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, tuple(names))
                rows = await cur.fetchall()
                categories = [CategoryData(name=row[0], description=row[1]) for row in rows]
                return categories

    async def delete_categories(self, names: list[str]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    DELETE FROM categories WHERE name = ?
                ''', [(name,) for name in names])
            await conn.commit()

    async def upsert_categories(self, category_list: list[CategoryData]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    INSERT INTO categories (name, description)
                    VALUES (?, ?)
                    ON CONFLICT(name) DO UPDATE SET description=excluded.description
                ''', [(category.name, category.description) for category in category_list])
            await conn.commit()
    
    async def delete_all_categories(self):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute('''
                    DELETE FROM categories
                ''')
            await conn.commit()

    async def upsert_new_categories(self, data_list_category_names_set: set[str]):
        existing_category_names_set = set([category.name for category in await self.get_categories()])
        new_category_names_set = data_list_category_names_set - existing_category_names_set
        new_categories = [CategoryData(name=category_name, description="") for category_name in new_category_names_set]
        if new_categories:
            await self.upsert_categories(new_categories)

    # relations関連
    async def get_relations(
            self, 
            from_nodes: list[str] = [], 
            to_nodes: list[str] = [], edge_types: list[str] = []
            ) -> list[RelationData]:
        conditions = []
        if from_nodes:
            conditions.append("from_node IN ({})".format(",".join("?" * len(from_nodes))))
        if to_nodes:
            conditions.append("to_node IN ({})".format(",".join("?" * len(to_nodes))))
        if edge_types:
            conditions.append("edge_type IN ({})".format(",".join("?" * len(edge_types))))

        query = "SELECT from_node, to_node, edge_type FROM relations"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, tuple(param for param in from_nodes + to_nodes + edge_types if param))
                rows = await cur.fetchall()
                relations = [RelationData(from_node=row[0], to_node=row[1], edge_type=row[2]) for row in rows]
                return relations

    async def upsert_relations(self, relations: list[RelationData]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    INSERT INTO relations (from_node, to_node, edge_type)
                    VALUES (?, ?, ?)
                    ON CONFLICT(from_node, to_node, edge_type) DO NOTHING
                ''', [(relation.from_node, relation.to_node, relation.edge_type) for relation in relations if relation.is_valid()])
            await conn.commit()

    async def delete_relations(self, relations: list[RelationData]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    DELETE FROM relations WHERE from_node = ? AND to_node = ? AND edge_type = ?
                ''', [(relation.from_node, relation.to_node, relation.edge_type) for relation in relations])
            await conn.commit()

    async def delete_all_relations(self):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute('''
                    DELETE FROM relations
                ''')
            await conn.commit()

    async def get_tags(self, names: list[str]) -> list[TagData]:
        conditions = []
        if names:
            conditions.append("name IN ({})".format(",".join("?" * len(names))))
        query = "SELECT name, description FROM tags"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, tuple(names))
                rows = await cur.fetchall()
                tags = [TagData(name=row[0], description=row[1]) for row in rows]
                return tags
        
    async def upsert_tags(self, tag_list: list[TagData]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    INSERT INTO tags (name, description)
                    VALUES (?, ?)
                    ON CONFLICT(name) DO UPDATE SET description=excluded.description
                ''', [(tag.name, tag.description) for tag in tag_list])
            await conn.commit()

    async def delete_tags(self, names: list[str]):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.executemany('''
                    DELETE FROM tags WHERE name = ?
                ''', [(name,) for name in names])
            await conn.commit()

    async def delete_all_tags(self):
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.cursor() as cur:
                await cur.execute('''
                    DELETE FROM tags
                ''')
            await conn.commit()

    async def upsert_new_tags(self, data_list_metadata_keys_set: set[str]):
        existing = {t.name for t in await self.get_tags(list(data_list_metadata_keys_set))}
        new_names = data_list_metadata_keys_set - existing
        new_tags = [TagData(name=n, description="") for n in new_names]
        if new_tags:
            await self.upsert_tags(new_tags)

class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig| None = None):
        if config is None:
            config = EmbeddingConfig()
        self.config = config

        self.client = LangchainClient.create_client(config)
        self.vector_db = LangChainVectorDB.create_vector_db(self.client)
        self.category_db_path: str = os.path.join(self.config.app_data_path, "vector_db_search_app.db")
        self.sqlite_client = SQLiteClient(self.category_db_path)

    async def vector_search(self, query: str, category: str = "", filter: dict[str, list[str]] ={}, top_k: int = 5) -> list[Document]:
        results = await self.vector_db.vector_search(query, category, filter, top_k)
        return results

    async def get_documents(
            self, 
            source_ids: list[str] = [], 
            category_ids: list[str] = [], 
            tags: dict[str, list[str]] ={}
            ) -> tuple[list[str], list[Document]]:
        if source_ids:
            tags[self.config.source_id_key] = source_ids
        if category_ids:
            tags[self.config.category_key] = category_ids

        return await self.vector_db.get_documents(tags)
    
    async def upsert_documents(self, data_list: list[EmbeddingData]):
        await self.vector_db.upsert_documents(data_list)
        # data_listのcategoryのsetを取得して、カテゴリDBに存在しない場合は追加する
        data_list_category_names_set = set([data.category for data in data_list if data.category is not None])
        # data_listのmetadataのkeyのsetを取得して、タグDBに存在しない場合は追加する
        data_list_metadata_keys_set = set([key for data in data_list for key in data.metadata.keys() if key is not None])

        # metadataに新規カテゴリがあれば追加
        await self.sqlite_client.upsert_new_categories(data_list_category_names_set)

        # metadataに新規タグがあれば追加
        await self.sqlite_client.upsert_new_tags(data_list_metadata_keys_set)

    async def delete_documents_by_source_ids(self, source_id_list: list[str], tags: dict[str, list[str]] ={}):
        tags[self.config.source_id_key] = source_id_list
        await self.vector_db.delete_documents_by_tags(tags)

    async def delete_all_documents(self):
        ids, _ = await self.vector_db.get_documents()
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
            filter = {self.config.category_key: [category.name]}
            vector_ids, _ = await self.vector_db.get_documents(filter)
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

    async def _process_row_(self, row_num: int, data: EmbeddingData, progress: tqdm_asyncio) -> int:
        await self.embedding_client.upsert_documents([data])
        progress.update(1)
        return row_num

    async def update(self, data_list: list[EmbeddingData]):
        progress = tqdm_asyncio(total=len(data_list), desc="progress")
        progress.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        concurrency  = int(self.embedding_client.config.concurrency)
        async with asyncio.Semaphore(concurrency):
            tasks = [self._process_row_(i, data, progress) for i, data in enumerate(data_list)]
            await asyncio.gather(*tasks)
            progress.close()

    def __create_documents_from_dataframe__(
        self, df: DataFrame, content_column: str, source_id_column: str, category_column: str, metadata_columns: list[str]
    ) -> list[EmbeddingData]:
        data_list: list[EmbeddingData] = []
        for _, row in df.iterrows():
            content = row.get(content_column, "")
            source_id = row.get(source_id_column, "")
            category = row.get(category_column, "") if category_column in df.columns else ""
            if not content or not source_id:
                continue
            metadata = {key: row.get(key, "") for key in metadata_columns}
            data = EmbeddingData(content=str(content), source_id=str(source_id), category=str(category), metadata=metadata)
            data_list.append(data)
        return data_list

    async def delete_documents_from_excel(
        self, file_path: str, source_id_column: str, tags: dict[str, Any] ={}
    ):
        df = pd.read_excel(file_path)
        df.replace(to_replace=r"_x000D_", value="", regex=True, inplace=True)

        source_id_list: list[str] = []
        if source_id_column in df.columns:
            source_id_list = df[source_id_column].astype(str).tolist()

        await self.embedding_client.delete_documents_by_source_ids(source_id_list, tags)
    
    def load_documents_from_excel(
        self, file_path: str, content_column: str, source_id_column: str, category_column: str, metadata_columns: list[str]
    ) -> list[EmbeddingData]:
        df = pd.read_excel(file_path)
        df.replace(to_replace=r"_x000D_", value="", regex=True, inplace=True)
        df.fillna("", inplace=True)
        return self.__create_documents_from_dataframe__(df, content_column, source_id_column, category_column, metadata_columns)
    
    async def unload_documents_to_excel(
        self, file_path: str,
        tags: dict[str, Any] ={}
    ):
        _, documents = await self.embedding_client.vector_db.get_documents(tags)
        keys = set()
        keys.add("content")
        data_list = []
        for document in documents:
            data = {
                "content": document.page_content,
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
