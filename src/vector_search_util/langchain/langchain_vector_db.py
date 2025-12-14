from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Optional, Sequence
import asyncio, os

from pydantic import Field
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

import chromadb
import chromadb.config

from langchain_chroma.vectorstores import Chroma # type: ignore
from langchain_postgres.vectorstores import PGVector

import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy.sql import text

from openai import RateLimitError

from vector_search_util.langchain.langchain_client import LangchainClient
from vector_search_util.langchain.models import EmbeddingData

import vector_search_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)


class LangChainVectorDB(ABC):
    """
    LangChainのベクトルDBを利用するための基底クラス。
    """
    client: LangchainClient = Field(..., description="LangchainClientのインスタンス")
    vector_db_url: str = Field(..., description="Vector DBのURL")
    collection_name: str = Field(default="", description="コレクション名")
    db: Optional[VectorStore] = Field(default=None, description="LangChainのVectorStoreインスタンス")

    @abstractmethod
    # document_idのリストとmetadataのリストを返す
    def _get_documents_(self, tags: dict[str, list[str]]) -> Tuple[List[str], List[Document]]:
        pass

    @abstractmethod
    def _create_search_kwargs_(self, k:int, tags: dict[str, list[str]]) -> dict[str, Any]:
        pass
    
    @classmethod
    def create_vector_db(cls, client: LangchainClient) -> 'LangChainVectorDB':
        vector_db_type = client.llm_config.vector_db_type
        vector_db_url = client.llm_config.vector_db_url
        collection_name = client.llm_config.vector_db_collection_name
        if vector_db_type == "chroma":
            vector_db = LangChainVectorDBChroma(client, vector_db_url, collection_name)
            return vector_db
        elif vector_db_type == "pgvector":
            vector_db = LangChainVectorDBPGVector(client, vector_db_url, collection_name)
            return vector_db
        else:
            raise ValueError(f"Unsupported vector_db_type: {vector_db_type}")

    ########################################
    # パブリック
    ########################################
    async def get_documents(self, tags: dict[str, list[str]] ={}) -> Tuple[List[str], List[Document]]:

        return self._get_documents_(tags)
    

    async def add_documents(self, data_list: list[EmbeddingData]):

        if self.db is None:
            raise ValueError("db is None")
 
        source_id_key = self.client.llm_config.source_id_key
        category_key = self.client.llm_config.category_key
        documents: list[Document] = []
        for data in data_list:
            # metadataにsource_idがない場合は追加する
            if source_id_key not in data.metadata.keys():
                data.metadata[source_id_key] = data.source_id

            # categoryが指定されている場合はmetadataに追加する
            if data.category and category_key not in data.metadata.keys():
                data.metadata[category_key] = data.category

            # Documentを作成
            document = Document(
                page_content=data.content,
                metadata=data.metadata
            )
            documents.append(document)
        await self.add_doucment_with_retry(self.db, documents)

    async def delete_documents_by_ids(self, doc_ids:list=[]):
        if len(doc_ids) == 0:
            return
        if self.db is None:
            raise ValueError("db is None")

        await self.db.adelete(ids=doc_ids)

        return len(doc_ids)    

    async def delete_documents_by_tags(self, tags: dict[str, list[str]] ={}):
        # ベクトルDB固有のvector id取得メソッドを呼び出し。
        vector_ids, _ = self._get_documents_(tags)

        # vector_idsが空の場合は何もしない
        if len(vector_ids) == 0:
            return
        await self.delete_documents_by_ids(vector_ids)

    async def upsert_documents(self, data_list: list[EmbeddingData]):
        
        # 既に存在するドキュメントを削除
        for data in data_list:
            tags = { self.client.llm_config.source_id_key: [data.source_id] }
            await self.delete_documents_by_tags(tags)

        # ドキュメントを格納する。
        await self.add_documents(data_list)

    # RateLimitErrorが発生した場合は、指数バックオフを行う
    async def add_doucment_with_retry(self, vector_db: VectorStore, documents: list[Document], max_retries: int = 5, delay: float = 1.0):
        for attempt in range(max_retries):
            try:
                await vector_db.aadd_documents(documents=documents)
                return
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"RateLimitError: {e}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Max retries reached. Failed to add documents: {e}")
                    break
            except Exception as e:
                logger.error(f"Error adding documents: {e}")
                break

    async def vector_search(self, query: str, category: str = "", filter: dict[str, list[str]] = {}, k: int = 5) -> List[Document]:
        """
        ベクトルDBからドキュメントを検索する。
        :param query: 検索クエリ
        :param search_kwargs: 検索キーワード
        :return: 検索結果のドキュメントリスト
        """
        if self.db is None:
            raise ValueError("db is None")

        modified_filter = filter.copy()
        # categoryが指定されている場合はfilterに追加
        if category:
            category_key = self.client.llm_config.category_key
            modified_filter[category_key] = [category]

        search_kwargs: dict[str, Any] = self._create_search_kwargs_(k, modified_filter)

        docs_and_scores = self.db.similarity_search_with_relevance_scores(query, **search_kwargs)
        # documentのmetadataにscoreを追加
        doc_ids: set[str] = set()
        documents: List[Document] = []
        for doc, score in docs_and_scores:
            doc.metadata["score"] = score
            documents.append(doc)
            doc_id = doc.metadata.get("doc_id", None)
            if doc_id is not None:
                doc_ids.add(doc_id)

        return documents

class LangChainVectorDBChroma(LangChainVectorDB):

    def __init__(self, client: LangchainClient, vector_db_url: str, collection_name: str = "") -> None:
        self.client: LangchainClient = client
        self.vector_db_url: str = vector_db_url
        self.collection_name: str = collection_name

        # ベクトルDB用のディレクトリが存在しない場合
        if not os.path.exists(self.vector_db_url):
            # ディレクトリを作成
            os.makedirs(self.vector_db_url)
            # ディレクトリが作成されたことをログに出力
            logger.info(f"create directory:{self.vector_db_url}")
        # params
        settings = chromadb.config.Settings(anonymized_telemetry=False)

        params: dict[str, Any]= {}
        params["client"] = chromadb.PersistentClient(path=self.vector_db_url, settings=settings)
        params["embedding_function"] = self.client.embedding
        params["collection_metadata"] = {
            "hnsw:space":"cosine", 
            "hnsw:construction_ef": 400, 
            "hnsw:search_ef": 200,
            "hnsw:M": 24,
        }
        # collectionが指定されている場合
        logger.info(f"collection_name:{self.collection_name}")
        if self.collection_name:
            params["collection_name"] = self.collection_name
                    
        db: VectorStore = Chroma(
            **params
            )
        self.db = db

    def __create_filter_condition__(self, tags: dict[str, list[str]]) -> dict[str, Any]:
        conditions = []
        for name, value_list in tags.items():
            conditions.append({name: {"$in": value_list}})
        if len(conditions) > 1:
            filter = {"$and": conditions}
        elif len(conditions) == 1:
            filter = conditions[0]
        else:
            filter = {}
        return filter

    def _create_search_kwargs_(self, k: int, tags: dict[str, list[str]]) -> dict[str, Any]:
        search_kwargs: dict[str, Any] = {"k": k}
        filter = self.__create_filter_condition__(tags)
        search_kwargs["filter"] = filter
        return search_kwargs

    def _get_documents_(self, tags: dict[str, list[str]]) -> Tuple[List[str], List[Document]]:
        ids=[]
        logger.debug(f"tags:{tags}")
        # K(key).is_in(value) & 条件 & ...の形の文を作成
        conditions = self.__create_filter_condition__(tags)

        if len(conditions) > 0:
            doc_dict = self.db.get(where=conditions) # type: ignore
        else:
            doc_dict = self.db.get() # type: ignore

        # デバッグ用
        logger.debug(f"_get_document_ids_by_tag doc_dict: {doc_dict}")

        # vector idを取得してidsに追加
        ids.extend(doc_dict.get("ids", []))
        content_list = doc_dict.get("documents", [])
        metadata_list = doc_dict.get("metadatas", [])

        documents = [Document(page_content=content, metadata=metadata) for content, metadata in zip(content_list, metadata_list)]

        return ids, documents

    
class LangChainVectorDBPGVector(LangChainVectorDB):

    def __init__(self, client: LangchainClient, vector_db_url: str, collection_name: str = ""):
        self.client: LangchainClient = client
        self.vector_db_url: str = vector_db_url
        self.collection_name: str = collection_name
        # params
        params: dict[str, Any] = {}
        params["connection"] = self.vector_db_url
        params["embeddings"] = self.client.embedding
        params["use_jsonb"] = True
        
        # collectionが指定されている場合
        logger.info("collection_name:", self.collection_name)
        if self.collection_name:
            params["collection_name"] = self.collection_name

        db: VectorStore = PGVector(
            **params
            )
        self.db = db

    def _create_search_kwargs_(self, k:int, tags: dict[str, list[str]]) -> dict[str, Any]:
        search_kwargs: dict[str, Any] = {"k": k}
        if tags:
            sql_where_clauses = ""
            for name, value_list in tags.items():
                # PGVectorのfilterはSQL文にする。
                if not sql_where_clauses:
                    sql_where_clauses += f"cmetadata->>'{name}' = ANY(ARRAY{value_list}::text[])"
                else:
                    sql_where_clauses += f" AND cmetadata->>'{name}' = ANY(ARRAY{value_list}::text[])"
            search_kwargs["filter"] = sql_where_clauses

        return search_kwargs
    

    def _get_documents_(self, tags: dict[str, list[str]] = {}) -> Tuple[List[str], List[Document]]:
        engine = sqlalchemy.create_engine(self.vector_db_url)
        with Session(engine) as session:
            stmt = text("SELECT uuid FROM langchain_pg_collection WHERE name=:name").bindparams(name=self.collection_name)
            row = session.execute(stmt).fetchone()
            if not row:
                return ([], [])
            collection_id = row[0]
            logger.debug(f"collection_id: {collection_id}")

            params = {"collection_id": collection_id}
            if tags:
                where_clauses = []
                for i, (name, value_list) in enumerate(tags.items()):
                    key_name = f"name_{i}"
                    val_name = f"value_{i}"
                    where_clauses.append(f"cmetadata->>:{key_name} = ANY(:{val_name})")
                    params[key_name] = name
                    params[val_name] = value_list
                where_sql = " AND ".join(where_clauses)
                query = f"""
                    SELECT id, document, cmetadata
                    FROM langchain_pg_embedding
                    WHERE collection_id=:collection_id AND {where_sql}
                """
            else:
                query = """
                    SELECT id, document, cmetadata
                    FROM langchain_pg_embedding
                    WHERE collection_id=:collection_id
                """

            try:
                rows = session.execute(text(query), params).all()
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                return ([], [])

            document_ids = [row[0] for row in rows]
            document_list = [Document(page_content=row[1], metadata=row[2]) for row in rows]
            return (document_ids, document_list)
