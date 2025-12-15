from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Optional
import asyncio, os, json

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
from vector_search_util.model import ConditionContainer


from vector_search_util._internal.langchain.langchain_client import LangchainClient
import vector_search_util._internal.log.log_settings as log_settings
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
    def _get_documents_(self, conditions: ConditionContainer = ConditionContainer()) -> Tuple[List[str], List[Document]]:
        pass

    def _create_search_kwargs_(self, k: int, conditions: ConditionContainer = ConditionContainer()) -> dict[str, Any]:
        search_kwargs: dict[str, Any] = {"k": k}
        filter = conditions.build()
        if filter:
            search_kwargs["filter"] = filter
        return search_kwargs

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
    async def get_documents(self, conditions: ConditionContainer = ConditionContainer()) -> Tuple[List[str], List[Document]]:

        return self._get_documents_(conditions)
    

    async def add_documents(self, documents: list[Document]) -> bool:

        if self.db is None:
            raise ValueError("db is None")
 
        await self.add_doucment_with_retry(self.db, documents)
        return True

    async def delete_documents_by_ids(self, doc_ids:list=[]):
        if len(doc_ids) == 0:
            return
        if self.db is None:
            raise ValueError("db is None")

        await self.db.adelete(ids=doc_ids)

        return len(doc_ids)    

    async def delete_documents_by_tags(self, conditions: ConditionContainer = ConditionContainer()):
        # ベクトルDB固有のvector id取得メソッドを呼び出し。
        vector_ids, _ = self._get_documents_(conditions)

        # vector_idsが空の場合は何もしない
        if len(vector_ids) == 0:
            return
        await self.delete_documents_by_ids(vector_ids)

    async def upsert_documents(self, data_list: list[Document], append_vectors: bool = False) -> bool:
        
        # 既に存在するドキュメントを削除
        conditions = ConditionContainer().add_in_condition(
            self.client.llm_config.source_id_key, [ data.metadata.get(self.client.llm_config.source_id_key, "") for data in data_list ]
            )
        if not append_vectors:
            await self.delete_documents_by_tags(conditions)
        else:
            logger.info("append_vectors is True. skip delete existing documents.")
            existing_ids, exsisting_docs = await self.get_documents(conditions)
            
        # ドキュメントを格納する。
        await self.add_documents(data_list)

        return True

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

    async def vector_search(self, query: str, category: str = "", conditions: ConditionContainer = ConditionContainer(), k: int = 5) -> List[Document]:
        """
        ベクトルDBからドキュメントを検索する。
        :param query: 検索クエリ
        :param search_kwargs: 検索キーワード
        :return: 検索結果のドキュメントリスト
        """
        if self.db is None:
            raise ValueError("db is None")

        # categoryが指定されている場合はconditionsに追加
        if category:
            category_key = self.client.llm_config.category_key
            conditions.add_in_condition(category_key, [category])

        search_kwargs: dict[str, Any] = self._create_search_kwargs_(k, conditions)

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


    def _get_documents_(self, conditions: ConditionContainer = ConditionContainer()) -> Tuple[List[str], List[Document]]:
        ids=[]
        logger.debug(f"conditions:{conditions}")
        
        condition_dict = conditions.build()
        if condition_dict:
            doc_dict = self.db.get(where=condition_dict) # type: ignore
        else:
            doc_dict = self.db.get() # type: ignore

        # デバッグ用
        logger.debug(f"_get_document_ids_by_tag doc_dict: {doc_dict}")

        # vector idを取得してidsに追加
        ids.extend(doc_dict.get("ids", []))
        content_list = doc_dict.get("documents", [])
        metadata_list: list[dict[str, Any]] = doc_dict.get("metadatas", [])
        source_id_key = self.client.llm_config.source_id_key
        category_key = self.client.llm_config.category_key

        documents = [
            Document(
                page_content=content, 
                metadata=metadata
            )  for content, metadata in zip(content_list, metadata_list)
        ]   
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

    def _get_documents_(self, conditions: Optional[ConditionContainer] = None) -> Tuple[List[str], List[Document]]:
        engine = sqlalchemy.create_engine(self.vector_db_url)
        with Session(engine) as session:
            stmt = text("SELECT uuid FROM langchain_pg_collection WHERE name=:name").bindparams(name=self.collection_name)
            row = session.execute(stmt).fetchone()
            if not row:
                return ([], [])
            collection_id = row[0]
            logger.debug(f"collection_id: {collection_id}")

            params = {"collection_id": collection_id}
            if conditions:
                where_sql = conditions.to_postgres_sql()
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

            documents: list[Document] = []
            ids: list[str] = []
            for row in rows:
                ids.append(row[0])
                content = row[1]
                cmetadata_dict = json.loads(row[2])
                doc = Document(
                    page_content=content, 
                    metadata=cmetadata_dict
                )
                documents.append(doc)

            return ids, documents