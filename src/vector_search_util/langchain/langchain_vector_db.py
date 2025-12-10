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
from vector_search_util.model.models import EmbeddingData

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
    def _get_document_ids_by_tag_(self, name: str = "", value: str = "") -> Tuple[List[str], List[dict[str, Any]]]:
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
    async def add_documents(self, data_list: list[EmbeddingData]):

        if self.db is None:
            raise ValueError("db is None")
 
        documents: list[Document] = []
        for data in data_list:
            # metadataにsource_idがない場合は追加する
            if "source_id" not in data.metadata.keys():
                data.metadata["source_id"] = data.source_id

            # Documentを作成
            document = Document(
                page_content=data.content,
                metadata=data.metadata
            )
            documents.append(document)
        await self.add_doucment_with_retry(self.db, documents)

    async def delete_documents(self, doc_ids:list=[]):
        if len(doc_ids) == 0:
            return
        if self.db is None:
            raise ValueError("db is None")

        await self.db.adelete(ids=doc_ids)

        return len(doc_ids)    

    def delete_collection(self):
        # self.dbがdelete_collectionメソッドを持っている場合はそれを呼び出す
        if hasattr(self.db, "delete_collection"):
            self.db.delete_collection() # type: ignore


    async def delete_documents_by_tag(self, tag_name: str, tag_value: str):
        # ベクトルDB固有のvector id取得メソッドを呼び出し。
        vector_ids, _ = self._get_document_ids_by_tag_(tag_name, tag_value)

        # vector_idsが空の場合は何もしない
        if len(vector_ids) == 0:
            return 0

        # ベクトルDB固有の削除メソッドを呼び出し
        await self.delete_documents(vector_ids)

    async def update_embeddings(self, data_list: list[EmbeddingData]):
        
        # 既に存在するドキュメントを削除
        await self.delete_documents([data.source_id for data in data_list])
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

    async def vector_search(self, query: str, k: int, filter: dict[str, Any]= {}) -> List[Document]:
        """
        ベクトルDBからドキュメントを検索する。
        :param query: 検索クエリ
        :param search_kwargs: 検索キーワード
        :return: 検索結果のドキュメントリスト
        """
        if self.db is None:
            raise ValueError("db is None")

        search_kwargs: dict[str, Any] = {"k": k}
        if filter:
            search_kwargs["filter"] = filter

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

    def _get_document_ids_by_tag_(self, name:str="", value:str="") -> Tuple[List, List]:
        ids=[]
        metadata_list = []
        doc_dict = self.db.get(where={name: value}) # type: ignore

        # デバッグ用
        logger.debug("_get_document_ids_by_tag doc_dict:", doc_dict)

        # vector idを取得してidsに追加
        ids.extend(doc_dict.get("ids", []))
        metadata_list.extend(doc_dict.get("metadata", []))

        return ids, metadata_list

    
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

    def _get_document_ids_by_tag_(self, name:str="", value:str="") -> Tuple[List, List]:
        engine = sqlalchemy.create_engine(self.vector_db_url)
        with Session(engine) as session:
            stmt = text("select uuid from langchain_pg_collection where name=:name")
            stmt = stmt.bindparams(name=self.collection_name)
            rows  = session.execute(stmt).fetchone()
            if rows is None or len(rows) == 0:
                return [], []
            collection_id = rows[0]
            logger.info(collection_id)
            stmt = text("select id, cmetadata from langchain_pg_embedding where collection_id=:collection_id and cmetadata->>:name=:value")
            stmt = stmt.bindparams(collection_id=collection_id, name=name, value=value)
            rows2: Sequence[Any] = session.execute(stmt).all() 
            document_ids = [row[0] for row in rows2]
            metadata_list = [row[1] for row in rows2]
            
            return document_ids, metadata_list


