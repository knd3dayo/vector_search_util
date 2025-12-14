import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Any, Callable
from datetime import datetime, timezone
from langchain_core.documents import Document

import vector_search_util._internal.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)
from typing import Optional

class EmbeddingConfig:

    def __init__(self):
        load_dotenv()

        # metadata用のキー設定
        self.source_id_key: str = os.getenv("SOURCE_ID_KEY","source_id")
        self.source_content_key: str = os.getenv("SOURCE_CONTENT_KEY","source_content")
        self.category_key: str = os.getenv("CATEGORY_KEY","category")
        self.updated_at_key: str = os.getenv("UPDATED_AT_KEY","updated_at")
        self.sequence_id_key: str = os.getenv("SEQUENCE_ID_KEY","sequence_id")

        # ベクトル化する際にSOURCE_CONTENTを分割する。その際のチャンクサイズ
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE","4000"))

        # 並列度の設定
        self.concurrency: int = int(os.getenv("EMBEDDING_CONCURRENCY","16"))
        
        self.app_data_path: str = os.getenv("APP_DATA_PATH","work/app_data")

        
        self.vector_db_type: str = os.getenv("VECTOR_DB_TYPE","chroma")
        self.vector_db_url: str = os.getenv("VECTOR_DB_URL", "work/chroma_db")
        self.vector_db_collection_name: str = os.getenv("VECTOR_DB_COLLECTION_NAME","")
        self.llm_provider: str = os.getenv("LLM_PROVIDER","openai")
        self.api_key: str = ""
        self.completion_model: str = ""
        self.embedding_model: str = ""
        self.api_version: Optional[str] = None
        self.endpoint: Optional[str] = None

        self.base_url: Optional[str] = None
        if self.llm_provider == "openai" or self.llm_provider == "azure_openai":
            self.api_key = os.getenv("OPENAI_API_KEY","")
            self.base_url = os.getenv("OPENAI_BASE_URL","") or None
            self.completion_model: str = os.getenv("OPENAI_COMPLETION_MODEL", "gpt-4o")
            self.embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        if self.llm_provider == "azure_openai":
            self.api_version: Optional[str] = os.getenv("AZURE_OPENAI_API_VERSION","")
            self.endpoint: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT","")


# source_document_data
class SourceDocumentData(BaseModel):
    embedding_config: ClassVar[EmbeddingConfig | None]  = None
    source_id: str
    source_content: str
    extended_properties : dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def get_embedding_config(cls) -> EmbeddingConfig:
        if cls.embedding_config is None:
            cls.embedding_config = EmbeddingConfig()
        return cls.embedding_config

    @classmethod
    def from_embedding_data(cls, embedding_data: "EmbeddingData") -> "SourceDocumentData":
        # source_id, categoryをmetadataに含める
        source_id_key = cls.get_embedding_config().source_id_key
        category_key = cls.get_embedding_config().category_key
        updated_at_key = cls.get_embedding_config().updated_at_key

        metadata = embedding_data.metadata.copy()
        metadata[source_id_key] = embedding_data.source_id
        metadata[category_key] = embedding_data.category
        metadata[updated_at_key] = embedding_data.updated_at.isoformat()

        return SourceDocumentData(
            source_id=embedding_data.source_id,
            source_content=embedding_data.source_content,
            extended_properties=metadata
        )

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


class EmbeddingData(BaseModel):

    embedding_config: ClassVar[EmbeddingConfig | None]  = None

    source_id: str
    source_content: str
    category: str = ""
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata : dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def to_langchain_documents(cls, data_list: list["EmbeddingData"]) -> list[Document]:
        """Convert to Langchain Document."""
        documents: list[Document] = []
        for data in data_list:
            docs = cls.__to_langchain_documents_from_single__(data)
            documents.extend(docs)
        return documents


    @classmethod
    def __to_langchain_documents_from_single__(cls, data: "EmbeddingData") -> list[Document]:
        """Convert to Langchain Document."""
        if EmbeddingData.embedding_config is None:
            EmbeddingData.embedding_config = EmbeddingConfig()
        documents: list[Document] = []

        # chunkingに基づいて、source_contentを分割する
        chunk_size = EmbeddingData.embedding_config.chunk_size

        updated_at_str = data.updated_at.isoformat()

        page_countents = [data.source_content[i:i+chunk_size] for i in range(0, len(data.source_content), chunk_size)]

        for i in range(len(page_countents)):
            metadata = data.metadata.copy()
            page_content = page_countents[i]
            # metadataにsequence_idを追加
            doc = Document(
                page_content=page_content,
                metadata={
                    EmbeddingData.embedding_config.source_id_key: data.source_id,
                    EmbeddingData.embedding_config.category_key: data.category,
                    EmbeddingData.embedding_config.updated_at_key: updated_at_str,
                    EmbeddingData.embedding_config.sequence_id_key: i,
                    **metadata
                }
            )
            documents.append(doc)
        return documents

    @classmethod
    def from_langchain_documents(cls, documents: list[Document], get_source_content_function: Callable) -> list["EmbeddingData"]:
        if EmbeddingData.embedding_config is None:
            EmbeddingData.embedding_config = EmbeddingConfig()

        first_sequence_docs = [doc for doc in documents if doc.metadata.get(EmbeddingData.embedding_config.sequence_id_key, 0) == 0]
        data_list: list[EmbeddingData] = []
        for doc in first_sequence_docs:
            data = EmbeddingData.__from_langchain_document__(doc, get_source_content_function)
            data_list.append(data)
        return data_list
    
    @classmethod
    def __from_langchain_document__(cls, document: Document, get_source_content_function: Callable[[str], str]) -> "EmbeddingData":
        """Convert from Langchain Document."""
        if EmbeddingData.embedding_config is None:
            EmbeddingData.embedding_config = EmbeddingConfig()
        source_id = document.metadata.get(EmbeddingData.embedding_config.source_id_key, "")
        category = document.metadata.get(EmbeddingData.embedding_config.category_key, "")

        updated_at_str = document.metadata.get(EmbeddingData.embedding_config.updated_at_key, None)
        if updated_at_str:
            updated_at = datetime.fromisoformat(updated_at_str)
        else:
            updated_at = datetime.now(timezone.utc)

        data = EmbeddingData(
            source_id=source_id,
            source_content=get_source_content_function(source_id),
            category=category,
            updated_at=updated_at,
            metadata=document.metadata
        )
        return data
from abc import ABC, abstractmethod
from typing import Union, Literal, Annotated, Any
from pydantic import BaseModel, Field

class Condition(ABC, BaseModel):
    """Base class for all conditions."""
    @abstractmethod
    def build(self):
        raise NotImplementedError

class EqCondition(Condition):
    field: str = Field(..., description="The field to compare.")
    value: Any = Field(..., description="The value to compare against.")

    def build(self):
        return {self.field: self.value}


class InCondition(Condition):
    field: str = Field(..., description="The field to compare.")
    values: list[Any] = Field(..., description="The list of values to compare against.")

    def build(self):
        return {self.field: {"$in": self.values}}


class ContainsCondition(Condition):
    """MongoDB の部分一致（正規表現）"""
    field: str = Field(..., description="The field to compare.")
    substring: str = Field(..., description="The substring to search for.")

    def build(self):
        return {self.field: {"$regex": self.substring}}


# -------------------------
# 比較条件 ($gte, $lte, $gt, $lt)
# -------------------------

class CompareCondition(Condition):
    field: str = Field(..., description="The field to compare.")
    operator: str = Field(..., description="The comparison operator.")
    value: Any = Field(..., description="The value to compare against.")
    def build(self):
        return {self.field: {self.operator: self.value}}


# -------------------------
# 論理条件 ($and, $or, $not)
# -------------------------

class AndCondition(Condition):
    conditions: list[Condition] = Field(..., description="List of conditions")

    def build(self):
        return {"$and": [c.build() for c in self.conditions]}


class OrCondition(Condition):
    conditions: list[Condition] = Field(..., description="List of conditions")

    def build(self):
        return {"$or": [c.build() for c in self.conditions]}


class NotCondition(Condition):
    condition: Condition = Field(..., description="The condition to negate")

    def build(self):
        # NOT は {field: {"$not": {...}}} の形にする必要がある
        built = self.condition.build()
        field, expr = list(built.items())[0]
        return {field: {"$not": expr}}

# -------------------------
# Query Builder
# -------------------------

class ConditionContainer(BaseModel):
    conditions: list[
        Union[
            EqCondition, InCondition, ContainsCondition, CompareCondition, AndCondition, OrCondition, NotCondition
            ]
        ] = Field(default_factory=list, description="List of conditions")

    # --- 基本条件 ---
    def add_eq_condition(self, field, value):
        self.conditions.append(EqCondition(field=field, value=value))
        return self

    def add_in_condition(self, field, values):
        self.conditions.append(InCondition(field=field, values=values))
        return self

    def add_contains_condition(self, field, substring):
        self.conditions.append(ContainsCondition(field=field, substring=substring))
        return self

    # --- 比較条件 ---
    def add_gte_condition(self, field, value):
        self.conditions.append(CompareCondition(field=field, operator="$gte", value=value))
        return self

    def add_lte_condition(self, field, value):
        self.conditions.append(CompareCondition(field=field, operator="$lte", value=value))
        return self

    def add_gt_condition(self, field, value):
        self.conditions.append(CompareCondition(field=field, operator="$gt", value=value))
        return self

    def add_lt_condition(self, field, value):
        self.conditions.append(CompareCondition(field=field, operator="$lt", value=value))
        return self

    # --- 論理条件 ---
    def add_and_condition(self, conditions):
        self.conditions.append(AndCondition(conditions=conditions))
        return self

    def add_or_condition(self, conditions):
        self.conditions.append(OrCondition(conditions=conditions))
        return self

    def add_not_condition(self, condition):
        self.conditions.append(NotCondition(condition=condition))
        return self

    # --- MongoDB風 dict 生成 ---
    def build(self):
        if len(self.conditions) == 0:
            return {}
        if len(self.conditions) == 1:
            return self.conditions[0].build()
        return {"$and": [c.build() for c in self.conditions]}

    # --- PostgreSQL JSONB SQL 生成 ---
    def to_postgres_sql(self):
        if len(self.conditions) == 0:
            return ""

        translator = PostgresJsonbTranslator()
        return translator.translate(self.build())
    

class PostgresJsonbTranslator:
    def translate(self, condition_dict):
        return self._translate_dict(condition_dict)

    def _translate_dict(self, d):
        clauses = []
        for key, value in d.items():
            if key == "$and":
                sub = [self._translate_dict(v) for v in value]
                clauses.append("(" + " AND ".join(sub) + ")")
            elif key == "$or":
                sub = [self._translate_dict(v) for v in value]
                clauses.append("(" + " OR ".join(sub) + ")")
            else:
                clauses.append(self._translate_field(key, value))
        return " AND ".join(clauses)

    def _translate_field(self, field, expr):
        if isinstance(expr, dict):
            if "$in" in expr:
                vals = ",".join([f"'{v}'" for v in expr["$in"]])
                return f"(cmetadata->>'{field}') IN ({vals})"

            if "$regex" in expr:
                return f"(cmetadata->>'{field}') LIKE '%{expr['$regex']}%'"

            if "$gte" in expr:
                return f"(cmetadata->>'{field}')::numeric >= {expr['$gte']}"

            if "$lte" in expr:
                return f"(cmetadata->>'{field}')::numeric <= {expr['$lte']}"

            if "$gt" in expr:
                return f"(cmetadata->>'{field}')::numeric > {expr['$gt']}"

            if "$lt" in expr:
                return f"(cmetadata->>'{field}')::numeric < {expr['$lt']}"

            if "$not" in expr:
                inner = self._translate_field(field, expr["$not"])
                return f"NOT ({inner})"

        # eq
        return f"(cmetadata->>'{field}') = '{expr}'"