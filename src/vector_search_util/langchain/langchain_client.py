
import os

# 抽象クラス
from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings

from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from vector_search_util.llm.embedding_config import EmbeddingConfig

import vector_search_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class LangchainClient:

    llm_config: EmbeddingConfig = EmbeddingConfig()
    embedding: Embeddings | None = None

    @classmethod
    def create_client(cls, llm_config: EmbeddingConfig) -> 'LangchainClient':

        if llm_config.llm_provider == "openai":
            client = LangchainOpenAIClient(llm_config)
            return client
        elif llm_config.llm_provider == "azure_openai":
            client = LangchainAzureOpenAIClient(llm_config)
            return client
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.llm_provider}")

        
class LangchainOpenAIClient(LangchainClient):

    def __init__(self, llm_config: EmbeddingConfig|None = None) -> None:
        if llm_config:
            self.llm_config = llm_config

        params = {}
        params["api_key"] = self.llm_config.api_key
        if self.llm_config.base_url:
            params["base_url"] = self.llm_config.base_url

        params["model"] = self.llm_config.embedding_model
        self.embedding = OpenAIEmbeddings(
                **params
            )
        
class LangchainAzureOpenAIClient(LangchainClient):

    def __init__(self, llm_config: EmbeddingConfig|None = None) -> None:
        if llm_config:
            self.llm_config = llm_config

        params = {}
        params["api_key"] = self.llm_config.api_key
        if self.llm_config.endpoint:
            params["azure_endpoint"] = self.llm_config.endpoint
        if self.llm_config.api_version:
            params["api_version"] = self.llm_config.api_version

        params["model"] = self.llm_config.embedding_model

        self.embedding = AzureOpenAIEmbeddings(
                **params
            )