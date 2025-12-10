import asyncio
from tqdm.asyncio import tqdm_asyncio

from vector_search_util.langchain.langchain_vector_db import LangChainVectorDB
from vector_search_util.model.models import EmbeddingData
from vector_search_util.llm.embedding_config import EmbeddingConfig
from vector_search_util.langchain.langchain_client import LangchainClient
import pandas as pd
from pandas import DataFrame
import argparse


class EmbeddingBatchClient:
    def __init__(self, vector_db: LangChainVectorDB):
        self.vector_db = vector_db

    async def _process_row_(self, row_num: int, data: EmbeddingData, progress: tqdm_asyncio) -> int:
        await self.vector_db.add_documents([data])
        progress.update(1)
        return row_num

    async def run(self, data_list: list[EmbeddingData]):
        progress = tqdm_asyncio(total=len(data_list), desc="progress")
        progress.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

        async with asyncio.Semaphore(5):
            tasks = [self._process_row_(i, data, progress) for i, data in enumerate(data_list)]
            await asyncio.gather(*tasks)
            progress.close()

    def create_data_from_dataframe(
        self, df: DataFrame, content_column: str, source_id_column: str, metadata_columns: list[str]
    ) -> list[EmbeddingData]:
        data_list: list[EmbeddingData] = []
        for _, row in df.iterrows():
            content = row.get(content_column, "")
            source_id = row.get(source_id_column, "")
            if not content or not source_id:
                continue
            metadata = {key: row.get(key, "") for key in metadata_columns}
            data = EmbeddingData(content=str(content), source_id=str(source_id), metadata=metadata)
            data_list.append(data)
        return data_list

    def load_data_from_excel(
        self, file_path: str, content_column: str, source_id_column: str, metadata_columns: list[str]
    ) -> list[EmbeddingData]:
        df = pd.read_excel(file_path)
        return self.create_data_from_dataframe(df, content_column, source_id_column, metadata_columns)


def main(args: argparse.Namespace| None = None):
    if args is None:
        parser = argparse.ArgumentParser(description="Load data from Excel file and add to vector DB.")
        parser.add_argument("-f", "--file_path", type=str, help="Path to the Excel file.")
        parser.add_argument("-c", "--content_column", type=str, default="content", help="Name of the content column. default is 'content'.")
        parser.add_argument("-i", "--source_id_column", type=str, default="source_id", help="Name of the source_id column.")
        parser.add_argument("-m", "--metadata_columns", type=str, nargs="*", default=[], help="List of metadata column names.")
        args = parser.parse_args()

    config = EmbeddingConfig()
    client = LangchainClient.create_client(config)
    vector_db = LangChainVectorDB.create_vector_db(client)
    batch_client = EmbeddingBatchClient(vector_db)

    data_list = batch_client.load_data_from_excel(
        args.file_path, args.content_column, args.source_id_column, args.metadata_columns
    )
    asyncio.run(batch_client.run(data_list))
    print(f"Loaded {len(data_list)} records from {args.file_path}.")


if __name__ == "__main__":
    main()
