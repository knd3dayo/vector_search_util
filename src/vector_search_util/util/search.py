import argparse
from vector_search_util.llm.embedding_config import EmbeddingConfig
from vector_search_util.langchain.langchain_client import LangchainClient
from vector_search_util.langchain.langchain_vector_db import LangChainVectorDB


def main(args: argparse.Namespace| None = None):
    if args is None:
        parser = argparse.ArgumentParser(description="Search vectors from the vector DB.")
        parser.add_argument("-q", "--query", type=str, required=True, help="Search query text.")
        parser.add_argument("-k", "--top_k", type=int, default=5, help="Number of top results to return.")
        parser.add_argument("-f", "--filter_file", type=str, help="Path to JSON file containing filter conditions.")
        args = parser.parse_args()
    
    # フィルタの読み込み
    import json, os
    filter_data = {}
    if args.filter_file and os.path.exists(args.filter_file):
        with open(args.filter_file, "r", encoding="utf-8") as f:
            filter_data = json.load(f)
        print(f"Loaded filter from {args.filter_file}: {filter_data}")

    # ベクトルDBの初期化
    config = EmbeddingConfig()
    client = LangchainClient.create_client(config)
    vector_db = LangChainVectorDB.create_vector_db(client)

    # 検索実行
    print(f"Searching for: {args.query}")
    import asyncio
    results = asyncio.run(vector_db.vector_search(args.query, args.top_k, filter_data))

    # 結果出力
    print("\n=== Search Results ===")
    for i, doc in enumerate(results, start=1):
        print(f"[{i}] {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 40)


if __name__ == "__main__":
    main()
