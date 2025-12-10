# vector_search_util

## 概要

**vector_search_util** は、LangChain ベースのベクトルデータベースを簡単に操作するためのユーティリティライブラリです。  
テキストデータをベクトル化してデータベースに登録し、自然言語による類似検索を行うことができます。

このライブラリは、`loader.py` によるデータ登録機能と、`search.py` による検索機能を中心に構成されています。  
Chroma または PostgreSQL (PGVector) をバックエンドとして利用可能です。

---

## 主な機能

### 📥 データ登録 (`loader.py`)
- Excelファイルからテキストデータを読み込み、ベクトルDBに登録。
- 各行を `EmbeddingData` として処理し、非同期で登録。
- メタデータを含む柔軟なデータ構造に対応。

### 🔍 ベクトル検索 (`search.py`)
- コマンドラインから自然言語クエリでベクトル検索を実行。
- 検索結果をスコア付きで出力。
- JSONファイルでフィルタ条件を指定可能。

---

## ディレクトリ構成

```
src/vector_search_util/
├── langchain/        # LangChainクライアント・ベクトルDB関連
├── llm/              # 埋め込みモデル設定
├── log/              # ログ設定
├── mcp/              # MCPサーバー実装
├── model/            # データモデル定義
└── util/             # loader.py, search.py などのユーティリティ
```

---

## インストール

### 前提条件

このプロジェクトを利用するには、以下がインストールされている必要があります。

- [Python 3.10+](https://www.python.org/downloads/)
- [uv](https://github.com/astral-sh/uv)

### セットアップ手順

リポジトリをクローンした後、依存関係を同期します。

```bash
uv sync
```

これにより、`pyproject.toml` に基づいて仮想環境が自動的に構築され、必要なパッケージがインストールされます。

---

## 依存関係

主要な依存パッケージは `requirements.txt` に記載されています。  
例：
```
langchain
chromadb
sqlalchemy
pandas
tqdm
```

---

## 環境変数設定

このプロジェクトでは、`.env` ファイルを使用して環境変数を管理します。  
`.env_template` を参考に `.env` ファイルを作成してください。

例：

```
VECTOR_DB_TYPE=chroma
VECTOR_DB_URL=./vector_db
VECTOR_DB_COLLECTION_NAME=default_collection
OPENAI_API_KEY=your_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### 主な環境変数の説明

| 変数名 | 説明 |
|--------|------|
| `VECTOR_DB_TYPE` | 使用するベクトルDBの種類（`chroma` または `pgvector`） |
| `VECTOR_DB_URL` | ベクトルDBの保存先または接続URL |
| `VECTOR_DB_COLLECTION_NAME` | コレクション名 |
| `OPENAI_API_KEY` | OpenAI APIキー |
| `OPENAI_EMBEDDING_MODEL` | 埋め込みモデル名 |

---

## 使用例

### Excelデータの登録

```bash
uv run -m vector_search_util load -f data.xlsx -c content -i source_id -m category author
```

- `-f`: Excelファイルのパス  
- `-c`: コンテンツ列名（デフォルト: `content`）  
- `-i`: ソースID列名（デフォルト: `source_id`）  
- `-m`: メタデータ列名のリスト  

---

### ベクトル検索の実行

```bash
uv run -m vector_search_util search --query "AIとは何か？" -k 5 -f filter.json
```

- `-q`: 検索クエリ  
- `-k`: 取得する上位件数（デフォルト: 5）  
- `-f`: フィルタ条件を記載したJSONファイルのパス  

#### filter.json の例

```json
{
  "category": "technology",
  "author": "John Doe"
}
```

---

## ライセンス

このプロジェクトは [MIT License](LICENSE) のもとで公開されています。

---

## リポジトリ

GitHub: [https://github.com/knd3dayo/vector_search_util](https://github.com/knd3dayo/vector_search_util)
