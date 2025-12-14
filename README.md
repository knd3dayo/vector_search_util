# vector_search_util

## 概要

**vector_search_util** は、LangChain ベースのベクトルデータベースを簡単に操作するためのユーティリティライブラリです。  
テキストデータをベクトル化してデータベースに登録し、自然言語による類似検索を行うことができます。

このライブラリは、`loader.py` によるデータ登録機能、`search.py` による検索機能、  
および `__main__.py` によるコマンドラインインターフェースを中心に構成されています。  
また、`mcp_server.py` により MCP サーバーとしても動作可能です。

---

## 特徴

- LangChain の `Document` オブジェクトの `metadata` に **`source_id`** と **`category`** タグを設定可能。  
- `source_id` をキーとして既存ベクトルの更新が可能。  
- `category` 単位でベクトルデータの取得・削除が可能。  
- 柔軟なメタデータ管理により、データの分類・再利用が容易。  
- Excelファイルを介したデータ入出力に対応し、非エンジニアでも運用可能。

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

### ⚙️ コマンドライン実行 (`__main__.py`)
`uv run -m vector_search_util` コマンドで CLI を実行できます。

利用可能なサブコマンド:
- `search`: ベクトル検索
- `load_data`: ExcelからデータをロードしてベクトルDBに登録
- `unload_data`: ベクトルDBからデータをExcelにエクスポート
- `delete_data`: Excelで指定したデータを削除
- `load_category`: カテゴリ情報をExcelからDBに登録
- `unload_category`: カテゴリ情報をDBからExcelに出力
- `delete_category`: カテゴリ情報を削除
- `load_relation`: 関連情報をExcelからDBに登録
- `unload_relation`: 関連情報をDBからExcelに出力
- `delete_relation`: 関連情報を削除

### 各コマンドのオプション

#### 🔍 search
| オプション | 説明 |
|-------------|------|
| `-q, --query` | 検索クエリ（必須） |
| `-k, --top_k` | 取得する上位件数（デフォルト: 5） |
| `-f, --filter_file` | フィルタ条件を含むJSONファイルのパス |

#### 📥 load_data
| オプション | 説明 |
|-------------|------|
| `-i, --input_file_path` | Excelファイルのパス |
| `--content_column` | コンテンツ列名（デフォルト: content） |
| `--source_id_column` | ソースID列名（デフォルト: source_id） |
| `--category_column` | カテゴリ列名（デフォルト: category） |
| `-m, --metadata_columns` | メタデータ列名のリスト |

#### 📤 unload_data
| オプション | 説明 |
|-------------|------|
| `-o, --output_file` | 出力先Excelファイルのパス |
| `-f, --filter_file` | フィルタ条件を含むJSONファイルのパス |

#### 🗑 delete_data
| オプション | 説明 |
|-------------|------|
| `-i, --input_file_path` | 削除対象のExcelファイルパス |
| `--source_id_column` | ソースID列名（デフォルト: source_id） |
| `-f, --filter_file` | フィルタ条件を含むJSONファイルのパス |

#### 🏷 load_category
| オプション | 説明 |
|-------------|------|
| `-i, --input_file_path` | Excelファイルのパス |
| `-n, --name_column` | 名前列名（デフォルト: name） |
| `-d, --description_column` | 説明列名（デフォルト: description） |

#### 📦 unload_category
| オプション | 説明 |
|-------------|------|
| `-o, --output_file` | 出力先Excelファイルのパス |
| `-f, --filter_file` | フィルタ条件を含むJSONファイルのパス |

#### ❌ delete_category
| オプション | 説明 |
|-------------|------|
| `-i, --input_file_path` | 削除対象のExcelファイルパス |
| `-n, --name_column` | 名前列名（デフォルト: name） |

例:
```bash
uv run -m vector_search_util load_data -i data.xlsx
uv run -m vector_search_util search --query "AIとは何か？"
uv run -m vector_search_util unload_data -o output.xlsx
uv run -m vector_search_util delete_data -i delete_list.xlsx
uv run -m vector_search_util load_category -i category.xlsx
uv run -m vector_search_util unload_category -o category_out.xlsx
uv run -m vector_search_util delete_category -i category_delete.xlsx
uv run -m vector_search_util load_relation -i relation.xlsx
uv run -m vector_search_util unload_relation -o relation_out.xlsx
uv run -m vector_search_util delete_relation -i relation_delete.xlsx
```

### 🔗 relation関連コマンド

`relation` は **カテゴリ間の関連性（関係性）を管理** するための機能です。  
たとえば、「製品カテゴリ」と「技術カテゴリ」のように、異なるカテゴリ間の関係を定義・検索・削除できます。  
これにより、知識グラフ的な構造をベクトルDB上で表現することが可能になります。

#### 📥 load_relation
| オプション | 説明 |
|-------------|------|
| `-i, --input_file_path` | Excelファイルのパス |
| `--source_column` | 関連元カテゴリ列名（デフォルト: source） |
| `--target_column` | 関連先カテゴリ列名（デフォルト: target） |
| `--type_column` | 関連タイプ列名（デフォルト: type） |

#### 📦 unload_relation
| オプション | 説明 |
|-------------|------|
| `-o, --output_file` | 出力先Excelファイルのパス |
| `-f, --filter_file` | フィルタ条件を含むJSONファイルのパス |

#### ❌ delete_relation
| オプション | 説明 |
|-------------|------|
| `-i, --input_file_path` | 削除対象のExcelファイルパス |
| `--source_column` | 関連元カテゴリ列名（デフォルト: source） |
| `--target_column` | 関連先カテゴリ列名（デフォルト: target） |
| `--type_column` | 関連タイプ列名（デフォルト: type） |

#### 💡 利用例

カテゴリ間の関係をExcelで定義し、知識構造を管理できます。

| source | target | type |
|---------|---------|------|
| AI | 機械学習 | includes |
| 機械学習 | 深層学習 | includes |
| AI | 自然言語処理 | related_to |

このような関係を登録することで、検索時に関連カテゴリをたどる高度な検索が可能になります。

---

## APIサーバー

`vector_search_util` は REST API サーバーとしても動作します。  
外部アプリケーションやサービスから HTTP 経由でベクトル検索・データ登録を行うことができます。

### 起動方法

```bash
uv run -m vector_search_util.api.api_server
```

### 主なエンドポイント

| メソッド | エンドポイント | 説明 |
|-----------|----------------|------|
| `POST` | `/api/load_data` | データをベクトルDBに登録 |
| `POST` | `/api/search` | クエリに基づくベクトル検索 |
| `POST` | `/api/delete_data` | 指定データを削除 |
| `GET` | `/api/health` | サーバーの稼働確認 |

### 例

```bash
curl -X POST http://localhost:8000/api/search -H "Content-Type: application/json" -d '{"query": "AIとは何か？"}'
```

---

## MCPサーバー

`vector_search_util` は MCP (Model Context Protocol) サーバーとしても動作します。  
外部ツールやAIエージェントからベクトル検索機能を利用可能です。

### 起動方法

```bash
uv run -m vector_search_util.mcp.mcp_server
```

### 主な機能
- ベクトル検索APIの提供
- LangChainベースの埋め込み生成
- OpenAIまたはAzure OpenAIを利用したLLM連携
- MCPツール経由での外部連携（例：AIChatUtil, WebSearchUtil など）

---

## ディレクトリ構成

```
src/vector_search_util/
├── langchain/        # LangChainクライアント・ベクトルDB関連
├── llm/              # 埋め込みモデル設定
├── log/              # ログ設定
├── mcp/              # MCPサーバー実装
├── util/             # loader.py, search.py などのユーティリティ
└── __main__.py       # CLIエントリーポイント
```

---

## インストール

### 前提条件

- [Python 3.10+](https://www.python.org/downloads/)
- [uv](https://github.com/astral-sh/uv)

### セットアップ手順

```bash
uv sync
```

これにより、`pyproject.toml` に基づいて仮想環境が構築され、依存パッケージがインストールされます。

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

このプロジェクトでは `.env` ファイルを使用して環境変数を管理します。  
`.env_template` を参考に `.env` ファイルを作成してください。

例：

```
APP_DATA_PATH=work/app_data
VECTOR_DB_TYPE=chroma
VECTOR_DB_URL=work/chroma_db
VECTOR_DB_COLLECTION_NAME=sample_collection
SOURCE_ID_KEY=source_id
CATEGORY_KEY=category
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_COMPLETION_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# OPENAI_BASE_URL=
# AZURE_OPENAI_API_VERSION=
# AZURE_OPENAI_ENDPOINT=
```

### 主な環境変数の説明

| 変数名 | 説明 |
|--------|------|
| `APP_DATA_PATH` | SQLiteなどのアプリデータ保存先 |
| `VECTOR_DB_TYPE` | 使用するベクトルDBの種類（`chroma` または `pgvector`） |
| `VECTOR_DB_URL` | ベクトルDBの保存先または接続URL |
| `VECTOR_DB_COLLECTION_NAME` | コレクション名 |
| `SOURCE_ID_KEY` | データ登録時のソースIDキー |
| `CATEGORY_KEY` | データ登録時のカテゴリキー |
| `LLM_PROVIDER` | 使用するLLMプロバイダ（`openai` または `azure_openai`） |
| `OPENAI_API_KEY` | OpenAI APIキー |
| `OPENAI_COMPLETION_MODEL` | OpenAIの生成モデル名 |
| `OPENAI_EMBEDDING_MODEL` | 埋め込みモデル名 |
| `OPENAI_BASE_URL` | OpenAI互換エンドポイント（任意） |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAIのAPIバージョン |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAIのエンドポイント |

---

## 使用例

### Excelデータの登録

```bash
uv run -m vector_search_util load_data -f data.xlsx -c content -i source_id -m category author
```

### ベクトル検索の実行

```bash
uv run -m vector_search_util search --query "AIとは何か？" -k 5 -f filter.json
```
