import asyncio
import argparse
from dotenv import load_dotenv
from fastmcp import FastMCP
from vector_search_util.api.api_server import (
    vector_search,
    get_documents,
    upsert_documents,
    delete_documents,
    get_categories,
    upsert_categories,
    delete_categories,
    get_relations,
    upsert_relations,
    delete_relations,
    get_tags,
    upsert_tags,
    delete_tags,
)

mcp = FastMCP()

# 引数解析用の関数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP server with specified mode and APP_DATA_PATH.")
    # -m オプションを追加
    parser.add_argument("-m", "--mode", choices=["sse", "http", "stdio"], default="stdio", help="Mode to run the server in: 'http' for HTTP server, 'stdio' for standard input/output.")
    # 引数を解析して返す
    # -t tools オプションを追加 toolsはカンマ区切りの文字列. search_wikipedia_ja_mcp, vector_search, etc. 指定されていない場合は空文字を設定
    parser.add_argument("-t", "--tools", type=str, default="", help="Comma-separated list of tools to use, e.g., 'search_wikipedia_ja_mcp,vector_search_mcp'. If not specified, no tools are loaded.")
    # -p オプションを追加　ポート番号を指定する modeがhttpの場合に使用.defaultは5001
    parser.add_argument("-p", "--port", type=int, default=5001, help="Port number to run the server on. Default is 5001.")
    # -v LOG_LEVEL オプションを追加 ログレベルを指定する. デフォルトは空白文字
    parser.add_argument("-v", "--log_level", type=str, default="", help="Log level to set for the server. Default is empty, which uses the default log level.")

    return parser.parse_args()

async def main():
    # load_dotenv() を使用して環境変数を読み込む
    load_dotenv()
    # 引数を解析
    args = parse_args()
    mode = args.mode

    # tools オプションが指定されている場合は、ツールを登録
    if args.tools:
        tools = [tool.strip() for tool in args.tools.split(",")]
        for tool_name in tools:
            # tool_nameという名前の関数が存在する場合は登録
            tool = globals().get(tool_name)
            if tool and callable(tool):
                mcp.tool()(tool)
            else:
                print(f"Warning: Tool '{tool_name}' not found or not callable. Skipping registration.")
    else:
        # デフォルトのツールを登録
        mcp.tool()(vector_search)
        mcp.tool()(get_documents)
        mcp.tool()(upsert_documents)
        mcp.tool()(delete_documents)
        mcp.tool()(get_categories)
        mcp.tool()(upsert_categories)
        mcp.tool()(delete_categories)
        mcp.tool()(get_relations)
        mcp.tool()(upsert_relations)
        mcp.tool()(delete_relations)
        mcp.tool()(get_tags)
        mcp.tool()(upsert_tags)
        mcp.tool()(delete_tags)

    if mode == "stdio":
        await mcp.run_async()

    elif mode == "sse":
        # port番号を取得
        port = args.port
        await mcp.run_async(transport="sse", host="0.0.0.0", port=port)

    elif mode == "http":
        # port番号を取得
        port = args.port
        await mcp.run_async(transport="streamable-http", host="0.0.0.0", port=port)


if __name__ == "__main__":
    asyncio.run(main())
