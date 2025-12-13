import argparse
import sys
import asyncio
import json
from vector_search_util.util import embedding_loader, search
from vector_search_util.util import category_loader


async def main():
    parser = argparse.ArgumentParser(
        description="Vector Search Utility CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # search サブコマンド
    search_parser = subparsers.add_parser("search", help="Execute search process")
    search_parser.add_argument("-q", "--query", type=str, required=True, help="Search query text.")
    search_parser.add_argument("-c", "--category", type=str, default="", help="Category to filter search results.")
    search_parser.add_argument("-f", "--filter_file", type=str, help="Path to JSON file containing filter conditions.")
    search_parser.add_argument("-k", "--top_k", type=int, default=5, help="Number of top results to return.")

    # load_data サブコマンド
    load_parser = subparsers.add_parser("load_data", help="Execute data loading process")
    load_parser.add_argument("-i", "--input_file_path", type=str, help="Path to the Excel file.")
    load_parser.add_argument("-m", "--metadata_columns", type=str, nargs="*", default=[], help="List of metadata column names.")
    load_parser.add_argument("--content_column", type=str, default="content", help="Name of the content column.")
    load_parser.add_argument("--source_id_column", type=str, default="source_id", help="Name of the source_id column.")
    load_parser.add_argument("--category_column", type=str, default="category", help="Category tag to filter documents.")

    # unload_data サブコマンド
    unload_parser = subparsers.add_parser("unload_data", help="Execute data unloading process")
    unload_parser.add_argument("-o", "--output_file", type=str, help="Path to output file for unloaded embeddings.")    
    unload_parser.add_argument("-f","--filter_file", type=str, help="Path to JSON file containing filter conditions.")

    # delete_data サブコマンド
    delete_parser = subparsers.add_parser("delete_data", help="Execute data deletion process")
    delete_parser.add_argument("-i", "--input_file_path", type=str, help="Path to the Excel file containing source IDs to delete.")
    delete_parser.add_argument("-f","--filter_file", type=str, help="Path to JSON file containing filter conditions.")
    delete_parser.add_argument("--source_id_column", type=str, default="source_id", help="Name of the source_id column.")

    # list_category サブコマンド
    list_category_parser = subparsers.add_parser("list_category", help="List all categories in the vector DB.")

    # load_category サブコマンド
    category_load_parser = subparsers.add_parser("load_category", help="Load categories from Excel file to vector DB.")
    category_load_parser.add_argument("-i", "--input_file_path", type=str, help="Path to the Excel file.")
    category_load_parser.add_argument("-n", "--name_column", type=str, default="name", help="Name of the name column. default is 'name'.")
    category_load_parser.add_argument("-d", "--description_column", type=str, default="description", help="Name of the description column. default is 'description'.")

    # unload_category サブコマンド
    category_unload_parser = subparsers.add_parser("unload_category", help="Unload categories from vector DB to Excel file.")
    category_unload_parser.add_argument("-o", "--output_file", type=str, help="Path to output file for unloaded categories.")
    category_unload_parser.add_argument("-f", "--filter_file", type=str, help="Path to JSON file containing filter conditions.")
    
    # delete_category サブコマンド
    category_delete_parser = subparsers.add_parser("delete_category", help="Delete categories from vector DB.")
    category_delete_parser.add_argument("-i", "--input_file_path", type=str, help="Path to the Excel file containing category names to delete.")
    category_delete_parser.add_argument("-n", "--name_column", type=str, default="name", help="Name of the name column. default is 'name'.")

    # list tag サブコマンド
    list_tag_parser = subparsers.add_parser("list_tag", help="List all tags in the vector DB.")

    # load tag サブコマンド
    tag_load_parser = subparsers.add_parser("load_tag", help="Load tags from Excel file to vector DB.")
    tag_load_parser.add_argument("-i", "--input_file_path", type=str, help="Path to the Excel file.")
    tag_load_parser.add_argument("-n", "--name_column", type=str, default="name", help="Name of the name column. default is 'name'.")
    tag_load_parser.add_argument("-d", "--description_column", type=str, default="description", help="Name of the description column. default is 'description'.")

    # unload tag サブコマンド
    tag_unload_parser = subparsers.add_parser("unload_tag", help="Unload tags from vector DB to Excel file.")
    tag_unload_parser.add_argument("-o", "--output_file", type=str, help="Path to output file for unloaded tags.")
    tag_unload_parser.add_argument("-f", "--filter_file", type=str, help="Path to JSON file containing filter conditions.")

    # delete tag サブコマンド
    tag_delete_parser = subparsers.add_parser("delete_tag", help="Delete tags from vector DB.")
    tag_delete_parser.add_argument("-i", "--input_file_path", type=str, help="Path to the Excel file containing tag names to delete.")
    tag_delete_parser.add_argument("-n", "--name_column", type=str, default="name", help="Name of the name column. default is 'name'.")

    
    args = parser.parse_args()
    print(f"Executing command: {args.command}")

    if args.command == "search":
        query = args.query
        # queryが空文字の場合はsub_parserのhelpを表示して終了
        if not query.strip():
            search_parser.print_help()
            sys.exit(1)
        category = args.category
        num_results = args.top_k
        filter_data = {}
        filter_file = args.filter_file
        filter_data = {}
        if filter_file:
            filter_data = json.load(open(filter_file, "r", encoding="utf-8"))

        results = await search.vector_search(query, category, filter_data, num_results)
        # 結果出力
        print("\n=== Search Results ===")
        for i, doc in enumerate(results, start=1):
            print(f"[{i}] {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 40)
    
    elif args.command == "load_data":
        file_path = args.input_file_path
        if not file_path:
            load_parser.print_help()
            sys.exit(1)
        content_column = args.content_column
        source_id_column = args.source_id_column
        category_column = args.category_column
        metadata_columns = args.metadata_columns
        await embedding_loader.load_documents_from_excel(file_path, content_column, source_id_column, category_column, metadata_columns)

    elif args.command == "unload_data":
        output_file = args.output_file
        if not output_file:
            unload_parser.print_help()
            sys.exit(1)
        filter_file = args.filter_file
        tags = {}
        if filter_file:
            tags = json.load(open(filter_file, "r", encoding="utf-8"))
        await embedding_loader.unload_documents_to_excel(output_file, tags)

    elif args.command == "delete_data":
        input_file_path = args.input_file_path
        if not input_file_path:
            delete_parser.print_help()
            sys.exit(1)
        source_id_column = args.source_id_column
        filter_file = args.filter_file
        tags = {}
        if filter_file:
            tags = json.load(open(filter_file, "r", encoding="utf-8"))
        await embedding_loader.delete_documents_from_excel(input_file_path, source_id_column, tags)

    elif args.command == "list_category":
        categories = await category_loader.list_categories()
        print("\n=== Categories in Vector DB ===")
        for i, cat in enumerate(categories, start=1):
            print(f"[{i}] Name: {cat.name}, Description: {cat.description}")

    elif args.command == "load_category":
        input_file_path = args.input_file_path
        if not input_file_path:
            category_load_parser.print_help()
            sys.exit(1)
        name_column = args.name_column
        description_column = args.description_column
        await category_loader.load_categories(input_file_path, name_column, description_column)

    elif args.command == "unload_category":
        output_file = args.output_file
        if not output_file:
            category_unload_parser.print_help()
            sys.exit(1)
        await category_loader.unload_categories(output_file)

    elif args.command == "delete_category":
        input_file_path = args.input_file_path
        if not input_file_path:
            category_delete_parser.print_help()
            sys.exit(1)
        name_column = args.name_column
        await category_loader.delete_categories(input_file_path, name_column)

    elif args.command == "list_tag":
        from vector_search_util.util import tag_loader
        tags = await tag_loader.list_tags()
        print("\n=== Tags in Vector DB ===")
        for i, tag in enumerate(tags, start=1):
            print(f"[{i}] Name: {tag.name}, Description: {tag.description}")
    elif args.command == "load_tag":
        from vector_search_util.util import tag_loader
        input_file_path = args.input_file_path
        if not input_file_path:
            tag_load_parser.print_help()
            sys.exit(1)
        name_column = args.name_column
        description_column = args.description_column
        await tag_loader.load_tags(input_file_path, name_column, description_column)
    elif args.command == "unload_tag":
        from vector_search_util.util import tag_loader
        output_file = args.output_file
        if not output_file:
            tag_unload_parser.print_help()
            sys.exit(1)
        await tag_loader.unload_tags(output_file)
    elif args.command == "delete_tag":
        from vector_search_util.util import tag_loader
        input_file_path = args.input_file_path
        if not input_file_path:
            tag_delete_parser.print_help()
            sys.exit(1)
        name_column = args.name_column
        await tag_loader.delete_tags(input_file_path, name_column)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
