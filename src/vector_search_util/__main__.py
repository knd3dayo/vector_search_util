import argparse
import sys
from vector_search_util.util import search, loader


def main():
    parser = argparse.ArgumentParser(
        description="Vector Search Utility CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # search サブコマンド
    search_parser = subparsers.add_parser("search", help="Execute search process")
    search_parser.add_argument("-q", "--query", type=str, required=True, help="Search query text.")
    search_parser.add_argument("-k", "--top_k", type=int, default=5, help="Number of top results to return.")
    search_parser.add_argument("-f", "--filter_file", type=str, help="Path to JSON file containing filter conditions.")

    # load サブコマンド
    load_parser = subparsers.add_parser("load", help="Execute data loading process")
    load_parser.add_argument("-f", "--file_path", type=str, help="Path to the Excel file.")
    load_parser.add_argument("-c", "--content_column", type=str, default="content", help="Name of the content column.")
    load_parser.add_argument("-i", "--source_id_column", type=str, default="source_id", help="Name of the source_id column.")
    load_parser.add_argument("-m", "--metadata_columns", type=str, nargs="*", default=[], help="List of metadata column names.")

    args = parser.parse_args()
    print(f"Executing command: {args.command}")

    if args.command == "search":
        if hasattr(search, "main"):
            search.main(args)
        else:
            print("search.py に main 関数が見つかりません。")
            sys.exit(1)
    elif args.command == "load":
        if hasattr(loader, "main"):
            loader.main(args)
        else:
            print("loader.py に main 関数が見つかりません。")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
