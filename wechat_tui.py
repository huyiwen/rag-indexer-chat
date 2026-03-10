import argparse
import logging
import signal
import sys
from pathlib import Path

from prompt_toolkit.shortcuts import radiolist_dialog

from llama_index.core import Document

from wechat_source import (
    WeChatDBConfig,
    list_group_chats,
    list_msg_tables,
    load_chat_messages_as_documents,
)
from indexer import index_wechat_docs


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wechat_tui.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Select a WeChat group and Msg_* table, then index its messages into the RAG SQLite backend.",
    )

    parser.add_argument(
        "--db",
        required=True,
        type=Path,
        metavar="PATH",
        help="Path to WeChat message_X.db (e.g. message_0.db)",
    )
    parser.add_argument(
        "--sqlite",
        required=True,
        type=Path,
        metavar="SQLITE_DB",
        help="Path to the RAG SQLite index database (same as indexer --sqlite)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model used for embeddings",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        metavar="TOKENS",
        help="Chunk size used when splitting chat messages",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        metavar="TOKENS",
        help="Token overlap between consecutive chunks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Optional limit on number of messages to index from the Msg_* table",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def choose_from_list(title: str, text: str, options: list[str]) -> str:
    """Simple radiolist selector using prompt_toolkit."""
    values = [(opt, opt) for opt in options]
    result = radiolist_dialog(
        title=title,
        text=text,
        values=values,
    ).run()

    if result is None:
        print("Cancelled by user.")
        sys.exit(0)

    return result


def main() -> None:
    args = parse_args()
    configure_logging(args.debug)

    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    cfg = WeChatDBConfig(db_path=args.db.expanduser())

    if not cfg.db_path.exists():
        log.error("WeChat DB not found at %s", cfg.db_path)
        sys.exit(1)

    group_chats = list_group_chats(cfg)
    if not group_chats:
        log.error("No group chats (%%@chatroom) found in Name2Id.")
        sys.exit(1)

    selected_chat = choose_from_list(
        "Select Group Chat",
        "Use arrow keys to pick a group chat, then press Enter.",
        sorted(group_chats),
    )
    log.info("Selected group chat: %s", selected_chat)

    msg_tables = list_msg_tables(cfg)
    if not msg_tables:
        log.error("No Msg_* tables found in %s", cfg.db_path)
        sys.exit(1)

    selected_table = choose_from_list(
        "Select Msg_* Table",
        f"Choose the Msg_* table that corresponds to {selected_chat}.\n"
        "You can memorise this mapping for later runs.",
        sorted(msg_tables),
    )
    log.info("Selected message table: %s", selected_table)

    docs: list[Document] = load_chat_messages_as_documents(
        cfg,
        selected_table,
        limit=args.limit,
        user_name=selected_chat,
    )

    if not docs:
        log.warning("No non-empty messages found in %s", selected_table)
        sys.exit(0)

    log.info("Loaded %d WeChat messages as documents. Indexing...", len(docs))

    index_wechat_docs(
        docs,
        sqlite_path=args.sqlite,
        model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    log.info("WeChat messages indexed successfully into %s", args.sqlite)


if __name__ == "__main__":
    main()

