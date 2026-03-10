import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from llama_index.core import Document


@dataclass
class WeChatDBConfig:
    """Configuration for accessing a single WeChat message database."""

    db_path: Path  # e.g. message_0.db


def _connect(cfg: WeChatDBConfig) -> sqlite3.Connection:
    conn = sqlite3.connect(str(cfg.db_path))
    return conn


def list_group_chats(cfg: WeChatDBConfig) -> List[str]:
    """Return all group chat user_name values from Name2Id."""
    conn = _connect(cfg)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT user_name FROM Name2Id "
            "WHERE is_session = 1 AND user_name LIKE '%@chatroom'"
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def list_msg_tables(cfg: WeChatDBConfig) -> List[str]:
    """Return all Msg_* tables from the given DB."""
    conn = _connect(cfg)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name LIKE 'Msg_%'"
        )
        tables = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
    return tables


def load_chat_messages_as_documents(
    cfg: WeChatDBConfig,
    msg_table: str,
    *,
    limit: Optional[int] = None,
    user_name: Optional[str] = None,
) -> List[Document]:
    """
    Load messages from a single Msg_* table as LlamaIndex Documents.

    Empty or whitespace‑only messages are skipped.
    """
    conn = _connect(cfg)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        sql = f"SELECT * FROM {msg_table} ORDER BY sort_seq"
        if limit:
            sql += f" LIMIT {limit}"
        cur.execute(sql)

        docs: List[Document] = []
        for row in cur.fetchall():
            text = row["message_content"] or ""
            if not text.strip():
                continue

            meta = {
                "source": "wechat",
                "msg_table": msg_table,
                "local_id": row["local_id"],
                "create_time": row["create_time"],
                "real_sender_id": row["real_sender_id"],
            }
            if user_name:
                meta["chat_user_name"] = user_name

            docs.append(Document(text=text, metadata=meta))
    finally:
        conn.close()

    return docs

