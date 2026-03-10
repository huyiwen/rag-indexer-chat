import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dtime
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llama_index.core import Document
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           ProgressColumn, TaskProgressColumn, TextColumn,
                           TimeElapsedColumn)
from rich.text import Text


@dataclass
class WeChatDBConfig:
    """Config for accessing a single WeChat message_*.db file."""

    db_path: Path


@dataclass
class WeChatContactConfig:
    """Config for accessing contact.db for nickname/remark mapping."""

    contact_db_path: Path


@dataclass
class MessageDBInfo:
    db_path: Path
    start_time: datetime
    end_time: datetime


class RowSpeedColumn(ProgressColumn):
    """Render rows/sec safely before Rich has enough samples to compute speed."""

    def render(self, task) -> Text:
        speed = task.speed
        if speed is None:
            return Text(" --.-/s")
        return Text(f"{speed:6.1f}/s")


def _connect_path(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _connect(cfg: WeChatDBConfig) -> sqlite3.Connection:
    return _connect_path(cfg.db_path)


def list_group_chats(cfg: WeChatDBConfig) -> List[str]:
    """Return all group chat user_name values (xxx@chatroom) for a DB.

    Some message_*.db files may not have the Name2Id table; in that case
    we simply return an empty list for that DB.
    """
    conn = _connect(cfg)
    cur = conn.cursor()

    # Check table existence first to avoid OperationalError
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='Name2Id'"
    )
    has_table = cur.fetchone() is not None
    if not has_table:
        conn.close()
        return []

    cur.execute(
        """
        SELECT user_name
        FROM Name2Id
        WHERE is_session = 1 AND user_name LIKE '%@chatroom'
        ORDER BY user_name
        """
    )
    rows = [r["user_name"] for r in cur.fetchall()]
    conn.close()
    return rows


def list_chat_sessions(
    cfg: WeChatDBConfig,
    include_groups: bool = True,
    include_direct: bool = True,
) -> List[str]:
    """Return active session talkers from Name2Id for a DB."""
    conn = _connect(cfg)
    cur = conn.cursor()

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='Name2Id'"
    )
    has_table = cur.fetchone() is not None
    if not has_table:
        conn.close()
        return []

    clauses = []
    if include_groups:
        clauses.append("user_name LIKE '%@chatroom'")
    if include_direct:
        clauses.append("user_name NOT LIKE '%@chatroom'")

    where_clause = " OR ".join(clauses) if clauses else "1 = 0"
    cur.execute(
        f"""
        SELECT user_name
        FROM Name2Id
        WHERE is_session = 1 AND ({where_clause})
        ORDER BY user_name
        """
    )
    rows = [r["user_name"] for r in cur.fetchall() if r["user_name"]]
    conn.close()
    return rows


def list_group_chats_from_contact(contact_cfg: WeChatContactConfig) -> List[str]:
    """Return all known chatroom usernames from contact.db."""
    conn = _connect_path(contact_cfg.contact_db_path)
    cur = conn.cursor()

    groups = set()

    cur.execute(
        """
        SELECT username
        FROM contact
        WHERE username LIKE '%@chatroom'
        """
    )
    groups.update(
        row["username"] for row in cur.fetchall()
        if row["username"]
    )

    # chat_room may contain rooms missing from contact rows
    cur.execute(
        """
        SELECT username
        FROM chat_room
        """
    )
    groups.update(
        row["username"] for row in cur.fetchall()
        if row["username"]
    )

    conn.close()
    return sorted(groups)


def list_msg_tables(cfg: WeChatDBConfig) -> List[str]:
    """Return all Msg_* tables in the DB."""
    conn = _connect(cfg)
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'Msg_%' ORDER BY name"
    )
    rows = [r["name"] for r in cur.fetchall()]
    conn.close()
    return rows


def parse_time_range(time_range: str) -> Tuple[datetime, datetime]:
    """
    Parse YYYY-MM-DD or YYYY-MM-DD~YYYY-MM-DD.

    Empty string or 'all' means from epoch to now.
    """
    raw = (time_range or "").strip()
    if not raw or raw.lower() == "all":
        return datetime.fromtimestamp(0), datetime.now()

    if "~" in raw:
        start_raw, end_raw = raw.split("~", 1)
    else:
        start_raw = raw
        end_raw = raw

    start_date = datetime.strptime(start_raw.strip(), "%Y-%m-%d").date()
    end_date = datetime.strptime(end_raw.strip(), "%Y-%m-%d").date()

    start_dt = datetime.combine(start_date, dtime.min)
    end_dt = datetime.combine(end_date, dtime.max.replace(microsecond=0))
    return start_dt, end_dt


def _decode_message_content(raw: object) -> str:
    """Decode plain or compressed message content into text."""
    if raw is None:
        return ""

    if isinstance(raw, str):
        return raw.strip()

    if not isinstance(raw, (bytes, bytearray)):
        return str(raw).strip()

    data = bytes(raw)
    if not data:
        return ""

    # zstd magic number used by chatlog's v4 decoder.
    if data.startswith(b"\x28\xb5\x2f\xfd"):
        try:
            import zstandard as zstd  # type: ignore

            decompressor = zstd.ZstdDecompressor()
            return decompressor.decompress(data).decode("utf-8", errors="ignore").strip()
        except Exception:
            pass

    return data.decode("utf-8", errors="ignore").strip()


def _talker_md5(talker: str) -> str:
    return hashlib.md5(talker.encode("utf-8")).hexdigest()


def load_message_db_infos(message_dir: Path) -> List[MessageDBInfo]:
    """Build time ranges for all message_*.db files using Timestamp table."""
    infos: List[Tuple[Path, datetime]] = []

    for db_path in sorted(message_dir.glob("message_*.db")):
        conn = _connect_path(db_path)
        cur = conn.cursor()
        try:
            cur.execute("SELECT timestamp FROM Timestamp LIMIT 1")
            row = cur.fetchone()
            if not row:
                continue
            infos.append((db_path, datetime.fromtimestamp(int(row["timestamp"]))))
        except sqlite3.DatabaseError:
            continue
        finally:
            conn.close()

    infos.sort(key=lambda item: item[1])

    result: List[MessageDBInfo] = []
    for idx, (db_path, start_time) in enumerate(infos):
        if idx == len(infos) - 1:
            end_time = datetime.now()
        else:
            end_time = infos[idx + 1][1]
        result.append(
            MessageDBInfo(
                db_path=db_path,
                start_time=start_time,
                end_time=end_time,
            )
        )
    return result


def get_message_dbs_for_time_range(
    message_dir: Path,
    start_time: datetime,
    end_time: datetime,
) -> List[MessageDBInfo]:
    """Return message DB files overlapping the requested time range."""
    infos = load_message_db_infos(message_dir)
    return [
        info for info in infos
        if info.start_time < end_time and info.end_time > start_time
    ]


def load_chat_messages_as_documents(
    cfg: WeChatDBConfig,
    msg_table: str,
    limit: Optional[int] = None,
) -> List[Document]:
    """
    Load messages from a given Msg_* table and convert them into LlamaIndex Documents.

    Currently we only use plaintext `message_content` and basic metadata.
    """
    conn = _connect(cfg)
    cur = conn.cursor()

    sql = f"""
        SELECT local_id,
               real_sender_id,
               create_time,
               message_content
        FROM {msg_table}
        ORDER BY sort_seq
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    cur.execute(sql)

    docs: List[Document] = []
    for row in cur.fetchall():
        raw = row["message_content"]

        # Some rows store binary blobs here; try to decode, otherwise skip.
        if isinstance(raw, bytes):
            try:
                text = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue
        else:
            text = (raw or "").strip()

        if not text:
            continue

        metadata = {
            "source": "wechat",
            "msg_table": msg_table,
            "local_id": row["local_id"],
            "create_time": row["create_time"],
            "real_sender_id": row["real_sender_id"],
        }
        docs.append(Document(text=text, metadata=metadata))

    conn.close()
    return docs


def load_contact_names(
    contact_cfg: WeChatContactConfig,
) -> Dict[str, str]:
    """
    Load mapping from username -> display name using contact.db.

    Prefer remark, then nick_name, otherwise fall back to username itself.
    """
    conn = _connect_path(contact_cfg.contact_db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT username, remark, nick_name
        FROM contact
        """
    )
    mapping: Dict[str, str] = {}
    for row in cur.fetchall():
        username = row["username"]
        remark = (row["remark"] or "").strip()
        nick = (row["nick_name"] or "").strip()
        if remark:
            display = remark
        elif nick:
            display = nick
        else:
            display = username
        mapping[username] = display

    conn.close()
    return mapping


def load_chatlog_documents_for_talker(
    base_dir: Path,
    talker: str,
    time_range: str,
    limit: int = 10000,
    offset: int = 0,
    contact_mapping: Optional[Dict[str, str]] = None,
) -> List[Document]:
    """
    Minimal local v4 implementation of chatlog's talker + time query.

    - Resolves target table as Msg_<md5(talker)>
    - Picks relevant message_*.db files using Timestamp table
    - Reads and merges messages within the requested time range
    """
    start_dt, end_dt = parse_time_range(time_range)
    message_dir = base_dir / "message"
    target_table = f"Msg_{_talker_md5(talker)}"

    db_infos = get_message_dbs_for_time_range(message_dir, start_dt, end_dt)
    if not db_infos:
        return []

    all_rows: List[dict] = []
    total_rows = 0
    for info in db_infos:
        conn = _connect_path(info.db_path)
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (target_table,),
            )
            if cur.fetchone() is None:
                continue
            cur.execute(
                f"SELECT COUNT(*) AS cnt FROM {target_table} WHERE create_time >= ? AND create_time <= ?",
                (int(start_dt.timestamp()), int(end_dt.timestamp())),
            )
            row = cur.fetchone()
            total_rows += int(row["cnt"]) if row else 0
        finally:
            conn.close()

    progress = Progress(
        TextColumn("[cyan]Loading WeChat messages"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        RowSpeedColumn(),
        transient=True,
    )

    with progress:
        task_id = progress.add_task("load", total=max(total_rows, 1))
        for info in db_infos:
            conn = _connect_path(info.db_path)
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (target_table,),
                )
                if cur.fetchone() is None:
                    continue

                query = f"""
                    SELECT
                        m.sort_seq,
                        m.server_id,
                        m.local_type,
                        n.user_name,
                        m.create_time,
                        m.message_content,
                        m.packed_info_data,
                        m.status
                    FROM {target_table} m
                    LEFT JOIN Name2Id n ON m.real_sender_id = n.rowid
                    WHERE m.create_time >= ? AND m.create_time <= ?
                    ORDER BY m.sort_seq ASC
                """
                cur.execute(query, (int(start_dt.timestamp()), int(end_dt.timestamp())))

                for row in cur.fetchall():
                    progress.advance(task_id, 1)
                    text = _decode_message_content(row["message_content"])
                    sender = row["user_name"] or ""

                    # Group messages are often stored as "sender:\ncontent"
                    if talker.endswith("@chatroom") and ":\n" in text:
                        sender_from_body, body = text.split(":\n", 1)
                        if sender_from_body:
                            sender = sender_from_body
                        text = body

                    text = text.strip()
                    if not text:
                        continue

                    sender_display = (contact_mapping or {}).get(sender, sender)
                    talker_display = (contact_mapping or {}).get(talker, talker)
                    timestamp = int(row["create_time"])
                    dt = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

                    formatted_text = f"[{dt}] {sender_display or sender}: {text}"
                    all_rows.append(
                        {
                            "sort_seq": int(row["sort_seq"]),
                            "db_name": info.db_path.name,
                            "server_id": row["server_id"],
                            "local_type": row["local_type"],
                            "sender": sender,
                            "sender_display": sender_display,
                            "talker": talker,
                            "talker_display": talker_display,
                            "create_time": timestamp,
                            "text": formatted_text,
                        }
                    )
            finally:
                conn.close()

    all_rows.sort(key=lambda item: item["sort_seq"])

    if offset > 0:
        all_rows = all_rows[offset:]
    if limit > 0:
        all_rows = all_rows[:limit]

    docs: List[Document] = []
    for item in all_rows:
        metadata = {
            "source": "wechat",
            "db_name": item["db_name"],
            "talker": item["talker"],
            "talker_display": item["talker_display"],
            "sender": item["sender"],
            "sender_display": item["sender_display"],
            "create_time": item["create_time"],
            "sort_seq": item["sort_seq"],
            "server_id": item["server_id"],
            "local_type": item["local_type"],
        }
        docs.append(Document(text=item["text"], metadata=metadata))

    return docs


def chunk_wechat_documents(
    docs: List[Document],
    chunk_messages: int = 20,
    chunk_chars: int = 4000,
) -> List[Document]:
    """
    Merge sequential WeChat message documents into larger context windows.

    This usually improves retrieval quality over indexing one short message per doc.
    """
    if chunk_messages <= 1 and chunk_chars <= 0:
        return docs

    chunked: List[Document] = []
    current_texts: List[str] = []
    current_docs: List[Document] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current_texts, current_docs, current_chars
        if not current_docs:
            return

        first_meta = current_docs[0].metadata or {}
        last_meta = current_docs[-1].metadata or {}
        db_names = sorted({
            d.metadata.get("db_name")
            for d in current_docs
            if d.metadata and d.metadata.get("db_name")
        })
        senders = sorted({
            d.metadata.get("sender")
            for d in current_docs
            if d.metadata and d.metadata.get("sender")
        })
        merged_metadata = {
            "source": "wechat",
            "talker": first_meta.get("talker"),
            "talker_display": first_meta.get("talker_display"),
            "start_time": first_meta.get("create_time"),
            "end_time": last_meta.get("create_time"),
            "message_count": len(current_docs),
            # Keep WeChat metadata compact so LlamaIndex's metadata-aware splitter
            # doesn't exceed chunk size on already-chunked chat windows.
            "db_count": len(db_names),
            "sender_count": len(senders),
            "first_sender": senders[0] if senders else None,
        }
        chunked.append(
            Document(
                text="\n".join(current_texts),
                metadata=merged_metadata,
            )
        )
        current_texts = []
        current_docs = []
        current_chars = 0

    for doc in docs:
        text = doc.text or ""
        text_len = len(text)

        over_message_limit = chunk_messages > 0 and len(current_docs) >= chunk_messages
        over_char_limit = chunk_chars > 0 and current_chars + text_len > chunk_chars

        if current_docs and (over_message_limit or over_char_limit):
            flush()

        current_docs.append(doc)
        current_texts.append(text)
        current_chars += text_len + 1

    flush()
    return chunked


