import argparse
import glob
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from textwrap import shorten
from typing import Any, Dict, List
from urllib import request
from urllib.error import HTTPError, URLError

from llama_index.core.node_parser import SimpleNodeParser
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
from rich.console import Console
from rich.panel import Panel

from backend.sqlite_kvstore import SQLiteKVStore
from wechat_source import (WeChatContactConfig, WeChatDBConfig,
                           chunk_wechat_documents, list_chat_sessions,
                           list_group_chats, list_group_chats_from_contact,
                           load_chatlog_documents_for_talker,
                           load_contact_names)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wechat_tui.py",
        description="TUI to select WeChat group and message table, then index messages.",
    )
    parser.add_argument(
        "--chatlog-dir",
        default=".",
        help="Base chatlog directory (containing message/ and contact/ subdirs)",
    )
    parser.add_argument(
        "--sqlite",
        default=None,
        help="Path to SQLite index DB used by the RAG indexer.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of messages to index.",
    )
    parser.add_argument(
        "--time",
        default="",
        help="Time range: YYYY-MM-DD or YYYY-MM-DD~YYYY-MM-DD. Empty means prompt.",
    )
    parser.add_argument(
        "--group-chat-id",
        default=None,
        help="WeChat talker id to use directly, skipping the TUI selector.",
    )
    parser.add_argument(
        "--chunk-messages",
        type=int,
        default=20,
        help="How many messages to merge into one indexed chunk.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=4000,
        help="Approx max characters per indexed chunk.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many chunks to retrieve for chat.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=20,
        help="Max QA pairs kept in chat history.",
    )
    parser.add_argument(
        "--embed-model",
        default="Qwen/Qwen3-Embedding-4B",
        help="Embedding model used for WeChat indexing and retrieval.",
    )
    parser.add_argument(
        "--export-format",
        choices=["txt", "json"],
        default=None,
        help="Export selected chat messages and skip chunking/indexing/chat.",
    )
    parser.add_argument(
        "--export-path",
        default=None,
        help="Output path for exported chat messages. Defaults to ./exports/<auto-name>.<fmt>",
    )
    return parser.parse_args()


def load_dotenv(dotenv_path: Path) -> Dict[str, str]:
    """Minimal .env loader for BASE_URL, API_KEY, MODEL_NAME."""
    env: Dict[str, str] = {}
    if not dotenv_path.exists():
        return env

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip().strip("'").strip('"')
    return env


def build_wechat_parser(args: argparse.Namespace) -> SimpleNodeParser:
    """Use a larger parser for already-chunked WeChat chat windows."""
    chunk_size = max(args.chunk_chars * 2, 2048)
    chunk_overlap = min(200, max(chunk_size // 10, 0))
    return SimpleNodeParser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def sanitize_filename_part(value: str) -> str:
    value = value.strip() or "unknown"
    value = value.replace("~", "_to_")
    return re.sub(r"[^A-Za-z0-9@._-]+", "_", value)


def build_default_sqlite_path(talker: str, time_range: str, args: argparse.Namespace) -> Path:
    index_dir = (Path.cwd() / "index").resolve()
    safe_talker = sanitize_filename_part(talker)
    safe_time = sanitize_filename_part(time_range or "all")
    filename = (
        f"{safe_talker}__{safe_time}"
        f"__m{args.chunk_messages}__c{args.chunk_chars}"
    )
    if args.limit > 0:
        filename += f"__limit{args.limit}"
    return index_dir / f"{filename}.db"


def build_default_export_path(
    talker: str,
    time_range: str,
    export_format: str,
    args: argparse.Namespace,
) -> Path:
    export_dir = (Path.cwd() / "exports").resolve()
    safe_talker = sanitize_filename_part(talker)
    safe_time = sanitize_filename_part(time_range or "all")
    filename = f"{safe_talker}__{safe_time}"
    if args.limit > 0:
        filename += f"__limit{args.limit}"
    return export_dir / f"{filename}.{export_format}"


def build_cache_label(sqlite_path: Path, meta: Dict[str, object] | None) -> str:
    if not meta:
        return sqlite_path.name
    talker = str(meta.get("talker", "unknown"))
    time_range = str(meta.get("time_range", "all"))
    chunk_messages = meta.get("chunk_messages", "?")
    chunk_chars = meta.get("chunk_chars", "?")
    return (
        f"{talker} | {time_range} | "
        f"m{chunk_messages} c{chunk_chars} | {sqlite_path.name}"
    )


def build_session_label(user_name: str, contact_mapping: Dict[str, str]) -> str:
    display = contact_mapping.get(user_name, user_name)
    session_type = "Group" if user_name.endswith("@chatroom") else "Direct"
    if display == user_name:
        return f"[{session_type}] {user_name}"
    return f"[{session_type}] {display} ({user_name})"


def choose_existing_cache_db(session: PromptSession) -> Path | None:
    index_dir = (Path.cwd() / "index").resolve()
    index_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(index_dir.glob("*.db"))
    if not candidates:
        return None

    labels: List[str] = []
    label_to_path: Dict[str, Path] = {}
    for path in candidates:
        label = build_cache_label(path, get_index_meta(path))
        labels.append(label)
        label_to_path[label] = path

    completer = FuzzyCompleter(WordCompleter(labels, ignore_case=True))
    while True:
        user_input = session.prompt(
            "Choose existing cache DB (ENTER to build new, TAB to complete): ",
            completer=completer,
        ).strip()
        if not user_input:
            return None
        if user_input in label_to_path:
            return label_to_path[user_input]
        matches = [label for label in labels if label.startswith(user_input)]
        if len(matches) == 1:
            return label_to_path[matches[0]]
        print("No exact cache match, please use TAB completion or press ENTER to build new.")


def print_retrieved_documents(nodes: List[object]) -> None:
    """Print retrieved RAG chunks for the current chat turn."""
    if not nodes:
        console.print("[yellow]RAG: no documents retrieved for this turn.[/]")
        return

    lines: List[str] = []
    for i, node in enumerate(nodes[:5], 1):
        metadata = getattr(node, "metadata", {}) or {}
        start_time = metadata.get("start_time") or metadata.get("create_time")
        end_time = metadata.get("end_time") or metadata.get("create_time")
        message_count = metadata.get("message_count", 1)
        snippet = shorten(
            (getattr(node, "text", "") or "").replace("\n", " "),
            width=220,
            placeholder=" ...",
        )
        lines.append(
            f"{i}. {start_time} ~ {end_time} | msgs={message_count}\n{snippet}"
        )

    if len(nodes) > 5:
        lines.append(f"... and {len(nodes) - 5} more chunk(s)")

    console.print(
        Panel(
            "\n\n".join(lines),
            title=f"RAG Retrieved Documents ({len(nodes)})",
            expand=False,
        )
    )


def export_chat_messages(
    docs: List[object],
    export_path: Path,
    export_format: str,
) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if export_format == "txt":
        content = "\n".join(format_export_text(getattr(doc, "text", "") or "") for doc in docs)
        export_path.write_text(content, encoding="utf-8")
        return

    rows: List[Dict[str, Any]] = []
    for doc in docs:
        rows.append(
            {
                "text": getattr(doc, "text", "") or "",
                "metadata": getattr(doc, "metadata", {}) or {},
            }
        )
    export_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _collapse_xml_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _clean_summary_field(value: str) -> str:
    value = (value or "").strip()
    return "" if value.lower() in {"null", "none"} else value


def _extract_raw_tag_text(raw: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", raw, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return _clean_summary_field(html.unescape(match.group(1)).strip())


def _extract_first_http_url(text: str) -> str:
    match = re.search(r"https?://[^\s<>\"]+", html.unescape(text or ""))
    return match.group(0) if match else ""


def _build_weapp_url(root: ET.Element) -> str:
    username = _clean_summary_field(root.findtext(".//weappinfo/username") or "")
    appid = _clean_summary_field(root.findtext(".//weappinfo/appid") or "")
    path = _clean_summary_field(root.findtext(".//pagepath") or root.findtext(".//path") or "")
    query = _clean_summary_field(root.findtext(".//query") or "")
    app = username or appid
    if not app:
        return ""
    suffix = path.lstrip("/") if path else ""
    url = f"weapp://{app}"
    if suffix:
        url = f"{url}/{suffix}"
    if query:
        connector = "&" if "?" in url else "?"
        url = f"{url}{connector}{query}"
    return url


def _extract_appmsg_url(root: ET.Element, raw: str) -> str:
    paths = [
        ".//url",
        ".//lowurl",
        ".//dataurl",
        ".//lowdataurl",
        ".//webviewshared/shareUrlOriginal",
        ".//webviewshared/shareUrlOpen",
        ".//shareUrlOriginal",
        ".//shareUrlOpen",
        ".//thumburl",
    ]
    for path in paths:
        value = _clean_summary_field(root.findtext(path) or "")
        if value.startswith("http://") or value.startswith("https://"):
            return value

    weapp_url = _build_weapp_url(root)
    if weapp_url:
        return weapp_url

    return _extract_first_http_url(raw)


def _format_structured_summary(label: str, title: str = "", des: str = "", url: str = "") -> str:
    parts = [part for part in [title, des] if part]
    summary = " | ".join(parts)
    if url:
        if summary:
            return f"[{label}] {summary} | {url}"
        return f"[{label}] {url}"
    if summary:
        return f"[{label}] {summary}"
    return f"[{label}]"


def _summarize_appmsg(root: ET.Element, raw: str, title: str, des: str, appmsg_type: str) -> str | None:
    has_weapp = bool(
        _clean_summary_field(root.findtext(".//weappinfo/username") or "") or
        _clean_summary_field(root.findtext(".//weappinfo/appid") or "") or
        _clean_summary_field(root.findtext(".//pagepath") or root.findtext(".//path") or "")
    )
    has_finder = any(
        root.find(path) is not None
        for path in [
            ".//findernamecard",
            ".//finderLiveProductShare",
            ".//finderShopWindowShare",
            ".//finderCollection",
            ".//finderFeed",
        ]
    )
    url = _extract_appmsg_url(root, raw)

    if appmsg_type == "5" or (url.startswith("http") and not has_weapp and not has_finder):
        return _format_structured_summary("Link", title, des, url)

    if appmsg_type in {"33", "36"} or has_weapp:
        return _format_structured_summary("MiniApp", title, des, url)

    if has_finder or appmsg_type in {"51", "63"}:
        return _format_structured_summary("Shared Post", title, des, url)

    if appmsg_type == "6":
        return _format_structured_summary("File", title, des, url)

    if appmsg_type in {"19", "40"}:
        return _format_structured_summary("Forwarded", title, des, url)

    if title or des:
        if url:
            return _format_structured_summary("Link", title, des, url)
        return None

    return None


def _quick_xml_summary(raw: str) -> str | None:
    raw_lc = raw.lower()
    if "<emoji" in raw_lc:
        return "[Sticker]"
    if "<img" in raw_lc:
        return "[Image]"

    if "<patinfo>" in raw_lc or "<patsuffix>" in raw_lc:
        title = _extract_raw_tag_text(raw, "title")
        patsuffix = _extract_raw_tag_text(raw, "patsuffix")
        if title:
            return f"[拍一拍] {title}"
        if patsuffix:
            return f"[拍一拍] {patsuffix}"
        return "[拍一拍]"

    if "<appmsg" in raw_lc:
        title = _extract_raw_tag_text(raw, "title")
        des = _extract_raw_tag_text(raw, "des")
        appmsg_type = _extract_raw_tag_text(raw, "type")
        has_weapp = any(token in raw_lc for token in ["<weappinfo", "<pagepath>", "<liteapp>"])
        has_finder = any(token in raw_lc for token in ["<findernamecard", "<finderliveproductshare", "<findershopwindowshare", "<findercollection", "<finderfeed"])
        url = (
            _extract_raw_tag_text(raw, "shareUrlOriginal")
            or _extract_raw_tag_text(raw, "shareUrlOpen")
            or _extract_raw_tag_text(raw, "url")
            or _extract_raw_tag_text(raw, "dataurl")
            or _extract_first_http_url(raw)
        )
        if appmsg_type == "5" or (url and not has_weapp and not has_finder):
            return _format_structured_summary("Link", title, des, url)
        if appmsg_type in {"33", "36"} or has_weapp:
            weapp_user = _extract_raw_tag_text(raw, "username")
            page_path = _extract_raw_tag_text(raw, "pagepath") or _extract_raw_tag_text(raw, "path")
            query = _extract_raw_tag_text(raw, "query")
            if not url and weapp_user:
                url = f"weapp://{weapp_user}"
                if page_path:
                    url = f"{url}/{page_path.lstrip('/')}"
                if query:
                    url = f"{url}?{query}"
            return _format_structured_summary("MiniApp", title, des, url)
        if has_finder or appmsg_type in {"51", "63"}:
            return _format_structured_summary("Shared Post", title, des, url)
        if appmsg_type == "6":
            return _format_structured_summary("File", title, des, url)
        if title or des:
            return " | ".join(part for part in [title, des] if part)

    return None


def _xml_placeholder(root: ET.Element) -> str | None:
    tag = (root.tag or "").lower()
    if tag == "sysmsg":
        return None

    if root.find(".//img") is not None:
        return "[Image]"

    emoji = root.find(".//emoji")
    if emoji is not None:
        return "[Sticker]"

    appmsg_type = (root.findtext(".//type") or "").strip()
    if appmsg_type in {"3", "5", "57", "62"}:
        return None
    if appmsg_type in {"2", "6", "8", "19"}:
        return "[Attachment]"

    return None


def _summarize_refer_content(content: str) -> str:
    content = (content or "").strip()
    if not content:
        return ""
    unescaped = html.unescape(content).strip()
    xml_starts = [pos for pos in [unescaped.find("<?xml"), unescaped.find("<msg"), unescaped.find("<sysmsg")] if pos >= 0]
    if xml_starts:
        xml_pos = min(xml_starts)
        prefix = unescaped[:xml_pos].strip(" :")
        xml_body = unescaped[xml_pos:].strip()
        xml_summary = _extract_xml_summary(xml_body) or "[Message]"
        if prefix:
            parts = [part.strip() for part in prefix.split(":") if part.strip()]
            speaker = next(
                (part for part in parts if not part.startswith("wxid_") and not part.endswith("@chatroom")),
                parts[0] if parts else "",
            )
            if speaker:
                return f"{speaker}: {xml_summary}"
        return xml_summary
    summary = _extract_xml_summary(unescaped)
    return summary or shorten(unescaped.replace("\n", " "), width=80, placeholder=" ...")


def _extract_xml_summary(body: str) -> str | None:
    raw = html.unescape((body or "")).strip()
    if not raw.startswith("<") and not raw.startswith("<?xml"):
        return None

    quick_summary = _quick_xml_summary(raw)
    if quick_summary and quick_summary in {"[Sticker]", "[Image]"}:
        return quick_summary

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return quick_summary or _collapse_xml_text(raw)[:200] or None

    if root.tag == "sysmsg":
        plain = (root.findtext(".//plain") or "").strip()
        template = (root.findtext(".//template") or "").strip()
        summary = plain or template
        summary = summary.replace("$username$", "").replace("$others$", "").strip(' "')
        return f"[系统消息] {summary}" if summary else "[系统消息]"

    title = _clean_summary_field(root.findtext(".//title") or "")
    des = _clean_summary_field(root.findtext(".//des") or "")
    appmsg_type = (root.findtext(".//type") or "").strip()
    refer_display = _clean_summary_field(root.findtext(".//refermsg/displayname") or "")
    refer_content = (root.findtext(".//refermsg/content") or "").strip()

    if appmsg_type == "62":
        pat_title = (root.findtext(".//appmsg/title") or title).strip()
        if pat_title:
            return f"[拍一拍] {pat_title}"
        patsuffix = (root.findtext(".//patinfo/patsuffix") or "").strip()
        if patsuffix:
            return f"[拍一拍] {patsuffix}"
        return "[拍一拍]"

    if refer_display or refer_content:
        parts: List[str] = []
        if title:
            parts.append(title)
        if refer_display or refer_content:
            refer_summary = _summarize_refer_content(refer_content)
            quoted = f"{refer_display}: {refer_summary}".strip(": ").strip()
            if quoted:
                parts.append(f"回复[{quoted}]")
        summary = " | ".join(part for part in parts if part)
        if summary:
            return summary

    appmsg_summary = _summarize_appmsg(root, raw, title, des, appmsg_type)
    if appmsg_summary:
        return appmsg_summary

    if title or des:
        return " | ".join(part for part in [title, des] if part)

    placeholder = _xml_placeholder(root)
    if placeholder is not None:
        return placeholder

    collapsed = _collapse_xml_text(raw)
    if collapsed:
        return collapsed[:200]
    if root.tag == "msg":
        return "[Message]"
    return None


def format_export_text(raw_text: str) -> str:
    """Make TXT exports more readable by summarizing XML-like message bodies."""
    timestamp_match = re.match(r"^(\[[^\]]+\]\s+)(.*)$", raw_text, flags=re.DOTALL)
    if not timestamp_match:
        summary = _extract_xml_summary(raw_text)
        return summary or raw_text

    timestamp_prefix, remainder = timestamp_match.groups()
    xml_starts = [
        pos for pos in [
            remainder.find("<?xml"),
            remainder.find("<msg"),
            remainder.find("<sysmsg"),
        ] if pos >= 0
    ]

    if xml_starts:
        xml_pos = min(xml_starts)
        sep_pos = remainder.rfind(": ", 0, xml_pos)
        if sep_pos >= 0:
            prefix = f"{timestamp_prefix}{remainder[:sep_pos + 2]}"
            body = remainder[sep_pos + 2:]
        else:
            prefix = timestamp_prefix
            body = remainder
    else:
        sep_pos = remainder.find(": ")
        if sep_pos < 0:
            summary = _extract_xml_summary(raw_text)
            return summary or raw_text
        prefix = f"{timestamp_prefix}{remainder[:sep_pos + 2]}"
        body = remainder[sep_pos + 2:]

    summary = _extract_xml_summary(body)
    if summary:
        return f"{prefix}{summary}"
    return raw_text


def chat_completion(base_url: str, api_key: str, model_name: str, messages: List[Dict[str, str]]) -> str:
    """Call an OpenAI-compatible chat completion endpoint."""
    root = base_url.rstrip("/")
    if not root.endswith("/v1"):
        root = f"{root}/v1"
    url = f"{root}/chat/completions"

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.2,
    }
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM HTTP error {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"LLM connection failed: {e}") from e

    try:
        return body["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected LLM response: {body}") from e


def build_index_meta(
    base_dir: Path,
    talker: str,
    time_range: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    return {
        "chatlog_dir": str(base_dir),
        "talker": talker,
        "time_range": time_range,
        "limit": args.limit,
        "chunk_messages": args.chunk_messages,
        "chunk_chars": args.chunk_chars,
    }


def get_index_meta(sqlite_path: Path) -> Dict[str, object] | None:
    if not sqlite_path.exists():
        return None
    kvstore = SQLiteKVStore(str(sqlite_path))
    try:
        return kvstore.get("wechat_index_meta")
    finally:
        kvstore.close()


def save_index_meta(sqlite_path: Path, meta: Dict[str, object]) -> None:
    kvstore = SQLiteKVStore(str(sqlite_path))
    try:
        kvstore["wechat_index_meta"] = meta
    finally:
        kvstore.close()


def run_chat(sqlite_path: Path, args: argparse.Namespace) -> None:
    """Interactive RAG chat over the just-built SQLite index using .env LLM config."""
    from chat import (count_unique_files, extract_filenames, format_history,
                      get_nodes, prepare_chat_state)
    from indexer import configure_settings

    env = load_dotenv(Path.cwd() / ".env")
    base_url = env.get("BASE_URL") or os.getenv("BASE_URL")
    api_key = env.get("API_KEY") or os.getenv("API_KEY")
    model_name = env.get("MODEL_NAME") or os.getenv("MODEL_NAME")

    if not base_url or not api_key or not model_name:
        print("Skipping chat: missing BASE_URL / API_KEY / MODEL_NAME in .env or environment.")
        return

    configure_settings(args.embed_model, build_wechat_parser(args))

    _, _, indexed_nodes, filenames, retriever = prepare_chat_state(sqlite_path, args.top_k)
    doc_count = len(indexed_nodes)
    file_count = count_unique_files(indexed_nodes)
    console.print(
        f"\n[bold cyan]Chat ready[/] | model={model_name} | top_k={args.top_k} | "
        f"docs={doc_count} | chunks={file_count}"
    )
    print("Enter your question. Type 'exit' or 'quit' to leave chat.")
    print("Commands: /mode search, /mode chat, /mode, /help")

    history: List[tuple[str, str]] = []
    search_only_mode = False
    chat_commands = [
        "/help",
        "/mode",
        "/mode chat",
        "/mode search",
    ]
    session = PromptSession(
        completer=WordCompleter(chat_commands, ignore_case=True, sentence=True)
    )
    system_prompt = (
        "You are a helpful assistant answering questions about the selected WeChat group chat. "
        "Use the retrieved chat context faithfully, cite uncertainty when context is incomplete, "
        "and answer concisely."
    )

    while True:
        user_input = session.prompt("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.startswith("/"):
            command = user_input.lower()
            if command == "/help":
                print("Commands:")
                print("  /mode          Show current mode")
                print("  /mode chat     Retrieve docs and send context to the model")
                print("  /mode search   Retrieve docs only, do not call the model")
                print("  /help          Show this help")
                continue
            if command == "/mode":
                mode_name = "search-only" if search_only_mode else "chat"
                print(f"Current mode: {mode_name}")
                continue
            if command == "/mode chat":
                search_only_mode = False
                print("Switched to chat mode.")
                continue
            if command == "/mode search":
                search_only_mode = True
                print("Switched to search-only mode.")
                continue
            print("Unknown command. Type /help for available commands.")
            continue

        nodes = get_nodes(user_input, filenames, indexed_nodes, retriever)
        print_retrieved_documents(nodes)
        if search_only_mode:
            print("Search-only mode: skipped model call.")
            continue
        context = "\n\n".join(node.text for node in nodes)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Conversation history:\n{format_history(history)}\n\n"
                    f"User question: {user_input}"
                ),
            },
        ]

        try:
            answer = chat_completion(base_url, api_key, model_name, messages)
        except Exception as e:
            print(f"Chat failed: {e}")
            continue

        retrieved_filenames = extract_filenames(nodes)
        if retrieved_filenames:
            print(f"[retrieved {len(nodes)} chunks from {len(retrieved_filenames)} source docs]")
        console.print(f"[bold green]AI:[/] {answer}")
        history.append((user_input, answer))
        history = history[-args.history_limit:]


def main() -> None:
    args = parse_args()
    session = PromptSession()

    if not args.sqlite and not args.export_format:
        cached_sqlite = choose_existing_cache_db(session)
        if cached_sqlite is not None:
            meta = get_index_meta(cached_sqlite)
            print(f"Using cached DB: {cached_sqlite}")
            if meta:
                print(
                    "Cached index metadata: "
                    f"talker={meta.get('talker')} time={meta.get('time_range')} "
                    f"chunk_messages={meta.get('chunk_messages')} chunk_chars={meta.get('chunk_chars')}"
                )
            run_chat(cached_sqlite, args)
            return

    base_dir = Path(args.chatlog_dir).expanduser().resolve()

    # Discover message DBs under message/
    message_dir = base_dir / "message"
    matched = glob.glob(str(message_dir / "message_*.db"))
    db_paths: List[Path] = [Path(p).resolve() for p in matched]
    if not db_paths:
        print(f"No message_*.db found under {message_dir}")
        return

    # Locate contact.db for name mapping
    contact_db_path = base_dir / "contact" / "contact.db"
    contact_mapping: Dict[str, str] = {}
    if contact_db_path.exists():
        contact_cfg = WeChatContactConfig(contact_db_path=contact_db_path)
        contact_mapping = load_contact_names(contact_cfg)
    else:
        print(f"Warning: contact.db not found at {contact_db_path}, will show raw IDs.")

    # 1) choose chat session (group or direct)
    if args.group_chat_id:
        chosen_group = args.group_chat_id.strip()
        chosen_label = contact_mapping.get(chosen_group, chosen_group)
        if chosen_label != chosen_group:
            chosen_label = f"{chosen_label} ({chosen_group})"
        print(f"\nSelected conversation: {chosen_label}")
    else:
        all_sessions = set()
        for db_path in db_paths:
            cfg = WeChatDBConfig(db_path=db_path)
            for session_name in list_chat_sessions(cfg):
                all_sessions.add(session_name)
        sessions = sorted(all_sessions)

        if not sessions:
            print("No chat sessions found in Name2Id.")
            return

        labelled_groups = [build_session_label(s, contact_mapping) for s in sessions]

        # Fuzzy-search TUI for session selection
        group_completer = FuzzyCompleter(
            WordCompleter(labelled_groups, ignore_case=True)
        )

        while True:
            user_input = session.prompt(
                "Type to search conversation (TAB to complete, ENTER to confirm): ",
                completer=group_completer,
            ).strip()
            if not user_input:
                continue
            if user_input in labelled_groups:
                chosen_label = user_input
                break
            # If user typed only the display name part, try to match prefix
            matches = [lbl for lbl in labelled_groups if lbl.startswith(user_input)]
            if len(matches) == 1:
                chosen_label = matches[0]
                break
            print("No exact match, please use TAB 补全再回车确认。")

        chosen_group = sessions[labelled_groups.index(chosen_label)]
        print(f"\nSelected conversation: {chosen_label}")

    time_range = args.time.strip()
    if not time_range:
        time_range = session.prompt(
            "Time range (YYYY-MM-DD or YYYY-MM-DD~YYYY-MM-DD, default all): "
        ).strip() or "all"
    print(f"Time range: {time_range}")

    if args.sqlite:
        sqlite_path = Path(args.sqlite).expanduser().resolve()
    else:
        sqlite_path = build_default_sqlite_path(chosen_group, time_range, args).resolve()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"SQLite path: {sqlite_path}")

    requested_meta = build_index_meta(base_dir, chosen_group, time_range, args)
    existing_meta = get_index_meta(sqlite_path)
    index_matches = existing_meta == requested_meta

    if index_matches:
        print("Existing matching index found; skipping message fetch, chunking, and reindexing.")
        run_chat(sqlite_path, args)
        return

    # 2) query messages directly by talker + time range
    raw_docs = load_chatlog_documents_for_talker(
        base_dir=base_dir,
        talker=chosen_group,
        time_range=time_range,
        limit=args.limit,
        contact_mapping=contact_mapping,
    )
    print(f"Loaded {len(raw_docs)} WeChat messages.")

    if not raw_docs:
        print("No non-empty messages found to index.")
        return

    if args.export_format:
        if args.export_path:
            export_path = Path(args.export_path).expanduser().resolve()
        else:
            export_path = build_default_export_path(
                chosen_group,
                time_range,
                args.export_format,
                args,
            )
        export_chat_messages(raw_docs, export_path, args.export_format)
        print(f"Exported {len(raw_docs)} WeChat messages to {export_path}")
        return

    # Only load the embedding model once we know we need chunking/indexing.
    from indexer import configure_settings, index_wechat_docs

    configure_settings(args.embed_model, build_wechat_parser(args))

    docs = chunk_wechat_documents(
        raw_docs,
        chunk_messages=args.chunk_messages,
        chunk_chars=args.chunk_chars,
    )
    print(
        f"Chunked into {len(docs)} documents "
        f"(chunk_messages={args.chunk_messages}, chunk_chars={args.chunk_chars})."
    )

    if not docs:
        print("No non-empty messages found to index.")
        return

    # 3) run indexing into the RAG SQLite backend
    index_wechat_docs(docs, sqlite_path)
    save_index_meta(sqlite_path, requested_meta)
    print(f"Indexed WeChat docs into {sqlite_path}")

    # 4) start chat over the freshly built group index
    run_chat(sqlite_path, args)


if __name__ == "__main__":
    main()

