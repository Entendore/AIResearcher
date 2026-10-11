"""
Genesis — Autonomous Local AI Agent
========================================
Features:
  • Local LLM streaming chat (OpenAI-compatible API, e.g. llama.cpp server)
  • Live reasoning-stream rendering with ildal annotation
  • Local embedding + ChromaDB persistent vector store (RAG)
  • Tool calling: web_search, fetch_page, search_local_kb, execute_python,
    write_file, read_file, list_files
  • Parallel tool execution
  • Session resume / history persistence
  • Document ingestion (txt, md, py, json, pdf)
  • Robust context trimming for long sessions
  • Retry-with-backoff for LLM requests
  • Workspace-scoped filesystem access (security)

Requirements:
  pip install httpx chromadb duckduckgo_search trafilatura html2text
              pymupdf rich
  Plus a running llama.cpp server (LLM on :8080, embeddings on :8081)
"""

import os
import re
import sys
import json
import time
import glob
import hashlib
import tempfile
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import httpx
import chromadb
from duckduckgo_search import DDGS
import trafilatura
import html2text
import fitz  # PyMuPDF
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

# =============================================================================
# Configuration
# =============================================================================
console = Console()

LLM_URL      = os.environ.get("GENESIS_LLM_URL",   "http://127.0.0.1:8080")
EMBED_URL    = os.environ.get("GENESIS_EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")
HISTORY_FILE = "genesis_chat_history.json"
CACHE_DIR    = "./.web_cache"
CHROMA_PATH  = "./genesis_memory"
COLLECTION   = "knowledge_base"

CONTEXT_LIMIT     = int(os.environ.get("GENESIS_CONTEXT_LIMIT", "32768"))
MAX_STEPS         = 10
MAX_TOOL_OUTPUT   = 6000
MAX_HISTORY_BYTES = 2 * 1024 * 1024  # 2 MB cap on saved history file
LLM_TIMEOUT       = 300.0
LLM_MAX_RETRIES   = 3
FETCH_TIMEOUT     = 15.0
PYTHON_TIMEOUT    = 15
LIST_DEPTH_LIMIT  = 4
LIST_MAX_FILES    = 50

# Reasoning-model thinking tags
THINK_OPEN  = "ildal"
THINK_CLOSE = "ldal"

# Directories excluded from list_files walks
EXCLUDED_DIRS = {
    "__pycache__", ".git", ".svn", ".hg", "node_modules", "venv",
    ".venv", "env", ".env", "dist", "build", ".idea", ".vscode",
    CHROMA_PATH, CACHE_DIR,
}

# Allowed extensions for directory ingestion
INGESTABLE_EXTS = {"txt", "md", "py", "json", "pdf", "rst", "csv", "log"}

# Allowed extensions for read_file (prevent reading binaries)
READABLE_EXTS = {"txt", "md", "py", "json", "pdf", "rst", "csv", "log",
                 "html", "htm", "xml", "yaml", "yml", "toml", "ini", "cfg",
                 "js", "ts", "tsx", "jsx", "c", "cpp", "h", "hpp", "rs",
                 "go", "rb", "sh", "bash", "zsh", "ps1", "bat"}

os.makedirs(CACHE_DIR, exist_ok=True)


# =============================================================================
# Embedding & Vector DB
# =============================================================================
class LlamaCppEmbedder:
    """OpenAI-compatible embedding function for ChromaDB."""

    def __init__(self, url: str):
        self.url = url
        self.client = httpx.Client(timeout=60.0)

    def __call__(self, input):
        # ChromaDB may pass a single string or a list of strings
        if isinstance(input, str):
            input = [input]
        try:
            res = self.client.post(self.url, json={"input": input}).json()
            return [d["embedding"] for d in res["data"]]
        except Exception as e:
            raise RuntimeError(f"Embedding service error: {e}") from e

    def name(self):
        return "llama_cpp_embedder"


embedder       = LlamaCppEmbedder(EMBED_URL)
chroma_client   = chromadb.PersistentClient(path=CHROMA_PATH)
_vector_db: Optional[chromadb.api.Collection] = None


def vector_db():
    """Lazy accessor so a missing embedding service doesn't crash startup."""
    global _vector_db
    if _vector_db is None:
        try:
            _vector_db = chroma_client.get_or_create_collection(
                name=COLLECTION, embedding_function=embedder
            )
        except Exception as e:
            console.print(f"[red]Failed to initialise vector DB:[/red] {e}")
            raise
    return _vector_db


# =============================================================================
# Live "think" stream renderer
# =============================================================================
class ThinkStreamRenderer:
    """
    Streams content token-by-token to the console, routing text inside
    ildal ...  ldal tags to a dim-blue style and everything else to the
    default style. Buffers up to (len(tag) - 1) characters so that split
    tag markers are never printed prematurely.
    """

    def __init__(self, console: Console):
        self.console = console
        self.buffer  = ""
        self.in_think = False

    def feed(self, text: str) -> None:
        self.buffer += text
        while True:
            if self.in_think:
                end = self.buffer.find(THINK_CLOSE)
                if end != -1:
                    if end > 0:
                        self.console.print(self.buffer[:end], style="dim blue", end="")
                    self.buffer  = self.buffer[end + len(THINK_CLOSE):]
                    self.in_think = False
                    continue
                safe = max(0, len(self.buffer) - (len(THINK_CLOSE) - 1))
                if safe > 0:
                    self.console.print(self.buffer[:safe], style="dim blue", end="")
                    self.buffer = self.buffer[safe:]
                return
            else:
                start = self.buffer.find(THINK_OPEN)
                if start != -1:
                    if start > 0:
                        self.console.print(self.buffer[:start], end="")
                    self.buffer  = self.buffer[start + len(THINK_OPEN):]
                    self.in_think = True
                    continue
                safe = max(0, len(self.buffer) - (len(THINK_OPEN) - 1))
                if safe > 0:
                    self.console.print(self.buffer[:safe], end="")
                    self.buffer = self.buffer[safe:]
                return

    def flush(self) -> None:
        if self.buffer:
            self.console.print(
                self.buffer,
                style="dim blue" if self.in_think else "white",
                end="",
            )
        self.console.print()  # final newline


# =============================================================================
# Core chat (streaming, with retries)
# =============================================================================
def _post_stream(payload: dict):
    return httpx.stream(
        "POST",
        f"{LLM_URL}/v1/chat/completions",
        json=payload,
        timeout=LLM_TIMEOUT,
    )


def chat_stream(messages, tools=None, temperature=0.4, max_tokens=4096,
                tool_choice: Optional[str] = None):
    """
    Stream a chat completion from the local LLM.
    Returns (assistant_message_dict, usage_dict).
    """
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"

    tool_calls    = []
    content_buffer = ""
    usage         = {}

    renderer = ThinkStreamRenderer(console)

    last_err = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            with _post_stream(payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: [DONE]"):
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("usage"):
                        usage = chunk["usage"]

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    # ---- tool calls ----
                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            while len(tool_calls) <= idx:
                                tool_calls.append({
                                    "id": "", "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                })
                            if "id" in tc:
                                tool_calls[idx]["id"] = tc["id"]
                            fn = tc.get("function", {})
                            if "name" in fn:
                                tool_calls[idx]["function"]["name"] += fn["name"]
                            if "arguments" in fn:
                                tool_calls[idx]["function"]["arguments"] += fn["arguments"]

                    # ---- content ----
                    if "content" in delta and delta["content"]:
                        content_buffer += delta["content"]
                        if not tool_calls:
                            renderer.feed(delta["content"])

            # Stream finished successfully
            renderer.flush()
            msg = {"role": "assistant", "content": content_buffer}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            return msg, usage

        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError,
                httpx.ReadError) as e:
            last_err = e
            console.print(f"[yellow]LLM connection issue (attempt {attempt+1}/"
                          f"{LLM_MAX_RETRIES}): {e}[/yellow]")
            time.sleep(1.5 * (attempt + 1))
            continue
        except Exception as e:
            last_err = e
            console.print(f"[red]LLM stream error:[/red] {e}")
            break

    raise RuntimeError(f"LLM request failed after retries: {last_err}")


# =============================================================================
# Context management
# =============================================================================
def _estimate_tokens(messages) -> int:
    """Cheap heuristic; real count comes from server's `usage`."""
    chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    return chars // 4


def trim_context(messages):
    """
    Compact older tool messages and tool_call arguments to keep context small.
    Always preserves system + last 6 messages verbatim.
    """
    if len(messages) <= 8:
        return messages

    head = messages[:1]              # system prompt
    body = messages[1:-6]
    tail = messages[-6:]

    compacted = []
    for m in body:
        m = dict(m)  # don't mutate original
        if m.get("role") == "tool" and len(m.get("content", "")) > 200:
            m["content"] = "[Old tool output trimmed to save memory.]"
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            # Keep names so context survives; drop (often huge) arguments
            m["tool_calls"] = [
                {"id": tc.get("id", ""), "type": "function",
                 "function": {"name": tc["function"].get("name", ""),
                              "arguments": "{}"}}
                for tc in m["tool_calls"]
            ]
        compacted.append(m)
    return head + compacted + tail


# =============================================================================
# Tools — definitions & implementations
# =============================================================================
TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the internet for real-time information.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_page",
        "description": "Download and read the full text of a webpage URL.",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "search_local_kb",
        "description": "Search the local vector database (RAG) for previously "
                       "ingested documents.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "execute_python",
        "description": "Execute a multi-line Python script for math, data "
                       "analysis, or file processing.",
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string"}},
                       "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Save text content to a local file in the workspace.",
        "parameters": {"type": "object",
                       "properties": {"filename": {"type": "string"},
                                     "content": {"type": "string"}},
                       "required": ["filename", "content"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read the entire text content of a local file in the workspace.",
        "parameters": {"type": "object",
                       "properties": {"filepath": {"type": "string"}},
                       "required": ["filepath"]}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files recursively in the current workspace "
                       "with file sizes.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string",
                                               "default": "."}}}}},
]


def truncate(obj: Any, limit: int = MAX_TOOL_OUTPUT) -> Any:
    """Recursively truncate long strings inside arbitrary structures."""
    if isinstance(obj, str):
        return obj[:limit] + "…[TRUNCATED]" if len(obj) > limit else obj
    if isinstance(obj, dict):
        return {k: truncate(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [truncate(v, limit) for v in obj]
    return obj


# ---------- web_search ----------
def web_search(query: str, n: int = 3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=n))
        return [{"title": r.get("title", ""),
                 "url":   r.get("href") or r.get("url", ""),
                 "snippet": r.get("body", "")}
                for r in results]
    except Exception as e:
        return {"error": f"web_search failed: {e}"}


# ---------- fetch_page ----------
def fetch_page(url: str):
    cache_hash = hashlib.md5(url.encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{cache_hash}.txt")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return truncate({"url": url, "content": f.read()})
        except Exception:
            pass  # fall through to re-fetch

    try:
        res = httpx.get(url, timeout=FETCH_TIMEOUT, follow_redirects=True,
                        headers={"User-Agent": "Mozilla/5.0 (Genesis V5)"})
        res.raise_for_status()
        html = res.text

        text = trafilatura.extract(html, include_comments=False) or ""
        if not text or len(text) < 200:
            h2t = html2text.HTML2Text()
            h2t.ignore_links = False
            h2t.ignore_images = True
            h2t.body_width   = 0
            text = h2t.handle(html)

        if text:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass
        return truncate({"url": url, "content": text or "[Empty page]"})

    except Exception as e:
        return {"error": f"fetch_page failed: {e}"}


# ---------- search_local_kb ----------
def search_local_kb(query: str, n_results: int = 3):
    try:
        results = vector_db().query(query_texts=[query], n_results=n_results)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [{"source": m.get("source", "unknown"), "content": d}
                for d, m in zip(docs, metas)]
    except Exception as e:
        return {"error": f"search_local_kb failed: {e}"}


# ---------- execute_python ----------
def execute_python(code: str):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                         delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_path = f.name

        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True, text=True, timeout=PYTHON_TIMEOUT,
        )
        return {
            "stdout": truncate(result.stdout),
            "stderr": truncate(result.stderr),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Python execution timed out after {PYTHON_TIMEOUT}s"}
    except Exception as e:
        return {"error": f"execute_python failed: {e}"}
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ---------- write_file ----------
def write_file(filename: str, content: str):
    # Restrict to current working directory (no path traversal)
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name.startswith("."):
        return {"error": "Invalid filename."}
    try:
        with open(safe_name, "w", encoding="utf-8") as f:
            f.write(content)
        return {"success": True, "message": f"Saved to {safe_name}",
                "bytes": len(content)}
    except Exception as e:
        return {"error": f"write_file failed: {e}"}


# ---------- read_file ----------
def _is_within_workspace(path: str) -> bool:
    base   = os.path.abspath(".")
    target = os.path.abspath(path)
    return target == base or target.startswith(base + os.sep)


def read_file(filepath: str):
    if not _is_within_workspace(filepath):
        return {"error": "Refusing to read files outside the workspace."}
    if not os.path.exists(filepath):
        return {"error": "File not found"}
    if os.path.isdir(filepath):
        return {"error": "Path is a directory, not a file"}

    ext = filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""
    if ext == "pdf":
        try:
            text = ""
            with fitz.open(filepath) as doc:
                for page in doc:
                    text += page.get_text()
            if not text.strip():
                return {"error": "PDF appears to be scanned (no text layer).",
                        "filepath": filepath}
            return truncate({"filepath": filepath, "content": text})
        except Exception as e:
            return {"error": f"PDF read failed: {e}"}

    if ext and ext not in READABLE_EXTS:
        return {"error": f"Refusing to read .{ext} file (not in allowlist)."}

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return truncate({"filepath": filepath, "content": f.read()})
    except Exception as e:
        return {"error": f"read_file failed: {e}"}


# ---------- list_files ----------
def list_files(path: str = "."):
    if not _is_within_workspace(path):
        return {"error": "Refusing to list outside the workspace."}
    if not os.path.isdir(path):
        return {"error": "Path is not a directory"}

    files = []
    base_depth = os.path.abspath(path).rstrip(os.sep).count(os.sep)
    for root, dirs, filenames in os.walk(path):
        # Depth limit
        rel_depth = os.path.abspath(root).rstrip(os.sep).count(os.sep) - base_depth
        if rel_depth >= LIST_DEPTH_LIMIT:
            dirs[:] = []
            continue
        # Filter excluded dirs in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for f in filenames:
            if f.startswith("."):
                continue
            fp = os.path.join(root, f)
            try:
                files.append({"path": fp, "size": os.path.getsize(fp)})
            except OSError:
                continue
        if len(files) >= LIST_MAX_FILES:
            break

    return {"files": files[:LIST_MAX_FILES],
            "truncated": len(files) >= LIST_MAX_FILES}


# ---------- dispatcher ----------
def execute_tool(name: str, args: dict):
    try:
        if   name == "web_search":       return web_search(args.get("query", ""))
        elif name == "fetch_page":       return fetch_page(args.get("url", ""))
        elif name == "search_local_kb":  return search_local_kb(args.get("query", ""))
        elif name == "execute_python":   return execute_python(args.get("code", ""))
        elif name == "write_file":       return write_file(args.get("filename", ""),
                                                           args.get("content", ""))
        elif name == "read_file":        return read_file(args.get("filepath", ""))
        elif name == "list_files":       return list_files(args.get("path", "."))
        return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": f"Tool {name} raised: {e}",
                "traceback": traceback.format_exc()[-400:]}


# =============================================================================
# RAG Ingestion
# =============================================================================
def extract_text(filepath: str) -> str:
    if filepath.lower().endswith(".pdf"):
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        if not text.strip():
            raise ValueError("PDF has no extractable text (possibly scanned).")
        return text
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, max_words: int = 300, overlap: int = 50):
    words = text.split()
    if len(words) <= max_words:
        return [text] if text.strip() else []
    step = max(1, max_words - overlap)
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), step)]


def ingest_file(filepath: str) -> int:
    chunks = chunk_text(extract_text(filepath))
    if not chunks:
        return 0
    # Idempotent IDs so re-ingesting overwrites rather than duplicates
    doc_id_base = hashlib.md5(filepath.encode()).hexdigest()
    vector_db().add(
        documents=chunks,
        metadatas=[{"source": filepath} for _ in chunks],
        ids=[f"{doc_id_base}_{i}" for i in range(len(chunks))],
    )
    return len(chunks)


# =============================================================================
# Agent Loop
# =============================================================================
SYSTEM_PROMPT = (
    "You are Genesis V5, an elite autonomous local AI. "
    "Reason step-by-step inside ildal...ldal tags BEFORE acting.\n"
    "1. If asked about local files, ALWAYS use list_files first.\n"
    "2. If asked about documents/notes, use search_local_kb.\n"
    "3. If you lack real-time info, use web_search and fetch_page.\n"
    "4. For math or data tasks, write multi-line Python code and use execute_python. "
    "If your code returns an error, read the traceback, fix the code, and try again.\n"
    "5. Provide final answers in clean Markdown."
)


def _parse_tool_args(raw: str) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to salvage common LLM mistakes (single quotes, trailing commas)
        try:
            return json.loads(raw.replace("'", '"').rstrip(","))
        except Exception:
            return {}


def _strip_think(text: str) -> str:
    return re.sub(re.escape(THINK_OPEN) + r".*?" + re.escape(THINK_CLOSE),
                  "", text, flags=re.DOTALL).strip()


def _beep() -> None:
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


def run_agent_turn(messages, max_steps: int = MAX_STEPS):
    if not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    for step in range(max_steps):
        messages = trim_context(messages)

        # On the final step, encourage a final answer instead of more tools
        tool_choice = "required" if step == max_steps - 1 else "auto"

        console.print()  # spacing before streaming
        try:
            msg, usage = chat_stream(
                messages, tools=TOOL_DEFINITIONS,
                temperature=0.4, tool_choice=tool_choice,
            )
        except Exception as e:
            console.print(f"[red]LLM error:[/red] {e}")
            return messages

        raw_content = msg.get("content", "") or ""
        cleaned = _strip_think(raw_content)
        if cleaned:
            msg["content"] = cleaned
        else:
            msg.pop("content", None)

        # ---- Tool branch ----
        if msg.get("tool_calls"):
            messages.append(msg)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for call in msg["tool_calls"]:
                    func_name = call["function"].get("name", "")
                    func_args = _parse_tool_args(call["function"].get("arguments", ""))

                    display_args = {}
                    for k, v in func_args.items():
                        if isinstance(v, str) and len(v) > 60:
                            display_args[k] = v[:60] + "..."
                        else:
                            display_args[k] = truncate(v, 60)
                    console.print(f"\n🔧 [yellow]Calling:[/yellow] "
                                  f"{func_name}({display_args})")

                    futures.append(executor.submit(execute_tool, func_name, func_args))

                for i, future in enumerate(futures):
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {"error": f"Tool execution crashed: {e}"}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": msg["tool_calls"][i].get("id", ""),
                        "name": msg["tool_calls"][i]["function"].get("name", ""),
                        "content": json.dumps(truncate(result)),
                    })
            continue

        # ---- Final-answer branch ----
        if msg.get("content"):
            messages.append({"role": "assistant", "content": msg["content"]})
            if usage and usage.get("prompt_tokens"):
                tokens = usage["prompt_tokens"]
                pct = (tokens / CONTEXT_LIMIT) * 100 if CONTEXT_LIMIT else 0
                bar_len = 20
                filled = int(bar_len * tokens / CONTEXT_LIMIT) if CONTEXT_LIMIT else 0
                bar = "█" * filled + "░" * (bar_len - filled)
                console.print(f"\n[dim]Context: {tokens}/{CONTEXT_LIMIT} tokens "
                              f"[{bar}] {pct:.0f}%[/dim]")
            _beep()
            return messages

        # No content and no tool calls — model produced nothing
        console.print("[yellow]⚠ Model returned empty response, retrying...[/yellow]")
        messages.append({"role": "assistant",
                         "content": "[No content returned — please continue.]"})

    console.print("\n[red]⚠️ Reached max steps without final answer.[/red]")
    return messages


# =============================================================================
# History persistence
# =============================================================================
def save_history(messages):
    clean_msgs = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "tool" and len(m.get("content", "")) > 500:
            m["content"] = "[Tool output omitted]"
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m["tool_calls"] = [
                {"id": tc.get("id", ""), "type": "function",
                 "function": {"name": tc["function"].get("name", ""),
                              "arguments": "{}"}}
                for tc in m["tool_calls"]
            ]
        clean_msgs.append(m)

    # Hard cap on file size — keep only the most recent turns
    try:
        encoded = json.dumps(clean_msgs, ensure_ascii=False, indent=2)
        if len(encoded.encode("utf-8")) > MAX_HISTORY_BYTES:
            # Keep system prompt + last N messages
            keep_n = 20
            clean_msgs = ([clean_msgs[0]] if clean_msgs and
                          clean_msgs[0].get("role") == "system" else []) \
                         + clean_msgs[-keep_n:]
            encoded = json.dumps(clean_msgs, ensure_ascii=False, indent=2)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            f.write(encoded)
    except Exception as e:
        console.print(f"[red]Failed to save history:[/red] {e}")


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[red]Failed to load history:[/red] {e}")
        return []


# =============================================================================
# CLI commands
# =============================================================================
def cmd_ingest_file(fp: str):
    if not os.path.exists(fp):
        console.print(f"[red]File not found:[/red] {fp}")
        return
    try:
        n = ingest_file(fp)
        console.print(f"[green]✅ Ingested {n} chunks from {fp}.[/green]")
    except Exception as e:
        console.print(f"[red]Ingestion failed:[/red] {e}")


def cmd_ingest_dir(dirpath: str):
    if not os.path.isdir(dirpath):
        console.print(f"[red]Directory not found:[/red] {dirpath}")
        return
    total = 0
    failed = 0
    with console.status(f"[cyan]Ingesting '{dirpath}'...[/cyan]"):
        for fp in glob.glob(os.path.join(dirpath, "**", "*"), recursive=True):
            if not os.path.isfile(fp):
                continue
            ext = fp.rsplit(".", 1)[-1].lower() if "." in fp else ""
            if ext not in INGESTABLE_EXTS:
                continue
            try:
                total += ingest_file(fp)
            except Exception as e:
                failed += 1
                console.print(f"[yellow]Skipped {fp}: {e}[/yellow]")
    console.print(f"[green]✅ Ingested {total} chunks "
                  f"({failed} file(s) skipped).[/green]")


def cmd_forget():
    global _vector_db
    try:
        chroma_client.delete_collection(COLLECTION)
    except Exception:
        pass  # collection may not exist
    _vector_db = None  # force re-creation on next access
    # Re-create empty collection immediately
    try:
        vector_db()
        console.print("[red]🔥 Vector DB wiped and re-initialised.[/red]")
    except Exception as e:
        console.print(f"[red]Failed to re-create collection:[/red] {e}")


def cmd_clear(messages):
    messages.clear()
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
        except OSError:
            pass
    console.print("[yellow]Chat history cleared.[/yellow]")
    return messages


def show_help():
    console.print(Panel.fit(
        "[bold]Commands:[/bold]\n"
        "  /ingest <file>      Ingest a single file into the vector DB\n"
        "  /ingest_dir <dir>   Ingest all supported files in a directory tree\n"
        "  /forget             Wipe the vector database\n"
        "  /clear              Clear chat history\n"
        "  /tokens             Show estimated current context usage\n"
        "  /help               Show this help message\n"
        "  exit / quit         Save history and exit",
        title="Genesis V5",
        border_style="magenta",
    ))


# =============================================================================
# Interactive CLI
# =============================================================================
BANNER = r"""
[bold magenta]   ____            _   __     __  ____  _       _     
  / ___|___  _ __ | | / _\  /\ \ \/ ___|| | __ _| |__  
 | |   / _ \| '_ \| |/ / \ \/ / \___ \| |/ _` | '_ \ 
 | |__| (_) | |_) | / /  / /\ \  ___) | | (_| | |_) |
  \____\___/| .__/|_|_|  \/  \/ |____/|_|\__,_|_.__/ 
            |_|    v5  —  autonomous local AI[/bold magenta]
"""


def main():
    console.print(BANNER)
    console.print(f"[dim]LLM: {LLM_URL}   Embed: {EMBED_URL}   "
                   f"Context: {CONTEXT_LIMIT}[/dim]")

    messages = []
    if os.path.exists(HISTORY_FILE):
        try:
            ans = console.input("[yellow]Previous session found. Resume? (y/n): [/yellow]")
            if ans.strip().lower() == "y":
                messages = load_history()
                console.print(f"[green]✅ Session restored ({len(messages)} messages).[/green]")
        except Exception:
            pass

    show_help()
    console.print()

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Saving history before exit...[/yellow]")
            save_history(messages)
            break

        if not user_input:
            continue

        low = user_input.lower()

        if low in ("exit", "quit"):
            save_history(messages)
            console.print("[magenta]Goodbye.[/magenta]")
            break

        if low == "/clear":
            messages = cmd_clear(messages)
            continue

        if low == "/forget":
            cmd_forget()
            continue

        if low == "/help":
            show_help()
            continue

        if low == "/tokens":
            est = _estimate_tokens(messages)
            console.print(f"[cyan]Estimated context: ~{est} tokens "
                          f"(of {CONTEXT_LIMIT}).[/cyan]")
            continue

        if user_input.startswith("/ingest_dir "):
            cmd_ingest_dir(user_input.split(" ", 1)[1].strip())
            continue

        if user_input.startswith("/ingest "):
            cmd_ingest_file(user_input.split(" ", 1)[1].strip())
            continue

        if user_input.startswith("/"):
            console.print(f"[red]Unknown command:[/red] {user_input}")
            show_help()
            continue

        # ---- Normal user message ----
        messages.append({"role": "user", "content": user_input})
        try:
            messages = run_agent_turn(messages)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — saving partial history...[/yellow]")
            save_history(messages)
            continue
        except Exception as e:
            console.print(f"[red]Agent error:[/red] {e}")
            console.print(traceback.format_exc())

        save_history(messages)


if __name__ == "__main__":
    main()