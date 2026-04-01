#!/usr/bin/env python3
"""
Memory Server — Shared vector memory for distributed Claude/OpenClaw agents.
Runs on 192.0.2.99:8765, accessible by Raspberry Pis at 192.0.2.101-103.
"""

import json
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import logging

import chromadb
import uvicorn
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from mcp.server import Server
from mcp.server.fastmcp import FastMCP as FastMCPServer
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

PORT = 8765
CHROMA_PATH = "./chroma_data"
COLLECTION_NAME = "agent_memory"

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
db = chromadb.PersistentClient(path=CHROMA_PATH)
ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = db.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"},
)


def _now() -> str:
    return datetime.now(ZoneInfo("America/Chicago")).isoformat()


def _migrate():
    """Backfill created_at, read_count, last_read on entries that predate these fields."""
    all_items = collection.get(include=["metadatas"])
    if not all_items["ids"]:
        return
    now = _now()
    updates_ids, updates_metas = [], []
    for mid, meta in zip(all_items["ids"], all_items["metadatas"]):
        changed = False
        if "created_at" not in meta:
            meta["created_at"] = now
            changed = True
        if "read_count" not in meta:
            meta["read_count"] = 0
            changed = True
        if "last_read" not in meta:
            meta["last_read"] = ""
            changed = True
        if changed:
            updates_ids.append(mid)
            updates_metas.append(meta)
    if updates_ids:
        collection.update(ids=updates_ids, metadatas=updates_metas)
        print(f"Migrated {len(updates_ids)} entries with new fields.")


_migrate()

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
mcp = Server("memory-server")


@mcp.list_tools()
async def list_tools():
    return [
        Tool(
            name="save_memory",
            description=(
                "Save anything worth remembering to shared agent memory — "
                "solutions, notes, configurations, facts, or recurring patterns. "
                "Call this after solving a non-trivial problem, learning something useful, "
                "or completing a setup that wasn't straightforward."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "memory": {
                        "type": "string",
                        "description": "A short, searchable title or summary of what this memory is about.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to store — commands, code, notes, steps, facts, anything.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional topic tags, e.g. ['networking', 'apt', 'opencv', 'python']",
                    },
                    "device": {
                        "type": "string",
                        "description": "Hostname or identifier of the device saving this memory.",
                    },
                    "hidden": {
                        "type": "boolean",
                        "description": "If true, this memory is hidden from the browser UI but still searchable by agents.",
                    },
                },
                "required": ["memory", "content"],
            },
        ),
        Tool(
            name="search_memory",
            description=(
                "Semantic search over shared agent memory. "
                "Call this before attempting a complex or unfamiliar task — "
                "another agent or a prior session may have already figured it out."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what you're looking for.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 3, max 10).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_recent_memories",
            description="List the most recently saved memories across all agents and devices.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max entries to return (default 10).",
                        "default": 10,
                    }
                },
            },
        ),
        Tool(
            name="update_memory",
            description=(
                "Update an existing memory in place. Use this when a past memory is outdated or incomplete "
                "rather than saving a duplicate. Finds the closest match to your query and overwrites it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find the memory you want to update.",
                    },
                    "memory": {
                        "type": "string",
                        "description": "New title/summary for the memory.",
                    },
                    "content": {
                        "type": "string",
                        "description": "New full content to replace the existing content.",
                    },
                    "tags": {
                        "description": "Updated tags as an array of strings.",
                    },
                    "device": {
                        "type": "string",
                        "description": "Device making the update.",
                    },
                    "hidden": {
                        "type": "boolean",
                        "description": "Set to true to hide from browser UI while keeping searchable by agents. Set to false to unhide.",
                    },
                },
                "required": ["query", "memory", "content"],
            },
        ),
        Tool(
            name="delete_memory",
            description=(
                "Delete a memory that is no longer accurate or relevant. "
                "Finds the closest match to your query and removes it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find the memory you want to delete.",
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "save_memory":
        return await _save_memory(**arguments)
    elif name == "search_memory":
        return await _search_memory(**arguments)
    elif name == "list_recent_memories":
        return await _list_recent(**arguments)
    elif name == "update_memory":
        return await _update_memory(**arguments)
    elif name == "delete_memory":
        return await _delete_memory(**arguments)
    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

DEDUP_THRESHOLD = 0.88  # cosine similarity above this = likely duplicate


async def _save_memory(
    memory: str,
    content: str,
    tags: list[str] | None = None,
    device: str | None = None,
    hidden: bool = False,
):
    now = _now()

    # Deduplication: if a very similar memory already exists, update it instead
    if collection.count() > 0:
        results = collection.query(
            query_texts=[f"{memory}\n{content}"],
            n_results=1,
        )
        if results["ids"][0]:
            similarity = 1 - results["distances"][0][0]
            if similarity >= DEDUP_THRESHOLD:
                existing_id = results["ids"][0][0]
                existing_meta = results["metadatas"][0][0]
                collection.update(
                    ids=[existing_id],
                    documents=[f"{memory}\n{content}"],
                    metadatas=[{
                        **existing_meta,
                        "memory": memory,
                        "content": content,
                        "tags": json.dumps(tags or json.loads(existing_meta.get("tags", "[]"))),
                        "device": device or existing_meta.get("device", "unknown"),
                        "last_read": existing_meta.get("last_read", ""),
                        "hidden": hidden,
                    }],
                )
                return [TextContent(
                    type="text",
                    text=f"Updated existing memory ({round(similarity*100,1)}% similar, id: {existing_id[:8]}...) instead of creating duplicate."
                )]

    memory_id = str(uuid.uuid4())
    collection.add(
        ids=[memory_id],
        documents=[f"{memory}\n{content}"],
        metadatas=[{
            "memory": memory,
            "content": content,
            "tags": json.dumps(tags or []),
            "device": device or "unknown",
            "created_at": now,
            "read_count": 0,
            "last_read": "",
            "hidden": hidden,
        }],
    )
    return [TextContent(
        type="text",
        text=f"Saved (id: {memory_id[:8]}...) | device: {device or 'unknown'} | {now[:10]}"
    )]


async def _update_memory(
    query: str,
    memory: str,
    content: str,
    tags: list[str] | None = None,
    device: str | None = None,
    hidden: bool | None = None,
):
    if collection.count() == 0:
        return [TextContent(type="text", text="Memory is empty — nothing to update.")]

    results = collection.query(query_texts=[query], n_results=1)
    if not results["ids"][0]:
        return [TextContent(type="text", text="No matching memory found.")]

    existing_id = results["ids"][0][0]
    existing_meta = results["metadatas"][0][0]
    similarity = round((1 - results["distances"][0][0]) * 100, 1)

    # Coerce tags: MCP SDK may deliver array as JSON string
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except (json.JSONDecodeError, ValueError):
            tags = [t.strip() for t in tags.strip("[]").split(",") if t.strip()]

    updated_hidden = hidden if hidden is not None else existing_meta.get("hidden", False)
    collection.update(
        ids=[existing_id],
        documents=[f"{memory}\n{content}"],
        metadatas=[{
            **existing_meta,
            "memory": memory,
            "content": content,
            "tags": json.dumps(tags or json.loads(existing_meta.get("tags", "[]"))),
            "device": device or existing_meta.get("device", "unknown"),
            "hidden": updated_hidden,
        }],
    )
    return [TextContent(
        type="text",
        text=f"Updated memory ({similarity}% match, id: {existing_id[:8]}...)\nTitle: {memory}"
    )]


async def _delete_memory(query: str):
    if collection.count() == 0:
        return [TextContent(type="text", text="Memory is empty — nothing to delete.")]

    results = collection.query(query_texts=[query], n_results=1)
    if not results["ids"][0]:
        return [TextContent(type="text", text="No matching memory found.")]

    existing_id = results["ids"][0][0]
    existing_meta = results["metadatas"][0][0]
    similarity = round((1 - results["distances"][0][0]) * 100, 1)

    collection.delete(ids=[existing_id])
    return [TextContent(
        type="text",
        text=f"Deleted memory ({similarity}% match, id: {existing_id[:8]}...)\nWas: {existing_meta.get('memory', '?')}"
    )]


def _increment_reads(ids: list[str], metadatas: list[dict]):
    """Increment read_count and update last_read for a set of retrieved entries."""
    now = _now()
    updated = []
    for meta in metadatas:
        updated.append({**meta, "read_count": int(meta.get("read_count", 0)) + 1, "last_read": now})
    collection.update(ids=ids, metadatas=updated)


async def _search_memory(query: str, n_results: int = 3):
    count = collection.count()
    if count == 0:
        return [TextContent(type="text", text="Memory is empty — nothing saved yet.")]

    results = collection.query(
        query_texts=[query],
        n_results=min(max(1, n_results), 10, count),
    )

    if not results["ids"][0]:
        return [TextContent(type="text", text="No relevant memories found.")]

    _increment_reads(results["ids"][0], results["metadatas"][0])

    lines = []
    for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
        similarity = round((1 - dist) * 100, 1)
        tags = json.loads(meta.get("tags", "[]"))
        lines.append(
            f"[{i+1}] {similarity}% match | device: {meta.get('device', '?')} | saved: {meta.get('created_at', '')[:10]} | reads: {int(meta.get('read_count', 0)) + 1}\n"
            f"MEMORY: {meta['memory']}\n"
            f"CONTENT: {meta['content']}\n"
            f"TAGS: {', '.join(tags) if tags else 'none'}"
        )

    return [TextContent(type="text", text="\n\n---\n\n".join(lines))]


async def _list_recent(limit: int = 10):
    count = collection.count()
    if count == 0:
        return [TextContent(type="text", text="Memory is empty.")]

    results = collection.get(limit=min(limit, count), include=["metadatas"])
    metas = results["metadatas"]
    ids = results["ids"]

    # Sort by created_at descending
    paired = sorted(zip(ids, metas), key=lambda x: x[1].get("created_at", ""), reverse=True)
    ids_sorted = [p[0] for p in paired]
    metas_sorted = [p[1] for p in paired]

    _increment_reads(ids_sorted, metas_sorted)

    lines = []
    for meta in metas_sorted:
        lines.append(
            f"[{meta.get('created_at', '')[:19]}] {meta.get('device', '?')} | reads: {int(meta.get('read_count', 0)) + 1}\n"
            f"  {meta['memory'][:120]}{'...' if len(meta['memory']) > 120 else ''}"
        )

    return [TextContent(type="text", text=f"{count} total memories\n\n" + "\n".join(lines))]


# ---------------------------------------------------------------------------
# Browser UI
# ---------------------------------------------------------------------------

async def handle_index(request: Request):
    q = request.query_params.get("q", "").strip()
    count = collection.count()

    if q and count > 0:
        results = collection.query(query_texts=[q], n_results=min(10, count))
        entries = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            if meta.get("hidden"):
                continue
            entries.append({**meta, "similarity": round((1 - dist) * 100, 1), "tags": json.loads(meta.get("tags", "[]"))})
    elif count > 0:
        raw = collection.get(limit=50, include=["metadatas"])
        entries = [{**m, "similarity": None, "tags": json.loads(m.get("tags", "[]"))} for m in raw["metadatas"] if not m.get("hidden")]
        entries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    else:
        entries = []

    rows = ""
    for e in entries:
        sim = f'<span class="badge">{e["similarity"]}% match</span>' if e["similarity"] is not None else ""
        tags_html = "".join(f'<span class="tag">{t}</span>' for t in e["tags"]) if e["tags"] else ""
        read_count = int(e.get("read_count", 0))
        last_read = e.get("last_read", "")
        last_read_str = f' · last read {last_read[:10]}' if last_read else ''
        rows += f"""
        <div class="card">
          <div class="card-meta">
            <span class="device">{e.get("device","?")}</span>
            <span class="ts">{e.get("created_at","")[:19].replace("T"," ")}</span>
            {sim}
            {tags_html}
            <span class="reads">{read_count} read{"s" if read_count != 1 else ""}{last_read_str}</span>
          </div>
          <div class="label">Memory</div>
          <div class="body">{e.get("memory","")}</div>
          <div class="label">Content</div>
          <div class="body content">{e.get("content","")}</div>
        </div>"""

    empty = '<p class="empty">No memories yet. Agents will populate this as they work.</p>' if not entries else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Memory</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 2rem; }}
  h1 {{ font-size: 1.4rem; font-weight: 600; color: #a78bfa; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #64748b; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  form {{ display: flex; gap: 0.5rem; margin-bottom: 1.5rem; }}
  input {{ flex: 1; padding: 0.6rem 1rem; background: #1e2130; border: 1px solid #2d3148; border-radius: 8px; color: #e2e8f0; font-size: 0.95rem; outline: none; }}
  input:focus {{ border-color: #a78bfa; }}
  button {{ padding: 0.6rem 1.2rem; background: #7c3aed; border: none; border-radius: 8px; color: white; cursor: pointer; font-size: 0.95rem; }}
  button:hover {{ background: #6d28d9; }}
  .card {{ background: #1e2130; border: 1px solid #2d3148; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1rem; }}
  .card-meta {{ display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem; }}
  .device {{ font-weight: 600; color: #a78bfa; font-size: 0.85rem; }}
  .ts {{ color: #475569; font-size: 0.8rem; }}
  .reads {{ color: #475569; font-size: 0.75rem; margin-left: auto; }}
  .badge {{ background: #14532d; color: #4ade80; font-size: 0.75rem; padding: 0.15rem 0.5rem; border-radius: 99px; }}
  .tag {{ background: #1e3a5f; color: #60a5fa; font-size: 0.75rem; padding: 0.15rem 0.5rem; border-radius: 99px; }}
  .label {{ font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-top: 0.6rem; margin-bottom: 0.2rem; }}
  .body {{ font-size: 0.9rem; color: #cbd5e1; white-space: pre-wrap; word-break: break-word; }}
  .content {{ color: #86efac; font-family: 'Menlo', 'Consolas', monospace; font-size: 0.85rem; background: #0d1f0d; padding: 0.5rem 0.75rem; border-radius: 6px; }}
  .empty {{ color: #475569; text-align: center; margin-top: 3rem; }}
  .count {{ color: #64748b; font-size: 0.85rem; margin-bottom: 1rem; }}
</style>
</head>
<body>
  <h1>Agent Memory</h1>
  <p class="subtitle">Shared knowledge base across all Pi agents — {count} {"entry" if count == 1 else "entries"}</p>
  <form method="get">
    <input name="q" placeholder="Search memories semantically…" value="{q}" autofocus>
    <button type="submit">Search</button>
    {"<a href='/' style='padding:0.6rem 1rem;color:#64748b;text-decoration:none;font-size:0.9rem'>Clear</a>" if q else ""}
  </form>
  {f'<p class="count">{len(entries)} result{"s" if len(entries)!=1 else ""} for &ldquo;{q}&rdquo;</p>' if q else ""}
  {rows}
  {empty}
</body>
</html>"""
    return HTMLResponse(html)


async def handle_api_memories(request: Request):
    count = collection.count()
    if count == 0:
        return JSONResponse({"total": 0, "memories": []})
    raw = collection.get(limit=200, include=["metadatas"])
    memories = []
    for mid, meta in zip(raw["ids"], raw["metadatas"]):
        if meta.get("hidden"):
            continue
        memories.append({
            "id": mid,
            **meta,
            "tags": json.loads(meta.get("tags", "[]")),
            "read_count": int(meta.get("read_count", 0)),
        })
    memories.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return JSONResponse({"total": count, "memories": memories})


# ---------------------------------------------------------------------------
# FastMCP — streamable HTTP transport (for Claude Code on this PC)
# Served at /mcp — no SSE session initialization race condition.
# ---------------------------------------------------------------------------
fmcp = FastMCPServer("memory-server")


@fmcp.tool(description="Save anything worth remembering to shared agent memory — solutions, notes, configurations, facts, or recurring patterns. Call this after solving a non-trivial problem, learning something useful, or completing a setup that wasn't straightforward.")
async def save_memory(memory: str, content: str, tags: list[str] | None = None, device: str | None = None, hidden: bool = False) -> str:
    result = await _save_memory(memory=memory, content=content, tags=tags, device=device, hidden=hidden)
    return result[0].text


@fmcp.tool(description="Semantic search over shared agent memory. Call this before attempting a complex or unfamiliar task — another agent or a prior session may have already figured it out.")
async def search_memory(query: str, n_results: int = 3) -> str:
    result = await _search_memory(query=query, n_results=n_results)
    return result[0].text


@fmcp.tool(description="List the most recently saved memories, newest first.")
async def list_recent_memories(limit: int = 10) -> str:
    result = await _list_recent(limit=limit)
    return result[0].text


@fmcp.tool(description="Update an existing memory that has become outdated or needs correction.")
async def update_memory(query: str, memory: str, content: str, tags: list[str] | None = None, device: str | None = None, hidden: bool | None = None) -> str:
    result = await _update_memory(query=query, memory=memory, content=content, tags=tags, device=device, hidden=hidden)
    return result[0].text


@fmcp.tool(description="Delete a memory that is no longer accurate or relevant.")
async def delete_memory(query: str) -> str:
    result = await _delete_memory(query=query)
    return result[0].text


# ---------------------------------------------------------------------------
# Legacy SSE transport (kept for Raspberry Pi agents at 192.0.2.101-103)
# ---------------------------------------------------------------------------
sse = SseServerTransport("/messages/")


async def handle_sse(request: Request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as (read_stream, write_stream):
        await mcp.run(read_stream, write_stream, mcp.create_initialization_options())


app = Starlette(
    routes=[
        Route("/", endpoint=handle_index),
        Route("/api/memories", endpoint=handle_api_memories),
        Mount("/mcp", app=fmcp.streamable_http_app()),
        Route("/sse", endpoint=handle_sse),
        Mount("/messages", app=sse.handle_post_message),
    ]
)

if __name__ == "__main__":
    print(f"Memory Server starting on 0.0.0.0:{PORT}")
    print(f"  Local:   http://localhost:{PORT}/")
    print(f"  Network: http://192.0.2.99:{PORT}/")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
