# Memory Server

A shared "long-term memory" service for AI coding agents running on several machines. Agents save short notes (a title, the full content, tags, the device they came from) and later search them by meaning, so a fix found on one machine can be reused on another. It exposes the memory as an MCP (Model Context Protocol) server and also serves a small web page for browsing what is stored.

## Stack

- Python 3.11
- ChromaDB (persistent vector store) with `sentence-transformers` embeddings (`all-MiniLM-L6-v2`, cosine similarity)
- MCP Python SDK (streamable HTTP at `/mcp`, plus SSE at `/sse` for older clients)
- Starlette and Uvicorn
- Docker and Docker Compose

## Running

```bash
docker compose up --build
# web page and API: http://localhost:8765/
# MCP endpoints:    http://localhost:8765/mcp  and  http://localhost:8765/sse
```

Or without Docker:

```bash
pip install -r requirements.txt
python server.py          # listens on 0.0.0.0:8765; data in ./chroma_data
```

The Docker build downloads the embedding model so the first query is fast; data lives in the `memory_data` volume.

### Pointing agents at it

Add an MCP server entry named `memory` to the agent's settings with the server's URL. Two helper scripts do this for Raspberry Pi machines:

- `pi_setup.sh` - run on a Pi. Reads `MEMORY_SERVER` (default placeholder `http://<memory-server-ip>:8765/sse`) and merges an `mcpServers.memory` entry into `~/.claude/settings.json`. Needs `jq`.
- `configure_pis.sh <ssh-user> <ssh-password>` - from one machine, writes `~/.claude/mcp.json` and a `CLAUDE.md` of memory-usage guidance to each Pi over SSH. Reads `MEMORY_URL` and `PI_IPS` (space-separated). Needs `sshpass`.

Replace the placeholders with your own LAN addresses. `start.bat` is a Windows convenience launcher with a hard-coded path you will need to edit.

## MCP tools

- `save_memory` - store a memory (title, content, tags, device, optional hidden flag)
- `search_memory` - semantic search; returns the top matches and counts reads
- `list_recent_memories` - most recent entries
- `update_memory` - edit the best match for a query
- `delete_memory` - delete the best match for a query

## Project layout

- `server.py` - MCP server, REST endpoint `/api/memories`, and the HTML index page
- `Dockerfile`, `docker-compose.yml` - container build and run
- `pi_setup.sh`, `configure_pis.sh`, `start.bat` - client and launcher helpers

## Status

Working and in daily personal use. It has no authentication and listens on all interfaces, so run it only on a trusted network. There are no automated tests.
