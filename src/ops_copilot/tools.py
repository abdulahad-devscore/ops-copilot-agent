"""The agent's tools: ERP tools from the ERP MCP Bridge (Project 1) plus a local
`search_documents` tool over the RAG index.

The agent holds no ERP credentials beyond an OAuth client for the bridge. When
a request carries the end user's own bridge token, calls are made with that
token instead, so the user's scopes and warehouse restrictions apply.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

SEARCH_TOOL = {
    "name": "search_documents",
    "description": (
        "Search Tembusu's policies, SOPs, price lists, product sheets, supplier and site documents. "
        "Returns numbered sources [S#] to cite. Only returns documents the current user may read."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

MAX_RESULT_CHARS = 12_000


@dataclass
class ToolResult:
    text: str
    is_error: bool
    data: Any = None


class ErpTools(Protocol):
    async def list_tools(self) -> list[dict]: ...

    async def call(self, name: str, args: dict, token: str | None = None) -> ToolResult: ...


class BridgeAuth:
    """OAuth client-credentials token for the agent's own service account, cached."""

    def __init__(self, bridge_url: str, client_id: str, client_secret: str):
        self.url = f"{bridge_url.rstrip('/')}/oauth/token"
        self.client_id = client_id
        self.client_secret = client_secret
        self._token: str | None = None
        self._expires = 0.0

    async def token(self) -> str:
        if self._token and time.time() < self._expires - 60:
            return self._token
        async with httpx.AsyncClient(timeout=10, trust_env=False) as http:
            resp = await http.post(
                self.url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )
            resp.raise_for_status()
            body = resp.json()
        self._token, self._expires = body["access_token"], time.time() + body["expires_in"]
        return self._token


class McpErpTools:
    def __init__(self, bridge_url: str, auth: BridgeAuth):
        self.mcp_url = f"{bridge_url.rstrip('/')}/mcp"
        self.auth = auth
        self._tools: list[dict] | None = None

    def _client(self, token: str) -> Client:
        http = httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {token}"}, trust_env=False, timeout=httpx2.Timeout(60.0)
        )
        return Client(streamable_http_client(self.mcp_url, http_client=http))

    async def list_tools(self) -> list[dict]:
        if self._tools is None:
            async with self._client(await self.auth.token()) as client:
                listed = (await client.list_tools()).tools
            self._tools = [
                {"name": t.name, "description": t.description or "", "input_schema": t.input_schema} for t in listed
            ]
        return self._tools

    async def call(self, name: str, args: dict, token: str | None = None) -> ToolResult:
        async with self._client(token or await self.auth.token()) as client:
            result = await client.call_tool(name, args)
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content) or "Tool error"
            return ToolResult(text=text, is_error=True)
        data = result.structured_content
        if isinstance(data, dict) and set(data) == {"result"}:
            data = data["result"]
        text = (
            json.dumps(data, default=str)
            if data is not None
            else " ".join(getattr(c, "text", "") for c in result.content)
        )
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + " ...[truncated: narrow the query]"
        return ToolResult(text=text, is_error=False, data=data)
