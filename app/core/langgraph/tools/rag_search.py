"""RAG search tool for LangGraph.

Provides a tool that queries the RAG API for relevant context and returns a
condensed summary of the retrieved passages for the agent to use.
"""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime, timedelta
from typing import Any, List, Optional, Type

import httpx
from jose import jwt
from langchain_core.tools.base import BaseTool
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger


class RAGQueryInput(BaseModel):
    """Input schema for the RAG search tool."""

    query: str = Field(..., description="Natural language question to search for.")
    file_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific file identifiers to search within. Leave empty to use the default knowledge base.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum number of passages to retrieve (defaults to environment setting).",
    )
    entity_id: Optional[str] = Field(
        default=None,
        description="Optional organization/entity identifier for access-controlled collections.",
    )


class RAGSearchTool(BaseTool):
    """Tool that queries the RAG API for relevant passages."""

    name: str = "rag_search"
    description: str = (
        "Use this tool to look up factual information from the internal knowledge base. "
        "Pass a clear question and optionally restrict by file identifiers."
    )
    args_schema: Type[RAGQueryInput] = RAGQueryInput

    def __init__(self) -> None:
        super().__init__()
        self._base_url = settings.RAG_BASE_URL.rstrip("/")
        self._query_endpoint = self._ensure_leading_slash(settings.RAG_QUERY_ENDPOINT)
        self._query_multiple_endpoint = self._ensure_leading_slash(settings.RAG_QUERY_MULTIPLE_ENDPOINT)
        self._default_file_ids = [fid for fid in settings.RAG_DEFAULT_FILE_IDS if fid]
        self._default_top_k = max(1, settings.RAG_TOP_K)
        self._entity_id = settings.RAG_ENTITY_ID
        self._timeout = settings.RAG_TIMEOUT_SECONDS
        self._jwt_secret = settings.RAG_JWT_SECRET
        self._jwt_algorithm = settings.RAG_JWT_ALGORITHM
        self._jwt_ttl_seconds = max(10, settings.RAG_JWT_TTL_SECONDS)
        self._service_subject = settings.RAG_SERVICE_SUBJECT or "agent_service"

    def _ensure_leading_slash(self, value: str) -> str:
        if not value:
            return ""
        return value if value.startswith("/") else f"/{value}"

    def _generate_jwt(self) -> Optional[str]:
        """Generate a short-lived JWT for authenticating with the RAG API."""
        if not self._jwt_secret:
            return None

        now = datetime.now(UTC)
        payload = {
            "sub": self._service_subject,
            "iat": now,
            "exp": now + timedelta(seconds=self._jwt_ttl_seconds),
        }

        try:
            return jwt.encode(payload, self._jwt_secret, algorithm=self._jwt_algorithm)
        except Exception as exc:
            logger.error("rag_jwt_generation_failed", error=str(exc))
            return None

    def _prepare_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if token := self._generate_jwt():
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _compose_payload(
        self,
        query: str,
        file_ids: List[str],
        top_k: int,
        entity_id: Optional[str],
    ) -> tuple[str, dict[str, Any]]:
        is_multiple = len(file_ids) > 1
        endpoint = self._query_multiple_endpoint if is_multiple else self._query_endpoint

        payload: dict[str, Any] = {"query": query, "k": top_k}
        if is_multiple:
            payload["file_ids"] = file_ids
        else:
            payload["file_id"] = file_ids[0]

        resolved_entity_id = entity_id or self._entity_id
        if resolved_entity_id:
            payload["entity_id"] = resolved_entity_id

        return endpoint, payload

    def _format_results(self, data: Any) -> str:
        if not data:
            return "No relevant context found in the knowledge base."

        formatted_chunks: List[str] = []

        for idx, item in enumerate(data, start=1):
            document = None
            score = None

            if isinstance(item, list) and len(item) >= 1:
                document = item[0]
                if len(item) > 1:
                    score = item[1]
            elif isinstance(item, dict):
                document = item.get("document") or item
                score = item.get("score")

            if not isinstance(document, dict):
                continue

            page_content = document.get("page_content") or document.get("content") or ""
            metadata = document.get("metadata") or {}

            snippet = textwrap.shorten(" ".join(page_content.split()), width=320, placeholder="...")
            source = metadata.get("source") or metadata.get("filename") or metadata.get("file_id") or "unknown source"
            score_text = f"{score:.3f}" if isinstance(score, (float, int)) else "n/a"

            formatted_chunks.append(f"{idx}. score={score_text} source={source}: {snippet}")

        if not formatted_chunks:
            return "No relevant context found in the knowledge base."

        return "\n".join(formatted_chunks)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("rag_search tool is async-only")

    async def _arun(
        self,
        query: str,
        file_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        entity_id: Optional[str] = None,
    ) -> str:
        if not query or not query.strip():
            return "A search query is required to use the knowledge base."

        resolved_file_ids = file_ids or self._default_file_ids
        resolved_file_ids = [fid for fid in resolved_file_ids if fid]
        if not resolved_file_ids:
            return (
                "No file identifiers were provided or configured for the RAG knowledge base. "
                "Add file IDs via `file_ids` argument or configure `RAG_DEFAULT_FILE_IDS`."
            )

        resolved_top_k = top_k or self._default_top_k
        if resolved_top_k < 1:
            resolved_top_k = 1

        endpoint, payload = self._compose_payload(
            query=query.strip(),
            file_ids=resolved_file_ids,
            top_k=resolved_top_k,
            entity_id=entity_id,
        )
        url = f"{self._base_url}{endpoint}"

        headers = self._prepare_headers()

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "rag_query_failed_status",
                status_code=exc.response.status_code,
                response=exc.response.text,
                url=url,
            )
            return (
                f"RAG search failed with status {exc.response.status_code}. "
                "Verify the file identifiers and query parameters."
            )
        except httpx.TimeoutException:
            logger.error("rag_query_timeout", url=url, timeout=self._timeout)
            return "RAG search timed out. Please try again with a simpler query."
        except httpx.HTTPError as exc:
            logger.error("rag_query_http_error", error=str(exc), url=url)
            return f"RAG search failed due to a network error: {exc}"
        except ValueError as exc:
            logger.error("rag_query_json_decode_failed", error=str(exc), url=url)
            return "RAG search returned an unexpected response format."

        logger.debug(
            "rag_query_success",
            results=len(data) if isinstance(data, list) else "unknown",
            url=url,
            top_k=resolved_top_k,
        )

        return self._format_results(data)


rag_search_tool = RAGSearchTool()
