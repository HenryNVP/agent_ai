"""Document management endpoints for the mock UI.

These routes proxy document operations to the RAG service so the UI can
upload and inspect knowledge-base files without having to know internal
network details or credentials.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from jose import jwt

from app.api.v1.auth import get_current_user
from app.core.config import settings
from app.core.logging import logger
from app.models.user import User

router = APIRouter(prefix="/documents", tags=["documents"])

_DEFAULT_TIMEOUT = settings.RAG_TIMEOUT_SECONDS + 20


def _normalize_file_id(value: Optional[str]) -> str:
    """Convert the provided value into a safe identifier."""
    if value:
        cleaned = value.strip().lower()
        cleaned = re.sub(r"[^a-z0-9_\-]", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if cleaned:
            return cleaned

    # Fall back to configured defaults or generate a short UUID tag
    if settings.RAG_DEFAULT_FILE_IDS:
        return settings.RAG_DEFAULT_FILE_IDS[0]
    return f"doc_{uuid.uuid4().hex[:8]}"


def _generate_rag_jwt() -> Optional[str]:
    """Create a short lived JWT compatible with the RAG service middleware."""
    secret = settings.RAG_JWT_SECRET
    if not secret:
        return None

    now = datetime.now(UTC)
    expires = now + timedelta(seconds=max(30, settings.RAG_JWT_TTL_SECONDS))
    payload = {
        "sub": settings.RAG_SERVICE_SUBJECT or "agent_service",
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }
    try:
        return jwt.encode(payload, secret, algorithm=settings.RAG_JWT_ALGORITHM)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("rag_jwt_generation_failed", error=str(exc))
        return None


def _rag_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if token := _generate_rag_jwt():
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _rag_request(
    method: str,
    path: str,
    *,
    data: Optional[dict[str, Any]] = None,
    files: Optional[dict[str, Any]] = None,
) -> Any:
    """Perform an HTTP request against the RAG API."""
    base_url = settings.RAG_BASE_URL.rstrip("/")
    url = f"{base_url}{path}"

    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
        response = await client.request(method, url, data=data, files=files, headers=_rag_headers())

    if response.status_code >= 400:
        detail = response.text
        logger.error("rag_proxy_request_failed", method=method, url=url, status=response.status_code, detail=detail)
        raise HTTPException(
            status_code=response.status_code,
            detail=f"RAG request failed: {detail or response.reason_phrase}",
        )

    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()

    return response.text


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document to the RAG knowledge base",
)
async def upload_document(
    file: UploadFile = File(...),
    file_id: Optional[str] = Form(None),
    entity_id: Optional[str] = Form(None),
    _: User = Depends(get_current_user),
):
    """Upload a document and index it in the RAG vector store."""
    resolved_file_id = _normalize_file_id(file_id)
    resolved_entity = entity_id or settings.RAG_ENTITY_ID

    payload: dict[str, Any] = {"file_id": resolved_file_id}
    if resolved_entity:
        payload["entity_id"] = resolved_entity

    file_bytes = await file.read()
    files = {"file": (file.filename, file_bytes, file.content_type or "application/octet-stream")}

    result = await _rag_request("POST", "/embed", data=payload, files=files)
    logger.info(
        "rag_document_uploaded",
        file_id=resolved_file_id,
        filename=file.filename,
        size=len(file_bytes),
    )

    return {
        "message": "Document uploaded successfully",
        "file_id": resolved_file_id,
        "rag_response": result,
    }


@router.get(
    "/ids",
    summary="List indexed document identifiers",
)
async def list_document_ids(_: User = Depends(get_current_user)):
    """Return all known file identifiers from the RAG store."""
    result = await _rag_request("GET", "/ids")
    return {"ids": result}


@router.get(
    "/{file_id}/preview",
    summary="Fetch a compact preview for a document",
)
async def preview_document(file_id: str, _: User = Depends(get_current_user)):
    """Fetch condensed context for the provided document identifier."""
    resolved_file_id = _normalize_file_id(file_id)
    result = await _rag_request("GET", f"/documents/{resolved_file_id}/context")
    return {"file_id": resolved_file_id, "chunks": result}

