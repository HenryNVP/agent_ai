"""UI routes for the lightweight testing interface."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings

router = APIRouter(tags=["ui"])

_TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=_TEMPLATE_DIR)


@router.get("/ui", response_class=HTMLResponse)
async def render_ui(request: Request) -> HTMLResponse:
    """Serve the mock chat + document upload interface."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_file_id": settings.RAG_DEFAULT_FILE_IDS[0] if settings.RAG_DEFAULT_FILE_IDS else "",
            "project_name": settings.PROJECT_NAME,
        },
    )

