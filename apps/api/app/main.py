from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent.controller import SatQueryAgent
from app.api.routes.v1 import router
from app.core.config import get_settings
from app.core.exceptions import SatQueryError
from app.core.logging import configure_logging
from app.models.manager import detect_device
from app.registry.model_registry import ModelRegistry
from app.registry.tool_registry import ToolRegistry
from app.services.analysis_service import AnalysisService
from app.services.history_service import HistoryService
from app.services.report_service import ReportService

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()
device = detect_device(settings.device)
history = HistoryService(settings.data_dir)
reports = ReportService(settings.data_dir)
model_registry = ModelRegistry(settings.model_dir.resolve(), device)
tool_registry = ToolRegistry()
analysis = AnalysisService(settings, model_registry, tool_registry, history)

app = FastAPI(
    title="SatQuery AI API",
    version="0.1.0",
    description="Local-first agentic remote-sensing analysis API.",
)
app.state.settings = settings
app.state.device = device
app.state.history = history
app.state.reports = reports
app.state.model_registry = model_registry
app.state.tool_registry = tool_registry
app.state.agent = SatQueryAgent(analysis)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.mount("/assets", StaticFiles(directory=settings.data_dir.resolve()), name="assets")


@app.exception_handler(SatQueryError)
async def satquery_exception_handler(_: Request, exc: SatQueryError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})


@app.exception_handler(Exception)
async def unexpected_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unexpected backend error")
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "The local analysis failed unexpectedly. Check the API logs for details."}},
    )
