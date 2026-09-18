"""FastAPI application.

Exposes exactly the two endpoints the judge harness exercises:

* ``GET  /health``           - readiness, returns ``{"status": "ok"}``
* ``POST /optimize-energy``  - interpretation + 24-hour optimized plan

HTTP status contract (from the canonical Problem Statement):

* ``200`` successful health or optimization response
* ``400`` malformed JSON or structurally invalid request
* ``422`` well-formed but semantically invalid request
* ``500`` controlled internal error - never a stack trace or secret

Error bodies are always ``{"error": {"code": ..., "message": ...}}`` with a
generic message, so no provider detail, credential, or prompt content can leak.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.config import get_settings
from app.errors import SemanticError, ServiceError
from app.optimization import solver
from app.pipeline import GridWiseService
from app.schemas import HealthResponse, OptimizeResponse, ScenarioRequest, semantic_problems

logger = logging.getLogger("gridwise")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    solver.warm_up()
    service = GridWiseService.from_settings(settings)
    app.state.service = service
    logger.info(
        "service ready | version=%s llm_configured=%s model=%s fallback=%s",
        __version__,
        settings.llm_configured,
        settings.llm_model if settings.llm_configured else "n/a",
        settings.allow_deterministic_fallback,
    )
    try:
        yield
    finally:
        await service.aclose()
        logger.info("service stopped")


app = FastAPI(
    title="GridWise LLM-Assisted Energy Optimizer",
    version=__version__,
    description=(
        "LLM-assisted operator-directive interpretation with deterministic "
        "guardrails and exact 24-hour cost optimization."
    ),
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Readiness probe used by the judge harness."""
    return HealthResponse(status="ok")


@app.post("/optimize-energy", response_model=OptimizeResponse, tags=["optimization"])
async def optimize_energy(request: ScenarioRequest, http_request: Request) -> OptimizeResponse:
    """Interpret the operator notes and return a cost-optimal valid plan."""
    problems = semantic_problems(request)
    if problems:
        raise SemanticError("; ".join(problems))

    service: GridWiseService = http_request.app.state.service
    return await service.optimize(request)


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


@app.exception_handler(ServiceError)
async def _service_error_handler(_: Request, exc: ServiceError) -> JSONResponse:
    if exc.status_code >= 500:
        logger.error("service error (%s): %s", exc.code, exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_payload())


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Malformed JSON and structural violations map to 400, not FastAPI's 422."""
    logger.info("rejected request: %s", exc.errors()[:3])
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "invalid_request",
                "message": "request body failed structural validation",
            }
        },
    )


@app.exception_handler(Exception)
async def _unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Last-resort guard: log internally, return a generic body."""
    logger.exception("unhandled error: %s", type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "internal server error"}},
    )


def main() -> None:  # pragma: no cover - process entry point
    """Run the service with uvicorn."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
