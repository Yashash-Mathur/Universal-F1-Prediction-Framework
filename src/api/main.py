"""
FastAPI application entry point.

Rate limiting (via slowapi):
  - Anonymous: 10 requests/minute (keyed by IP)
  - API key holders: 60 requests/minute (keyed by API key header)

Startup creates all DB tables if they don't exist.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.routes import health, races, predictions, model_versions, admin
from src.api.middleware import UsageLoggingMiddleware
from src.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


def _rate_key(request: Request) -> str:
    """Use API key as the limit key when present, otherwise fall back to IP."""
    api_key = request.headers.get("x-api-key")
    if api_key:
        return f"key:{api_key}"
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_key)

app = FastAPI(
    title="Universal F1 Prediction API",
    description="Pre-race and post-qualifying podium predictions powered by CatBoost.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(UsageLoggingMiddleware)


# ── Routes ──────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(races.router)
app.include_router(predictions.router)
app.include_router(model_versions.router)
app.include_router(admin.router)
