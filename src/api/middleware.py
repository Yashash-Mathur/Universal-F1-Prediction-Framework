"""
Request logging middleware.

Writes one ApiUsage row per response, non-blocking (background task).
Skips the /health endpoint to avoid log noise.
"""

from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.db.session import SessionLocal
from src.db.models import ApiUsage, ApiKey
import hashlib


class UsageLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        # Skip health checks — no value in logging them
        if request.url.path == "/health":
            return response

        # Resolve api_key_id if header present
        api_key_id: int | None = None
        raw_key = request.headers.get("x-api-key")
        if raw_key:
            h = hashlib.sha256(raw_key.encode()).hexdigest()
            db = SessionLocal()
            try:
                key_obj = db.query(ApiKey).filter_by(key_hash=h, is_active=True).first()
                if key_obj:
                    api_key_id = key_obj.id
            finally:
                db.close()

        client_ip = request.client.host if request.client else "unknown"

        db = SessionLocal()
        try:
            db.add(ApiUsage(
                api_key_id=api_key_id,
                client_ip=client_ip,
                endpoint=request.url.path,
                status_code=response.status_code,
            ))
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

        return response
