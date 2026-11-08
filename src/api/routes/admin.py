"""
Admin API key management endpoints.

Requires a valid API key with tier="admin" via x-api-key header.

POST   /api/v1/admin/keys          — create a new key
GET    /api/v1/admin/keys          — list all keys
DELETE /api/v1/admin/keys/{key_id} — revoke a key
"""

import hashlib
import secrets

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, get_api_key
from src.db.models import ApiKey

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class KeyCreateRequest(BaseModel):
    client_name: str
    tier: str = "free"   # free | pro | admin


class KeyResponse(BaseModel):
    id: int
    client_name: str
    tier: str
    is_active: bool
    created_at: str
    # Only present on creation — never stored, never returned again
    raw_key: str | None = None

    model_config = {"from_attributes": True}


# ── Admin dependency ──────────────────────────────────────────────────────────

def require_admin(api_key: ApiKey | None = Depends(get_api_key)):
    if api_key is None or api_key.tier != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return api_key


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/keys", response_model=KeyResponse, status_code=201)
def create_key(
    body: KeyCreateRequest,
    db: Session = Depends(get_db),
    _: ApiKey = Depends(require_admin),
):
    raw = secrets.token_hex(32)   # 64-char hex key
    h = hashlib.sha256(raw.encode()).hexdigest()
    key = ApiKey(key_hash=h, client_name=body.client_name, tier=body.tier)
    db.add(key)
    db.commit()
    db.refresh(key)
    return KeyResponse(
        id=key.id,
        client_name=key.client_name,
        tier=key.tier,
        is_active=key.is_active,
        created_at=str(key.created_at),
        raw_key=raw,  # shown exactly once
    )


@router.get("/keys", response_model=list[KeyResponse])
def list_keys(
    db: Session = Depends(get_db),
    _: ApiKey = Depends(require_admin),
):
    keys = db.query(ApiKey).order_by(ApiKey.id).all()
    return [
        KeyResponse(
            id=k.id,
            client_name=k.client_name,
            tier=k.tier,
            is_active=k.is_active,
            created_at=str(k.created_at),
        )
        for k in keys
    ]


@router.delete("/keys/{key_id}", status_code=204)
def revoke_key(
    key_id: int,
    db: Session = Depends(get_db),
    _: ApiKey = Depends(require_admin),
):
    key = db.get(ApiKey, key_id)
    if not key:
        raise HTTPException(status_code=404, detail="Key not found.")
    key.is_active = False
    db.commit()
