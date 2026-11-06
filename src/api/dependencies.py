"""
FastAPI dependencies: DB session and optional API key validation.
"""

import hashlib
import os
from typing import Generator

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session

from src.db.session import SessionLocal
from src.db.models import ApiKey


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_api_key(x_api_key: str | None = Header(default=None), db: Session = Depends(get_db)) -> ApiKey | None:
    """
    Optional API key auth. Returns None for anonymous users.
    Raises 401 only if a key is provided but invalid/inactive.
    """
    if x_api_key is None:
        return None
    h = hashlib.sha256(x_api_key.encode()).hexdigest()
    key_obj = db.query(ApiKey).filter_by(key_hash=h, is_active=True).first()
    if not key_obj:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key.")
    return key_obj
