"""
Database session factory.

DATABASE_URL priority:
  1. Environment variable DATABASE_URL
  2. .env file
  3. Fallback: SQLite at data/f1_platform.db (zero-config local dev)
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/f1_platform.db")

# Auto-create directory for SQLite file if it doesn't exist
if DATABASE_URL.startswith("sqlite:///"):
    _db_path = DATABASE_URL.replace("sqlite:///", "")
    if _db_path and _db_path != ":memory:":
        _dir_name = os.path.dirname(os.path.abspath(_db_path))
        if _dir_name:
            os.makedirs(_dir_name, exist_ok=True)

# SQLite needs check_same_thread=False for FastAPI's thread model
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables (no-op if they exist)."""
    from src.db.models import Base  # noqa: F401 — imports trigger metadata registration
    Base.metadata.create_all(bind=engine)
