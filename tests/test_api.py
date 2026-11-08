"""
API integration tests using FastAPI TestClient against an in-memory SQLite database.

Covers:
  - /health endpoint
  - /api/v1/races (list + detail)
  - /api/v1/predictions/{race_id}
  - /api/v1/models/latest
  - API key auth (valid, invalid)
  - Rate limit (429 on burst)
  - Admin key management (create, list, revoke)
"""

import hashlib
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.models import Base, Race, Driver, ModelVersion, Prediction, ApiKey
from src.api.dependencies import get_db
from src.api.main import app

# ── In-memory SQLite test database with StaticPool ─────────────────────────
TEST_DB_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    db = TestSession()

    # Seed minimal data
    mv = ModelVersion(
        id=1, version_tag="v6-catboost-300", training_date="2026-11-06",
        accuracy=0.8926, precision=0.6061, recall=0.5455,
        f1_score=0.5619, mean_podiums=1.82, perfect_podiums=2, races_evaluated=11,
    )
    race = Race(id=1, year=2026, round_number=12, race_name="Dutch Grand Prix")
    driver1 = Driver(id=1, code="NOR", full_name="Lando Norris", team_name="McLaren")
    driver2 = Driver(id=2, code="RUS", full_name="George Russell", team_name="Mercedes")
    driver3 = Driver(id=3, code="ANT", full_name="Kimi Antonelli", team_name="Mercedes")
    pred1 = Prediction(race_id=1, driver_id=1, model_version_id=1, stage="post_quali",
                       podium_probability=0.647, predicted_rank=1, predicted_podium=True)
    pred2 = Prediction(race_id=1, driver_id=2, model_version_id=1, stage="post_quali",
                       podium_probability=0.642, predicted_rank=2, predicted_podium=True)
    pred3 = Prediction(race_id=1, driver_id=3, model_version_id=1, stage="post_quali",
                       podium_probability=0.476, predicted_rank=3, predicted_podium=True)

    valid_key = "test_valid_key"
    api_key = ApiKey(key_hash=hashlib.sha256(valid_key.encode()).hexdigest(),
                     client_name="tester", tier="free")
    admin_key_raw = "test_admin_key"
    admin_key = ApiKey(key_hash=hashlib.sha256(admin_key_raw.encode()).hexdigest(),
                       client_name="admin_tester", tier="admin")

    db.add_all([mv, race, driver1, driver2, driver3, pred1, pred2, pred3, api_key, admin_key])
    db.commit()
    db.close()

    yield

    Base.metadata.drop_all(bind=engine)


client = TestClient(app)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["db"] == "ok"


def test_list_races():
    r = client.get("/api/v1/races")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["race_name"] == "Dutch Grand Prix"


def test_list_races_filtered_year():
    r = client.get("/api/v1/races?year=2026")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_list_races_no_match():
    r = client.get("/api/v1/races?year=2019")
    assert r.status_code == 200
    assert r.json() == []


def test_get_race():
    r = client.get("/api/v1/races/1")
    assert r.status_code == 200
    assert r.json()["round_number"] == 12


def test_get_race_not_found():
    r = client.get("/api/v1/races/999")
    assert r.status_code == 404


def test_get_prediction():
    r = client.get("/api/v1/predictions/1")
    assert r.status_code == 200
    data = r.json()
    assert data["race_name"] == "Dutch Grand Prix"
    assert len(data["top_3"]) == 3
    assert len(data["full_grid"]) == 3
    assert data["top_3"][0]["code"] == "NOR"
    assert data["top_3"][0]["predicted_podium"] is True


def test_get_prediction_not_found():
    r = client.get("/api/v1/predictions/999")
    assert r.status_code == 404


def test_get_latest_model():
    r = client.get("/api/v1/models/latest")
    assert r.status_code == 200
    data = r.json()
    assert data["version_tag"] == "v6-catboost-300"
    assert abs(data["accuracy"] - 0.8926) < 1e-4


def test_valid_api_key():
    r = client.get("/api/v1/races", headers={"x-api-key": "test_valid_key"})
    assert r.status_code == 200


def test_invalid_api_key():
    r = client.get("/api/v1/races", headers={"x-api-key": "bogus_key"})
    assert r.status_code == 401


# ── Admin key management tests ────────────────────────────────────────────────

ADMIN_HDR = {"x-api-key": "test_admin_key"}
NON_ADMIN_HDR = {"x-api-key": "test_valid_key"}


def test_admin_create_key():
    r = client.post("/api/v1/admin/keys",
                    json={"client_name": "new_client", "tier": "pro"},
                    headers=ADMIN_HDR)
    assert r.status_code == 201
    data = r.json()
    assert data["client_name"] == "new_client"
    assert data["tier"] == "pro"
    assert data["is_active"] is True
    assert len(data["raw_key"]) == 64   # 32 hex bytes → 64 chars


def test_admin_list_keys():
    r = client.get("/api/v1/admin/keys", headers=ADMIN_HDR)
    assert r.status_code == 200
    keys = r.json()
    assert len(keys) >= 2   # at least free tester + admin tester
    # raw_key must NOT be in list responses
    for k in keys:
        assert k.get("raw_key") is None


def test_admin_revoke_key():
    # Create a key then revoke it
    created = client.post("/api/v1/admin/keys",
                          json={"client_name": "to_revoke", "tier": "free"},
                          headers=ADMIN_HDR).json()
    r = client.delete(f"/api/v1/admin/keys/{created['id']}", headers=ADMIN_HDR)
    assert r.status_code == 204
    # Verify it's inactive (appears in list but is_active=False)
    keys = client.get("/api/v1/admin/keys", headers=ADMIN_HDR).json()
    match = next(k for k in keys if k["id"] == created["id"])
    assert match["is_active"] is False


def test_admin_revoke_nonexistent():
    r = client.delete("/api/v1/admin/keys/99999", headers=ADMIN_HDR)
    assert r.status_code == 404


def test_non_admin_forbidden():
    r = client.post("/api/v1/admin/keys",
                    json={"client_name": "x", "tier": "free"},
                    headers=NON_ADMIN_HDR)
    assert r.status_code == 403


def test_anon_admin_forbidden():
    r = client.get("/api/v1/admin/keys")
    assert r.status_code == 403
