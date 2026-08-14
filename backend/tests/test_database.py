"""Minimal database smoke tests: connection, table creation, and a query."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app import models
from app.database import Base, SessionLocal, engine
from app.database.init_db import init_db


def test_postgresql_connection():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.scalar() == 1


def test_tables_can_be_created():
    init_db()
    table_names = set(Base.metadata.tables.keys())
    assert "doctors" in table_names
    assert "soap_claims" in table_names
    assert "evidence_links" in table_names


def test_basic_database_query():
    db = SessionLocal()
    try:
        doctor_count = db.query(models.Doctor).count()
        assert doctor_count >= 0
    finally:
        db.close()
