from sqlalchemy import text

from app.database.base import Base
from app.database.session import engine
from app import models  # noqa: F401  (registers all models on Base.metadata)


def _ensure_doctor_auth_columns() -> None:
    """Additively migrate the pre-existing doctors table for authentication.

    create_all() only creates tables that don't exist yet; doctors already
    existed before this step, so its new auth columns are added here via
    idempotent ALTER TABLE statements instead of introducing Alembic.
    """
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE doctors ADD COLUMN IF NOT EXISTS password_hash TEXT NOT NULL DEFAULT ''"))
        conn.execute(text("ALTER TABLE doctors ALTER COLUMN password_hash DROP DEFAULT"))
        conn.execute(text("ALTER TABLE doctors ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT true"))


def _ensure_consultation_status_default() -> None:
    """Realign the consultations.status default with the approved lifecycle.

    The Step 5A design originally defaulted this column to 'scheduled'. Step 7
    defines the lifecycle as draft/active/completed/cancelled instead. status
    is a plain TEXT column (no CHECK constraint), so only its DEFAULT clause
    needs correcting here; the four allowed values are enforced in the API.
    """
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE consultations ALTER COLUMN status SET DEFAULT 'draft'"))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_doctor_auth_columns()
    _ensure_consultation_status_default()


if __name__ == "__main__":
    init_db()
    print("Database tables created.")
