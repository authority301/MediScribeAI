from app.database.base import Base
from app.database.session import engine
from app import models  # noqa: F401  (registers all models on Base.metadata)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database tables created.")
