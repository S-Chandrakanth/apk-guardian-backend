"""Engine and session handling for SQLite (all access goes through SQLAlchemy, with bound parameters)."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import Engine, event, create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.database.models import SCHEMA_VERSION, Base

logger = logging.getLogger("apkguardian.db")


def make_engine(settings: Settings) -> Engine:
    url = settings.resolved_database_url
    sqlite_file = settings.sqlite_path
    if sqlite_file is not None:
        sqlite_file.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False, "timeout": 10},
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")  # SQLite ignores foreign keys unless asked
        cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create all tables (safe to run repeatedly) and record the schema version."""
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(f"PRAGMA user_version = {int(SCHEMA_VERSION)}"))


def seed_default_users(engine: Engine) -> None:
    """Ensure baseline accounts exist so server restarts never lock users out."""
    from sqlalchemy.orm import Session
    from sqlalchemy import select
    from app.database.models import User
    from app.security.passwords import hash_password

    defaults = [
        ("Test User", "testuser@example.com", "StrongPass123!"),
        ("Security Analyst", "analyst@example.com", "Password123!"),
    ]
    with Session(engine) as session:
        for name, email, password in defaults:
            existing = session.scalar(select(User).where(User.email == email))
            if existing is None:
                user = User(name=name, email=email, password_hash=hash_password(password))
                session.add(user)
        session.commit()


def database_status(engine: Engine) -> str:
    """'ok', 'not_initialized', 'schema_mismatch', or 'error'. Never raises."""
    try:
        existing = set(inspect(engine).get_table_names())
        if not set(Base.metadata.tables) <= existing:
            return "not_initialized"
        with engine.connect() as conn:
            version = conn.execute(text("PRAGMA user_version")).scalar()
            conn.execute(text("SELECT 1"))
        return "ok" if version == SCHEMA_VERSION else "schema_mismatch"
    except SQLAlchemyError:
        logger.exception("database status check failed")
        return "error"
