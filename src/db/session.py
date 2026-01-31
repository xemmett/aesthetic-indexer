from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def get_database_url() -> str:
    load_dotenv(override=False)
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Create a .env from .env.example and set DATABASE_URL."
        )
    return url


def get_engine(echo: bool = False) -> Engine:
    return create_engine(get_database_url(), echo=echo, pool_pre_ping=True, future=True)


def get_session_factory(echo: bool = False) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(echo=echo), expire_on_commit=False, future=True)


@contextmanager
def session_scope(echo: bool = False) -> Iterator[Session]:
    SessionLocal = get_session_factory(echo=echo)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


