# -*- coding: utf-8 -*-
"""

Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Mihai Criveti

"""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mcpgateway.config import Settings
from mcpgateway.db import Base


from typing import Generator, Any
from fastapi import FastAPI
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, Any, None]:
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """Return the URL for the test database."""
    return "sqlite:///./test.db"


@pytest.fixture(scope="session")
def test_engine(test_db_url: str) -> Generator[Engine, Any, None]:
    """Create a SQLAlchemy engine for testing."""
    engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("./test.db"):
        os.remove("./test.db")


@pytest.fixture
def test_db(test_engine: Engine) -> Generator[Session, Any, None]:
    """Create a fresh database session for a test."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db: Session = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with in-memory database."""
    return Settings(
        database_url="sqlite:///:memory:",
        basic_auth_user="testuser",
        basic_auth_password="testpass",
        auth_required=False,
    )


@pytest.fixture
def app(test_settings: Settings) -> Generator[FastAPI, Any, None]:
    """Create a FastAPI test application."""
    with patch("mcpgateway.config.get_settings", return_value=test_settings):
        from mcpgateway.main import app as fastapi_app # Renamed to avoid conflict

        yield fastapi_app


@pytest.fixture
def mock_http_client() -> AsyncMock:
    """Create a mock HTTP client."""
    mock = AsyncMock()
    mock.aclose = AsyncMock()
    return mock


@pytest.fixture
def mock_websocket() -> AsyncMock:
    """Create a mock WebSocket."""
    mock = AsyncMock()
    mock.accept = AsyncMock()
    mock.send_json = AsyncMock()
    mock.receive_json = AsyncMock()
    mock.close = AsyncMock()
    return mock
