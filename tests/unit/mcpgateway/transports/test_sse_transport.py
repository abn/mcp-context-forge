# -*- coding: utf-8 -*-
"""

Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Mihai Criveti

Tests for the MCP Gateway SSE transport implementation.
"""

import asyncio
import json
from unittest.mock import Mock
from typing import Dict, Any

import pytest
from fastapi import Request
from sse_starlette.sse import EventSourceResponse

from mcpgateway.transports.sse_transport import SSETransport


@pytest.fixture
def sse_transport() -> SSETransport:
    """Create an SSE transport instance."""
    return SSETransport(base_url="http://test.example")


@pytest.fixture
def mock_request() -> Mock:
    """Create a mock FastAPI request."""
    mock = Mock(spec=Request)
    return mock


class TestSSETransport:
    """Tests for the SSETransport class."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, sse_transport: SSETransport) -> None:
        """Test connecting and disconnecting from SSE transport."""
        # Initially should not be connected
        assert await sse_transport.is_connected() is False

        # Connect
        await sse_transport.connect()
        assert await sse_transport.is_connected() is True
        assert sse_transport._connected is True

        # Disconnect
        await sse_transport.disconnect()
        assert await sse_transport.is_connected() is False
        assert sse_transport._connected is False
        assert sse_transport._client_gone.is_set()

    @pytest.mark.asyncio
    async def test_send_message(self, sse_transport: SSETransport) -> None:
        """Test sending a message over SSE."""
        # Connect first
        await sse_transport.connect()

        # Test message
        message: Dict[str, Any] = {"jsonrpc": "2.0", "method": "test", "id": 1}

        # Send message
        await sse_transport.send_message(message)

        # Verify message was queued
        assert sse_transport._message_queue.qsize() == 1
        queued_message = await sse_transport._message_queue.get()
        assert queued_message == message

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self, sse_transport: SSETransport) -> None:
        """Test sending message when not connected raises error."""
        # Don't connect
        message: Dict[str, Any] = {"jsonrpc": "2.0", "method": "test", "id": 1}

        # Should raise error
        with pytest.raises(RuntimeError, match="Transport not connected"):
            await sse_transport.send_message(message)

    @pytest.mark.asyncio
    async def test_create_sse_response(self, sse_transport: SSETransport, mock_request: Mock) -> None:
        """Test creating SSE response."""
        # Connect first
        await sse_transport.connect()

        # Create SSE response
        response = await sse_transport.create_sse_response(mock_request)

        # Should be an EventSourceResponse
        assert isinstance(response, EventSourceResponse)

        # Verify response headers
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-cache"
        assert response.headers["Content-Type"] == "text/event-stream"
        assert response.headers["X-MCP-SSE"] == "true"

    @pytest.mark.asyncio
    async def test_receive_message(self, sse_transport: SSETransport) -> None:
        """Test receiving messages from client."""
        # Connect first
        await sse_transport.connect()

        # Get receive generator
        receive_gen = sse_transport.receive_message()

        # Should yield initialize message first
        first_message = await receive_gen.__anext__()
        assert first_message["jsonrpc"] == "2.0"
        assert first_message["method"] == "initialize"

        # Trigger client disconnection to end the loop
        sse_transport._client_gone.set()

        # Wait for the generator to end
        with pytest.raises(StopAsyncIteration):
            # Use a timeout in case the generator doesn't end
            async def wait_with_timeout() -> None:
                await asyncio.wait_for(receive_gen.__anext__(), timeout=1.0)

            await wait_with_timeout()

    @pytest.mark.asyncio
    async def test_event_generator(self, sse_transport: SSETransport, mock_request: Mock) -> None:
        """Test the event generator for SSE."""
        # Connect first
        await sse_transport.connect()

        # Create SSE response
        response = await sse_transport.create_sse_response(mock_request)

        # Access the generator from the response
        generator = response.body_iterator

        # First event should be endpoint
        event = await generator.__anext__()
        assert "event" in event
        assert event["event"] == "endpoint"
        assert sse_transport._session_id in event["data"]

        # Second event should be keepalive
        event = await generator.__anext__()
        assert event["event"] == "keepalive"

        # Queue a test message
        test_message: Dict[str, Any] = {"jsonrpc": "2.0", "result": "test", "id": 1}
        await sse_transport._message_queue.put(test_message)

        # Next event should be the message
        event = await generator.__anext__()
        assert event["event"] == "message"
        assert json.loads(event["data"]) == test_message

        # Cancel the generator to clean up
        sse_transport._client_gone.set()
