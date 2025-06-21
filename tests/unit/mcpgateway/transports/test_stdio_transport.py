# -*- coding: utf-8 -*-
"""Tests for the stdio transport implementation.

Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Mihai Criveti

This module tests the stdio transport for MCP, ensuring it properly handles
communication over standard input/output streams.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List # Added List

import pytest

from mcpgateway.transports.stdio_transport import StdioTransport


@pytest.fixture
def stdio_transport() -> StdioTransport:
    """Create a StdioTransport instance for testing."""
    return StdioTransport()


class TestStdioTransport:
    """Test suite for the StdioTransport class."""

    @patch("asyncio.get_running_loop")
    async def test_connect(self, mock_get_loop: MagicMock, stdio_transport: StdioTransport) -> None:
        """Test establishing a connection."""
        # Set up mocks
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        MagicMock()
        mock_reader_protocol = MagicMock()

        mock_transport = MagicMock()
        mock_protocol = MagicMock()

        # Mock the connect_read_pipe and connect_write_pipe methods
        mock_loop.connect_read_pipe = AsyncMock(return_value=(mock_transport, mock_reader_protocol))
        mock_loop.connect_write_pipe = AsyncMock(return_value=(mock_transport, mock_protocol))

        # Call the method under test
        await stdio_transport.connect()

        # Verify the expected calls
        mock_get_loop.assert_called_once()
        mock_loop.connect_read_pipe.assert_called_once()
        mock_loop.connect_write_pipe.assert_called_once()

        # Verify the connection state
        assert stdio_transport._connected is True

    @patch("asyncio.StreamWriter")
    async def test_disconnect(self, mock_writer: MagicMock, stdio_transport: StdioTransport) -> None:
        """Test closing a connection."""
        # Set up mock
        stdio_transport._stdout_writer = mock_writer
        stdio_transport._connected = True

        # Call the method under test
        await stdio_transport.disconnect()

        # Verify the expected calls
        mock_writer.close.assert_called_once()
        mock_writer.wait_closed.assert_called_once()

        # Verify the connection state
        assert stdio_transport._connected is False

    async def test_disconnect_not_connected(self, stdio_transport: StdioTransport) -> None:
        """Test disconnecting when not connected."""
        # Ensure not connected
        stdio_transport._stdout_writer = None
        stdio_transport._connected = False

        # Call the method under test
        await stdio_transport.disconnect()

        # Verify the connection state
        assert stdio_transport._connected is False

    async def test_send_message_not_connected(self, stdio_transport: StdioTransport) -> None:
        """Test sending a message when not connected."""
        # Ensure not connected
        stdio_transport._stdout_writer = None
        stdio_transport._connected = False

        # Verify the expected exception
        with pytest.raises(RuntimeError, match="Transport not connected"):
            await stdio_transport.send_message({"type": "test"})

    @patch("asyncio.StreamWriter")
    async def test_send_message(self, mock_writer: MagicMock, stdio_transport: StdioTransport) -> None:
        """Test sending a message."""
        # Set up mock
        stdio_transport._stdout_writer = mock_writer
        stdio_transport._connected = True

        # Call the method under test
        await stdio_transport.send_message({"type": "test", "data": "message"})

        # Verify the expected calls
        mock_writer.write.assert_called_once()
        mock_writer.drain.assert_called_once()

        # Verify the encoded message
        call_args = mock_writer.write.call_args[0][0]
        assert b'{"type": "test", "data": "message"}\n' == call_args

    @patch("asyncio.StreamWriter")
    async def test_send_message_exception(self, mock_writer: MagicMock, stdio_transport: StdioTransport) -> None:
        """Test sending a message when an error occurs."""
        # Set up mock with an exception
        stdio_transport._stdout_writer = mock_writer
        stdio_transport._connected = True
        mock_writer.write.side_effect = Exception("Write error")

        # Verify the expected exception
        with pytest.raises(Exception, match="Write error"):
            await stdio_transport.send_message({"type": "test"})

    async def test_receive_message_not_connected(self, stdio_transport: StdioTransport) -> None:
        """Test receiving messages when not connected."""
        # Ensure not connected
        stdio_transport._stdin_reader = None
        stdio_transport._connected = False

        # Verify the expected exception
        with pytest.raises(RuntimeError, match="Transport not connected"):
            async for _ in stdio_transport.receive_message():
                pass

    @patch("asyncio.StreamReader")
    async def test_receive_message(self, mock_reader: MagicMock, stdio_transport: StdioTransport) -> None:
        """Test receiving messages."""
        # Set up mock with two messages followed by EOF
        stdio_transport._stdin_reader = mock_reader
        stdio_transport._connected = True

        message1 = b'{"type": "message1", "data": "test1"}\n'
        message2 = b'{"type": "message2", "data": "test2"}\n'

        # Configure the mock to return messages and then an empty response (EOF)
        mock_reader.readline = AsyncMock(side_effect=[message1, message2, b""])

        # Collect received messages
        received: List[Dict[str, Any]] = []
        async for message in stdio_transport.receive_message():
            received.append(message)

        # Verify the expected calls and received messages
        assert mock_reader.readline.call_count == 3
        assert len(received) == 2
        assert received[0] == {"type": "message1", "data": "test1"}
        assert received[1] == {"type": "message2", "data": "test2"}

    @patch("asyncio.StreamReader")
    async def test_receive_message_exception(self, mock_reader: MagicMock, stdio_transport: StdioTransport) -> None:
        """Test receiving messages when a non-fatal error occurs."""
        # Set up mock with a valid message, then an invalid one, then EOF
        stdio_transport._stdin_reader = mock_reader
        stdio_transport._connected = True

        message1 = b'{"type": "message1", "data": "test1"}\n'
        invalid_message = b"not valid json\n"

        # Configure the mock to return a valid message, then an invalid one, then EOF
        mock_reader.readline = AsyncMock(side_effect=[message1, invalid_message, b""])

        # Collect received messages (should only get the valid one)
        received: List[Dict[str, Any]] = []
        async for message in stdio_transport.receive_message():
            received.append(message)

        # Verify the expected calls and received messages
        assert mock_reader.readline.call_count == 3
        assert len(received) == 1
        assert received[0] == {"type": "message1", "data": "test1"}

    @patch("asyncio.StreamReader")
    async def test_receive_message_cancellation(self, mock_reader: MagicMock, stdio_transport: StdioTransport) -> None:
        """Test receiving messages with cancellation."""
        # Set up mock with a message
        stdio_transport._stdin_reader = mock_reader
        stdio_transport._connected = True

        message_bytes = b'{"type": "message", "data": "test"}\n'

        # Configure the mock to return a message and then raise a cancellation
        mock_reader.readline = AsyncMock(side_effect=[message_bytes, asyncio.CancelledError()])

        # Collect received messages until cancellation
        received: List[Dict[str, Any]] = []
        try:
            async for message_dict in stdio_transport.receive_message():
                received.append(message_dict)
        except asyncio.CancelledError:
            pass  # Expected

        # Verify the expected calls and received messages
        assert mock_reader.readline.call_count == 2
        assert len(received) == 1
        assert received[0] == {"type": "message", "data": "test"}

    async def test_is_connected(self, stdio_transport: StdioTransport) -> None:
        """Test checking connection status."""
        # Test when not connected
        stdio_transport._connected = False
        assert await stdio_transport.is_connected() is False

        # Test when connected
        stdio_transport._connected = True
        assert await stdio_transport.is_connected() is True

    @patch.object(asyncio, "StreamReader")
    @patch.object(asyncio, "get_running_loop")
    async def test_full_lifecycle(
        self, mock_get_loop: MagicMock, mock_stream_reader_cls: MagicMock, stdio_transport: StdioTransport
    ) -> None:
        """Test a full lifecycle of connect, send/receive, and disconnect."""
        # Set up mocks
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        mock_reader_instance = MagicMock(spec=asyncio.StreamReader) # Instance for _stdin_reader
        mock_reader_protocol = MagicMock() # For connect_read_pipe return

        mock_transport = MagicMock() # For connect_write_pipe return
        mock_writer_protocol = MagicMock() # For connect_write_pipe return (renamed for clarity)
        mock_writer_instance = MagicMock(spec=asyncio.StreamWriter) # Instance for _stdout_writer

        # Mock the connect_read_pipe and connect_write_pipe methods
        mock_loop.connect_read_pipe = AsyncMock(return_value=(mock_reader_protocol, mock_reader_protocol)) # protocol is ReaderProtocol
        mock_loop.connect_write_pipe = AsyncMock(return_value=(mock_transport, mock_writer_protocol)) # protocol is Protocol

        # Assign instances to the transport
        stdio_transport._stdout_writer = mock_writer_instance
        stdio_transport._stdin_reader = mock_reader_instance


        # Configure reader with a test message
        mock_message_bytes = b'{"type": "test", "content": "hello"}\n'
        mock_reader_instance.readline = AsyncMock(side_effect=[mock_message_bytes, b""]) # Add EOF

        # Test connect
        await stdio_transport.connect()
        assert stdio_transport._connected is True

        # Test is_connected
        assert await stdio_transport.is_connected() is True

        # Test send_message
        await stdio_transport.send_message({"type": "response", "content": "world"})
        mock_writer_instance.write.assert_called_once()
        mock_writer_instance.drain.assert_called_once()

        # Test receive_message
        async def get_first_message() -> Dict[str, Any]:
            async for message_dict in stdio_transport.receive_message():
                return message_dict
            raise AssertionError("No message received") # Should not happen

        message_data = await get_first_message()
        assert message_data == {"type": "test", "content": "hello"}

        # Test disconnect
        await stdio_transport.disconnect()
        assert stdio_transport._connected is False
        mock_writer_instance.close.assert_called_once()
        mock_writer_instance.wait_closed.assert_called_once()
