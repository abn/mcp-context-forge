# -*- coding: utf-8 -*-
"""WebSocket Transport Implementation.

Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Mihai Criveti

This module implements WebSocket transport for MCP, handling
bidirectional JSON-RPC communication over a WebSocket connection.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect

from mcpgateway.config import settings
from mcpgateway.transports.base import Transport

logger = logging.getLogger(__name__)


class WebSocketTransport(Transport):
    """Transport implementation using WebSockets."""

    def __init__(self, websocket: WebSocket):
        """Initialize WebSocket transport.

        Args:
            websocket: FastAPI WebSocket connection object.
        """
        self.websocket = websocket
        self._connected = False
        self._ping_task: Optional[asyncio.Task[None]] = None

    async def connect(self) -> None:
        """Accept WebSocket connection and start ping loop."""
        await self.websocket.accept()
        self._connected = True
        self._ping_task = asyncio.create_task(self._ping_loop())
        logger.info(f"WebSocket connected: {self.websocket.client}")

    async def disconnect(self) -> None:
        """Close WebSocket connection and stop ping loop."""
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
            self._ping_task = None
        if self._connected:
            try:
                await self.websocket.close()
            except RuntimeError as e: # Handle cases where socket is already closed
                logger.warning(f"Error closing websocket (already closed?): {e}")
            self._connected = False
            logger.info(f"WebSocket disconnected: {self.websocket.client}")


    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send a JSON message over WebSocket.

        Args:
            message: Message to send.

        Raises:
            RuntimeError: If transport is not connected.
        """
        if not self._connected:
            raise RuntimeError("Transport not connected")
        await self.websocket.send_json(message)

    async def receive_message(self) -> AsyncGenerator[Dict[str, Any], None]: # type: ignore[override]
        """Receive JSON messages from WebSocket.

        Yields:
            Received messages.

        Raises:
            RuntimeError: If transport is not connected.
        """
        if not self._connected:
            raise RuntimeError("Transport not connected")
        try:
            while True:
                data = await self.websocket.receive_json()
                yield data
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected by client: {self.websocket.client}")
            await self.disconnect() # Ensure internal state is updated and ping loop stops
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
            await self.disconnect() # Disconnect on other errors too
            # Re-raise other exceptions to be handled by the caller if necessary
            # For example, if it's a JSONDecodeError, it might be an invalid message
            if not isinstance(e, json.JSONDecodeError): # Don't re-raise disconnects as they are handled
                 raise


    async def is_connected(self) -> bool:
        """Check if transport is connected.

        Returns:
            True if connected.
        """
        return self._connected

    async def send_ping(self) -> None:
        """Send a WebSocket ping frame."""
        if self._connected:
            try:
                await self.websocket.send_bytes(b"ping") # Standard WebSocket ping/pong uses opcodes, not custom bytes.
                                                       # For a custom ping, client needs to handle it.
                                                       # Consider using standard PING opcode if client/server supports.
                                                       # For now, sending bytes as a custom ping.
            except Exception as e:
                logger.warning(f"Failed to send ping: {e}")
                await self.disconnect()


    async def _ping_loop(self) -> None:
        """Periodically send pings to keep connection alive."""
        while self._connected:
            await asyncio.sleep(settings.websocket_ping_interval)
            if self._connected: # Re-check after sleep
                await self.send_ping()
                # Optionally, wait for a pong here if implementing a custom pong response
                # try:
                #     pong_waiter = await asyncio.wait_for(self.websocket.receive_bytes(), timeout=5)
                #     if pong_waiter != b"pong": # Example custom pong
                #         logger.warning("Pong not received or incorrect.")
                #         await self.disconnect()
                #         break
                # except asyncio.TimeoutError:
                #     logger.warning("Pong receive timeout.")
                #     await self.disconnect()
                #     break
                # except WebSocketDisconnect: # Already handled by receive_message
                #     break
                # except Exception as e:
                #     logger.error(f"Error in pong handling: {e}")
                #     await self.disconnect()
                #     break
            else:
                break
        logger.debug("Ping loop ended.")
