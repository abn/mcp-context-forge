# -*- coding: utf-8 -*-
"""SSE Transport Implementation.

Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Mihai Criveti

This module implements Server-Sent Events (SSE) transport for MCP,
providing server-to-client streaming with proper session management.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional # Ensured Optional is here

from fastapi import Request
from sse_starlette.sse import EventSourceResponse

from mcpgateway.config import settings
from mcpgateway.transports.base import Transport

logger = logging.getLogger(__name__)


class SSETransport(Transport):
    """Transport implementation using Server-Sent Events with proper session management."""

    def __init__(self, base_url: Optional[str] = None): # Changed str to Optional[str]
        """Initialize SSE transport.

        Args:
            base_url: Base URL for client message endpoints
        """
        self._base_url = base_url or f"http://{settings.host}:{settings.port}"
        self._connected = False
        self._message_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._client_gone = asyncio.Event()
        self._session_id = str(uuid.uuid4())

        logger.info(f"Creating SSE transport with base_url={self._base_url}, session_id={self._session_id}")

    async def connect(self) -> None:
        """Set up SSE connection."""
        self._connected = True
        logger.info(f"SSE transport connected: {self._session_id}")

    async def disconnect(self) -> None:
        """Clean up SSE connection."""
        if self._connected:
            self._connected = False
            self._client_gone.set()
            logger.info(f"SSE transport disconnected: {self._session_id}")

    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send a message over SSE.

        Args:
            message: Message to send

        Raises:
            RuntimeError: If transport is not connected
            Exception: If unable to put message to queue
        """
        if not self._connected:
            raise RuntimeError("Transport not connected")

        try:
            await self._message_queue.put(message)
            logger.debug(f"Message queued for SSE: {self._session_id}, method={message.get('method', '(response)')}")
        except Exception as e:
            logger.error(f"Failed to queue message: {e}")
            raise

    async def receive_message(self) -> AsyncGenerator[Dict[str, Any], None]: # type: ignore[override]
        """Receive messages from the client over SSE transport.

        This method implements a continuous message-receiving pattern for SSE transport.
        Since SSE is primarily a server-to-client communication channel, this method
        yields an initial initialize placeholder message and then enters a waiting loop.
        The actual client messages are received via a separate HTTP POST endpoint
        (not handled in this method).

        The method will continue running until either:
        1. The connection is explicitly disconnected (client_gone event is set)
        2. The receive loop is cancelled from outside

        Yields:
            Dict[str, Any]: JSON-RPC formatted messages. The first yielded message is always
                an initialize placeholder with the format:
                {"jsonrpc": "2.0", "method": "initialize", "id": 1}

        Raises:
            RuntimeError: If the transport is not connected when this method is called
            asyncio.CancelledError: When the SSE receive loop is cancelled externally
        """
        if not self._connected:
            raise RuntimeError("Transport not connected")

        yield {"jsonrpc": "2.0", "method": "initialize", "id": 1}

        try:
            while not self._client_gone.is_set():
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info(f"SSE receive loop cancelled for session {self._session_id}")
            raise
        finally:
            logger.info(f"SSE receive loop ended for session {self._session_id}")

    async def is_connected(self) -> bool:
        return self._connected

    async def create_sse_response(self, _request: Request) -> EventSourceResponse:
        endpoint_url = f"{self._base_url}/message?session_id={self._session_id}"

        async def event_generator() -> AsyncGenerator[Dict[str, Any], None]:
            yield {
                "event": "endpoint",
                "data": endpoint_url,
                "retry": settings.sse_retry_timeout,
            }
            yield {
                "event": "keepalive",
                "data": "{}",
                "retry": settings.sse_retry_timeout,
            }
            try:
                while not self._client_gone.is_set():
                    try:
                        message = await asyncio.wait_for(
                            self._message_queue.get(),
                            timeout=30.0,
                        )
                        data = json.dumps(message, default=lambda obj: (obj.strftime("%Y-%m-%d %H:%M:%S") if isinstance(obj, datetime) else TypeError("Type not serializable")))
                        logger.debug(f"Sending SSE message: {data}")
                        yield {
                            "event": "message",
                            "data": data,
                            "retry": settings.sse_retry_timeout,
                        }
                    except asyncio.TimeoutError:
                        yield {
                            "event": "keepalive",
                            "data": "{}",
                            "retry": settings.sse_retry_timeout,
                        }
                    except Exception as e:
                        logger.error(f"Error processing SSE message: {e}")
                        yield {
                            "event": "error",
                            "data": json.dumps({"error": str(e)}),
                            "retry": settings.sse_retry_timeout,
                        }
            except asyncio.CancelledError:
                logger.info(f"SSE event generator cancelled: {self._session_id}")
            except Exception as e:
                logger.error(f"SSE event generator error: {e}")
            finally:
                logger.info(f"SSE event generator completed: {self._session_id}")

        return EventSourceResponse(
            event_generator(),
            status_code=200,
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-MCP-SSE": "true",
            },
        )

    async def _client_disconnected(self, _request: Request) -> bool:
        return self._client_gone.is_set()

    @property
    def session_id(self) -> str:
        return self._session_id
