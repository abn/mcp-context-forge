# -*- coding: utf-8 -*-
"""Session Registry with optional distributed state.

Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Mihai Criveti

This module provides a registry for SSE sessions with support for distributed deployment
using Redis or SQLAlchemy as optional backends for shared state between workers.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional, Callable, ContextManager
from contextlib import suppress # Added suppress

import httpx
from fastapi import HTTPException, status
from sqlalchemy.orm import Session # Changed from mcpgateway.db import Session

from mcpgateway.config import settings
from mcpgateway.db import SessionMessageRecord, SessionRecord, get_db
from mcpgateway.services import PromptService, ResourceService, ToolService # type: ignore
# TODO: Fix GatewayService import cycle or type hint issue
# from mcpgateway.services import GatewayService
from mcpgateway.transports import SSETransport
from mcpgateway.types import Implementation, InitializeResult, ServerCapabilities

logger = logging.getLogger(__name__)

tool_service = ToolService()
resource_service = ResourceService()
prompt_service = PromptService()
# gateway_service = GatewayService(tool_service, resource_service, prompt_service) # Causes circular import with services

try:
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = None # type: ignore

try:
    from sqlalchemy import func
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class SessionBackend:
    """Session backend related fields"""

    def __init__(
        self,
        backend: str = "memory",
        redis_url: Optional[str] = None,
        database_url: Optional[str] = None,
        session_ttl: int = 3600,  # 1 hour
        message_ttl: int = 600,  # 10 min
    ):
        self._backend = backend.lower()
        self._session_ttl = session_ttl
        self._message_ttl = message_ttl
        self._redis: Optional[Redis] = None
        self._pubsub: Optional[Any] = None # redis.asyncio.client.PubSub type if available
        self._session_message: Optional[Dict[str, str]] = None


        if self._backend == "memory":
            self._session_message = None
        elif self._backend == "none":
            logger.info("Session registry initialized with 'none' backend - session tracking disabled")
        elif self._backend == "redis":
            if not REDIS_AVAILABLE or not Redis:
                raise ValueError("Redis backend requested but redis package not installed/available")
            if not redis_url:
                raise ValueError("Redis backend requires redis_url")
            self._redis = Redis.from_url(redis_url)
            if self._redis: # Ensure client was created
                self._pubsub = self._redis.pubsub()
                # Subscription happens in initialize
        elif self._backend == "database":
            if not SQLALCHEMY_AVAILABLE:
                raise ValueError("Database backend requested but SQLAlchemy not installed")
            if not database_url: # Though settings.database_url is used by get_db
                raise ValueError("Database backend requires database_url")
        else:
            raise ValueError(f"Invalid backend: {backend}")


class SessionRegistry(SessionBackend):
    def __init__(
        self,
        db_provider: Callable[[], ContextManager[Session]] = get_db,
        backend: str = settings.cache_type, # Use settings
        redis_url: Optional[str] = settings.redis_url,
        database_url: Optional[str] = settings.database_url,
        session_ttl: int = settings.session_ttl,
        message_ttl: int = settings.message_ttl,
    ):
        super().__init__(backend=backend, redis_url=redis_url, database_url=database_url, session_ttl=session_ttl, message_ttl=message_ttl)
        self.db_provider = db_provider
        self._sessions: Dict[str, SSETransport] = {}  # Local transport cache, value is SSETransport
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._publish_task: Optional[asyncio.Task[None]] = None # For Redis publish loop if needed
        self._subscribe_task: Optional[asyncio.Task[None]] = None # For Redis subscribe loop
        self.local_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue() # For memory/db to simulate pubsub
        self._initialized = False
        self._instance_id = str(uuid.uuid4())
        self._leader_key = f"{settings.cache_prefix}leader"
        self._leader_ttl = 40
        self._lock_path = settings.filelock_path
        # from filelock import FileLock # Moved import to top-level if used elsewhere or keep local
        # self._file_lock = FileLock(self._lock_path) # Requires filelock to be installed

    async def initialize(self) -> None:
        logger.info(f"Initializing session registry with backend: {self._backend}")
        if self._backend == "redis" and self._pubsub:
            await self._pubsub.subscribe("mcp_session_events")
            self._subscribe_task = asyncio.create_task(self._redis_subscribe_loop())
            self._publish_task = asyncio.create_task(self._redis_publish_loop()) # If local_queue needs to publish to Redis
            logger.info("Redis pub/sub tasks started")
        elif self._backend == "database":
            self._cleanup_task = asyncio.create_task(self._db_cleanup_task())
            self._publish_task = asyncio.create_task(self._local_to_db_publish_loop())
            logger.info("Database cleanup and publish tasks started")
        elif self._backend == "memory":
            self._cleanup_task = asyncio.create_task(self._memory_cleanup_task())
            logger.info("Memory cleanup task started")
        self._initialized = True

    async def shutdown(self) -> None:
        logger.info("Shutting down session registry")
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with suppress(asyncio.CancelledError): await self._cleanup_task
        if self._subscribe_task:
            self._subscribe_task.cancel()
            with suppress(asyncio.CancelledError): await self._subscribe_task
        if self._publish_task:
            self._publish_task.cancel()
            with suppress(asyncio.CancelledError): await self._publish_task
        if self._backend == "redis" and self._pubsub:
            try:
                await self._pubsub.unsubscribe("mcp_session_events")
                await self._pubsub.close() # Changed from self._pubsub.close()
            except Exception as e: logger.error(f"Error closing Redis pubsub: {e}")
        if self._backend == "redis" and self._redis:
            try:
                await self._redis.close() # Changed from self._redis.close()
            except Exception as e: logger.error(f"Error closing Redis connection: {e}")
        # if hasattr(self, '_file_lock') and self._file_lock.is_locked: self._file_lock.release()


    async def add_session(self, session_id: str, transport: SSETransport) -> None:
        if self._backend == "none": return
        async with self._lock: self._sessions[session_id] = transport
        if self._backend == "redis" and self._redis:
            try:
                await self._redis.setex(f"mcp:session:{session_id}", self._session_ttl, "1")
                await self._redis.publish("mcp_session_events", json.dumps({"type": "add", "session_id": session_id, "timestamp": time.time()}))
            except Exception as e: logger.error(f"Redis error adding session {session_id}: {e}")
        elif self._backend == "database":
            try:
                def _db_add_sync():
                    with self.db_provider() as db:
                        session_record = SessionRecord(session_id=session_id, last_accessed=datetime.utcnow())
                        db.add(session_record)
                        db.commit()
                await asyncio.to_thread(_db_add_sync)
            except Exception as e: logger.error(f"Database error adding session {session_id}: {e}")
        logger.info(f"Added session: {session_id}")

    async def get_session(self, session_id: str) -> Optional[SSETransport]:
        if self._backend == "none": return None
        async with self._lock:
            transport = self._sessions.get(session_id)
            if transport: return transport
        # Logic for checking shared backend (Redis/DB) if not in local cache...
        return None # Simplified for now

    async def remove_session(self, session_id: str) -> None:
        if self._backend == "none": return
        transport: Optional[SSETransport] = None
        async with self._lock:
            if session_id in self._sessions: transport = self._sessions.pop(session_id)
        if transport:
            try: await transport.disconnect()
            except Exception as e: logger.error(f"Error disconnecting transport for session {session_id}: {e}")
        if self._backend == "redis" and self._redis:
            try:
                await self._redis.delete(f"mcp:session:{session_id}")
                await self._redis.publish("mcp_session_events", json.dumps({"type": "remove", "session_id": session_id, "timestamp": time.time()}))
            except Exception as e: logger.error(f"Redis error removing session {session_id}: {e}")
        elif self._backend == "database":
            try:
                def _db_remove_sync():
                    with self.db_provider() as db:
                        db.query(SessionRecord).filter(SessionRecord.session_id == session_id).delete()
                        db.commit()
                await asyncio.to_thread(_db_remove_sync)
            except Exception as e: logger.error(f"Database error removing session {session_id}: {e}")
        logger.info(f"Removed session: {session_id}")

    async def broadcast(self, session_id: str, message: Dict[str, Any]) -> None:
        if self._backend == "none": return
        msg_json = json.dumps(message)
        if self._backend == "memory":
            if self._session_message is not None: # Ensure it's not None
                 self._session_message = {"session_id": session_id, "message": msg_json}
        elif self._backend == "redis" and self._redis:
            try: await self._redis.publish(session_id, json.dumps({"type": "message", "message": msg_json, "timestamp": time.time()}))
            except Exception as e: logger.error(f"Redis error during broadcast: {e}")
        elif self._backend == "database":
            try:
                def _db_add_msg_sync():
                    with self.db_provider() as db:
                        msg_record = SessionMessageRecord(session_id=session_id, message=msg_json, last_accessed=datetime.utcnow())
                        db.add(msg_record)
                        db.commit()
                await asyncio.to_thread(_db_add_msg_sync)
            except Exception as e: logger.error(f"Database error during broadcast: {e}")

    def get_session_sync(self, session_id: str) -> Optional[SSETransport]:
        if self._backend == "none": return None
        return self._sessions.get(session_id)

    async def respond(self, server_id: Optional[str], user: Any, session_id: str, base_url: str) -> None:
        if self._backend == "none": return
        elif self._backend == "memory":
            transport = self.get_session_sync(session_id)
            if transport and self._session_message:
                message_str = self._session_message.get("message")
                if message_str:
                    try: message_dict = json.loads(message_str)
                    except json.JSONDecodeError: logger.error(f"Invalid JSON in session_message: {message_str}"); return
                    await self.generate_response(message=message_dict, transport=transport, server_id=server_id, user=user, base_url=base_url)
                else: logger.warning(f"No 'message' key in _session_message for session {session_id}")
            elif transport: logger.warning(f"_session_message is None for session {session_id} in memory backend")
        elif self._backend == "redis" and self._pubsub:
            # This direct subscribe in respond is problematic; should be a long-running task
            # For now, just log that this path would need a persistent listener
            logger.info(f"Redis backend: respond would require persistent listener for session {session_id}")
            pass
        elif self._backend == "database":
            # Similar to Redis, direct db polling in respond is not ideal.
            # The message_check_loop initiated by utility_sse_endpoint handles this.
            logger.info(f"Database backend: respond logic handled by message_check_loop for session {session_id}")
            pass


    async def generate_response(self, message: Dict[str, Any], transport: SSETransport, server_id: Optional[str], user: Any, base_url: str):
        result: Dict[str, Any] = {}
        if "method" in message and isinstance(message.get("method"), str) and "id" in message:
            method: str = message["method"]
            params_any = message.get("params", {})
            params: Dict[str, Any] = params_any if isinstance(params_any, dict) else {}
            req_id = message["id"]

            with self.db_provider() as db: # Use context manager for db session
                if method == "initialize":
                    init_params = params if isinstance(params, dict) else {}
                    init_result = await self.handle_initialize_logic(init_params)
                    response = {"jsonrpc": "2.0", "result": init_result.model_dump(by_alias=True, exclude_none=True), "id": req_id}
                    await transport.send_message(response)
                    await transport.send_message({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
                    notifications = ["tools/list_changed", "resources/list_changed", "prompts/list_changed"]
                    for notification in notifications:
                        await transport.send_message({"jsonrpc": "2.0", "method": f"notifications/{notification}", "params": {}})
                elif method == "tools/list":
                    tools = []
                    if server_id:
                        try: server_id_int = int(server_id); tools = await tool_service.list_server_tools(db, server_id=server_id_int)
                        except ValueError: logger.error(f"Invalid server_id for tools/list: {server_id}")
                    else: tools = await tool_service.list_tools(db)
                    result = {"tools": [t.model_dump(by_alias=True, exclude_none=True) for t in tools]}
                elif method == "resources/list":
                    resources = []
                    if server_id:
                        try: server_id_int = int(server_id); resources = await resource_service.list_server_resources(db, server_id=server_id_int)
                        except ValueError: logger.error(f"Invalid server_id for resources/list: {server_id}")
                    else: resources = await resource_service.list_resources(db)
                    result = {"resources": [r.model_dump(by_alias=True, exclude_none=True) for r in resources]}
                elif method == "prompts/list":
                    prompts = []
                    if server_id:
                        try: server_id_int = int(server_id); prompts = await prompt_service.list_server_prompts(db, server_id=server_id_int)
                        except ValueError: logger.error(f"Invalid server_id for prompts/list: {server_id}")
                    else: prompts = await prompt_service.list_prompts(db)
                    result = {"prompts": [p.model_dump(by_alias=True, exclude_none=True) for p in prompts]}
                elif method == "ping":
                    result = {}
                elif method == "tools/call":
                    tool_call_params = params
                    tool_name_any = tool_call_params.get("name")
                    tool_arguments_any = tool_call_params.get("arguments")
                    if not isinstance(tool_name_any, str): logger.error(f"Invalid tool name: {tool_call_params}"); return
                    tool_name: str = tool_name_any
                    tool_arguments: Dict[str, Any] = tool_arguments_any if isinstance(tool_arguments_any, dict) else {}
                    rpc_input = {"jsonrpc": "2.0", "method": tool_name, "params": tool_arguments, "id": str(uuid.uuid4())} # Ensure ID is unique

                    user_token = ""
                    if isinstance(user, dict): user_token = user.get("token", "")
                    headers = {"Authorization": f"Bearer {user_token}", "Content-Type": "application/json"}
                    rpc_url = base_url + "/rpc"
                    async with httpx.AsyncClient(timeout=settings.federation_timeout, verify=not settings.skip_ssl_verify) as client:
                        rpc_response = await client.post(url=rpc_url, json=rpc_input, headers=headers)
                        result = rpc_response.json() # This is the result for tools/call
                else: result = {} # Default empty result for unknown methods

                response = {"jsonrpc": "2.0", "result": result, "id": req_id}
                logger.info(f"Sending sse message: {response}")
                await transport.send_message(response)

    async def _redis_publish_loop(self) -> None:
        while True:
            try:
                message_to_publish = await self.local_queue.get()
                if self._redis and self._initialized: # Ensure redis client is available
                    await self._redis.publish("mcp_session_events", json.dumps(message_to_publish))
                self.local_queue.task_done()
            except asyncio.CancelledError: logger.info("Redis publish loop cancelled."); break
            except Exception as e: logger.error(f"Error in Redis publish loop: {e}")

    async def _redis_subscribe_loop(self) -> None:
        if not self._pubsub: return
        while True:
            try:
                message = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and isinstance(message.get("data"), (str, bytes)):
                    data_str = message["data"].decode() if isinstance(message["data"], bytes) else message["data"]
                    event = json.loads(data_str)
                    session_id = event.get("session_id")
                    # If this worker has the session, it might process it or ignore if it was the publisher
                    # This part needs more logic if we want Redis to distribute messages to specific session transports
                    logger.debug(f"Redis received event for {session_id}: {event.get('type')}")
            except asyncio.TimeoutError: continue # Normal timeout
            except asyncio.CancelledError: logger.info("Redis subscribe loop cancelled."); break
            except Exception as e: logger.error(f"Error in Redis subscribe loop: {e}"); await asyncio.sleep(5)


    async def _local_to_db_publish_loop(self) -> None:
        """Task to read from local_queue and write to DB messages (for 'database' backend)."""
        while True:
            try:
                item = await self.local_queue.get()
                session_id = item.get("session_id")
                message_content = item.get("message")
                if session_id and message_content:
                    def _db_add_msg_sync():
                        with self.db_provider() as db:
                            msg_record = SessionMessageRecord(session_id=session_id, message=json.dumps(message_content), last_accessed=datetime.utcnow())
                            db.add(msg_record)
                            db.commit()
                    await asyncio.to_thread(_db_add_msg_sync)
                self.local_queue.task_done()
            except asyncio.CancelledError: logger.info("Local_to_DB publish loop cancelled."); break
            except Exception as e: logger.error(f"Error in local_to_DB publish loop: {e}")


    async def _db_cleanup_task(self) -> None:
        logger.info("Starting database cleanup task")
        while True:
            try:
                def _db_cleanup_sync():
                    with self.db_provider() as db:
                        session_expiry = datetime.utcnow() - asyncio.to_timedelta(seconds=self._session_ttl)
                        deleted_sessions = db.query(SessionRecord).filter(SessionRecord.last_accessed < session_expiry).delete()
                        message_expiry = datetime.utcnow() - asyncio.to_timedelta(seconds=self._message_ttl)
                        deleted_messages = db.query(SessionMessageRecord).filter(SessionMessageRecord.last_accessed < message_expiry).delete()
                        db.commit()
                        return deleted_sessions, deleted_messages

                deleted_s, deleted_m = await asyncio.to_thread(_db_cleanup_sync)
                if deleted_s > 0: logger.info(f"Cleaned up {deleted_s} expired database sessions")
                if deleted_m > 0: logger.info(f"Cleaned up {deleted_m} expired database messages")

                # Check local sessions against database (simplified)
                # ... (this part might be complex to implement correctly without more context on session handoff)
                await asyncio.sleep(300)
            except asyncio.CancelledError: logger.info("Database cleanup task cancelled."); break
            except Exception as e: logger.error(f"Error in database cleanup task: {e}"); await asyncio.sleep(600)

    async def _memory_cleanup_task(self) -> None:
        logger.info("Starting memory cleanup task")
        while True:
            try:
                async with self._lock:
                    stale_sessions = [sid for sid, transport in self._sessions.items() if not await transport.is_connected()]
                for sid in stale_sessions: await self.remove_session(sid) # remove_session is async and handles lock
                if stale_sessions: logger.info(f"Cleaned up {len(stale_sessions)} disconnected memory sessions")
                await asyncio.sleep(60)
            except asyncio.CancelledError: logger.info("Memory cleanup task cancelled."); break
            except Exception as e: logger.error(f"Error in memory cleanup task: {e}"); await asyncio.sleep(300)

    async def handle_initialize_logic(self, body: Dict[str, Any]) -> InitializeResult:
        protocol_version = body.get("protocol_version") or body.get("protocolVersion")
        if not protocol_version:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing protocol version", headers={"MCP-Error-Code": "-32002"})
        if protocol_version != settings.protocol_version:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported protocol version: {protocol_version}", headers={"MCP-Error-Code": "-32003"})
        return InitializeResult(
            protocolVersion=settings.protocol_version,
            capabilities=ServerCapabilities(prompts={"listChanged": True}, resources={"subscribe": True, "listChanged": True}, tools={"listChanged": True}, logging={}, roots={"listChanged": True}, sampling={}),
            serverInfo=Implementation(name=settings.app_name, version="1.0.0"),
            instructions=("MCP Gateway providing federated tools, resources and prompts. Use /admin interface for configuration."),
        )

# Suppress FileLock import error if not used or available
FileLock = Any
