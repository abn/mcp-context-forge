# -*- coding: utf-8 -*-
"""Gateway Service Implementation.

Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Mihai Criveti

This module implements gateway federation according to the MCP specification.
It handles:
- Gateway discovery and registration
- Request forwarding
- Capability aggregation
- Health monitoring
- Active/inactive gateway management
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple
from contextlib import suppress # Added suppress

import httpx
from filelock import FileLock, Timeout # type: ignore
from mcp import ClientSession # type: ignore
from mcp.client.sse import sse_client # type: ignore
from mcp.client.streamable_http import streamablehttp_client # type: ignore
from sqlalchemy import select, delete # Added delete
from sqlalchemy.orm import Session

from mcpgateway.config import settings
from mcpgateway.db import Gateway as DbGateway, SessionLocal
from mcpgateway.db import Tool as DbTool
from mcpgateway.db import Resource as DbResource
from mcpgateway.db import Prompt as DbPrompt
from mcpgateway.schemas import GatewayCreate, GatewayRead, GatewayUpdate, ToolCreate
from mcpgateway.types import ServerCapabilities
from mcpgateway.services.tool_service import ToolService
from mcpgateway.utils.services_auth import decode_auth

try:
    import redis # type: ignore

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None # type: ignore
    logging.info("Redis is not utilized in this environment.")

logger = logging.getLogger(__name__)


GW_FAILURE_THRESHOLD = settings.unhealthy_threshold
GW_HEALTH_CHECK_INTERVAL = settings.health_check_interval


class GatewayError(Exception):
    """Base class for gateway-related errors."""


class GatewayNotFoundError(GatewayError):
    """Raised when a requested gateway is not found."""


class GatewayNameConflictError(GatewayError):
    """Raised when a gateway name conflicts with existing (active or inactive) gateway."""

    def __init__(self, name: str, is_active: bool = True, gateway_id: Optional[int] = None):
        self.name = name
        self.is_active = is_active
        self.gateway_id = gateway_id
        message = f"Gateway already exists with name: {name}"
        if not is_active:
            message += f" (currently inactive, ID: {gateway_id})"
        super().__init__(message)


class GatewayConnectionError(GatewayError):
    """Raised when gateway connection fails."""


class GatewayService:
    def __init__(self):
        self._event_subscribers: List[asyncio.Queue[Dict[str, Any]]] = []
        self._http_client = httpx.AsyncClient(timeout=settings.federation_timeout, verify=not settings.skip_ssl_verify)
        self._health_check_interval = GW_HEALTH_CHECK_INTERVAL
        self._health_check_task: Optional[asyncio.Task[None]] = None
        self._active_gateways: Set[str] = set()
        self.tool_service = ToolService()
        self._gateway_failure_counts: Dict[int, int] = {} # Key is gateway.id (int)

        self.redis_url = settings.redis_url if settings.cache_type == "redis" else None
        self._redis_client: Optional[Any] = None # redis.Redis type
        self._instance_id: Optional[str] = None
        self._leader_key: Optional[str] = None
        self._leader_ttl: int = 40

        if self.redis_url and REDIS_AVAILABLE and redis:
            self._redis_client = redis.from_url(self.redis_url)
            self._instance_id = str(uuid.uuid4())
            self._leader_key = f"{settings.cache_prefix}gateway_service_leader"
        elif settings.cache_type != "none" and hasattr(settings, 'filelock_path'): # Check if filelock_path exists
            self._lock_path = settings.filelock_path
            self._file_lock = FileLock(self._lock_path)

    async def initialize(self) -> None:
        logger.info("Initializing gateway service")
        if self._redis_client and self._leader_key and self._instance_id:
            try:
                if hasattr(self._redis_client, "ping"): # Ensure client has ping method
                    pong = await self._redis_client.ping()
                    if not pong: raise ConnectionError("Redis ping failed.")
                is_leader = await self._redis_client.set(self._leader_key, self._instance_id, ex=self._leader_ttl, nx=True)
                if is_leader:
                    logger.info("Acquired Redis leadership. Starting health check task.")
                    self._health_check_task = asyncio.create_task(self._run_health_checks())
            except Exception as e:
                logger.error(f"Redis initialization failed: {e}. Health checks may not be leader-elected.")
                self._health_check_task = asyncio.create_task(self._run_health_checks(assume_leader_for_non_redis=True))
        else:
            self._health_check_task = asyncio.create_task(self._run_health_checks(assume_leader_for_non_redis=True))

    async def shutdown(self) -> None:
        if self._health_check_task:
            self._health_check_task.cancel()
            with suppress(asyncio.CancelledError): await self._health_check_task
        await self._http_client.aclose()
        self._event_subscribers.clear()
        self._active_gateways.clear()
        if self._redis_client and hasattr(self._redis_client, "close"): await self._redis_client.close()
        if hasattr(self, '_file_lock') and self._file_lock.is_locked: self._file_lock.release()
        logger.info("Gateway service shutdown complete")

    async def _initialize_gateway(self, url: str, encoded_auth_value: Optional[str] = None, transport: str = "SSE") -> Tuple[Dict[str, Any], List[ToolCreate]]:
        try:
            auth_headers: Optional[Dict[str, str]] = None
            if encoded_auth_value:
                decoded = decode_auth(encoded_auth_value)
                if isinstance(decoded, dict): auth_headers = {str(k): str(v) for k, v in decoded.items()}
                else: logger.warning(f"Failed to decode auth_value for gateway {url}")

            headers_for_client = auth_headers or {}

            async def connect_and_init_session(client_manager, server_url: str, headers: Dict[str, str]) -> Tuple[Dict[str, Any], List[ToolCreate]]:
                async with client_manager(url=server_url, headers=headers) as session_params:
                    if len(session_params) == 3:
                        read_stream, write_stream, _ = session_params
                        client_session = ClientSession(read_stream, write_stream)
                    else:
                        client_session = ClientSession(*session_params)

                    async with client_session as session:
                        init_response = await session.initialize()
                        capabilities = init_response.capabilities.model_dump(by_alias=True, exclude_none=True)
                        tools_response = await session.list_tools()
                        tools_data = [tool.model_dump(by_alias=True, exclude_none=True) for tool in tools_response.tools]
                        validated_tools = [ToolCreate.model_validate(td) for td in tools_data]
                        if transport.lower() == "streamablehttp":
                            for tool_obj in validated_tools:
                                if hasattr(tool_obj, 'request_type'): tool_obj.request_type = "STREAMABLEHTTP"
                        return capabilities, validated_tools

            if transport.lower() == "sse":
                return await connect_and_init_session(sse_client, url, headers_for_client)
            elif transport.lower() == "streamablehttp":
                return await connect_and_init_session(streamablehttp_client, url, headers_for_client)
            else:
                raise GatewayConnectionError(f"Unsupported transport type '{transport}' specified for gateway {url}")
        except Exception as e:
            raise GatewayConnectionError(f"Failed to initialize gateway at {url}: {str(e)}")

    async def register_gateway(self, db: Session, gateway_create: GatewayCreate) -> GatewayRead:
        try:
            existing_gateway = db.execute(select(DbGateway).where(DbGateway.name == gateway_create.name)).scalar_one_or_none()
            if existing_gateway:
                raise GatewayNameConflictError(gateway_create.name, is_active=existing_gateway.is_active, gateway_id=existing_gateway.id)

            auth_type = gateway_create.auth_type
            encoded_auth_val = gateway_create.auth_value

            capabilities_dict, tools_from_remote = await self._initialize_gateway(str(gateway_create.url), encoded_auth_val, gateway_create.transport)

            db_tools_for_gateway: List[DbTool] = []
            if tools_from_remote: # Check if tools_from_remote is not None
                for tool_data in tools_from_remote:
                    if not isinstance(tool_data, ToolCreate): # Ensure item is ToolCreate
                        logger.warning(f"Skipping invalid tool data item: {tool_data}")
                        continue

                    # Simplified: always create new DbTool linked to this new gateway
                    # A more complex sync would update existing tools if they were previously federated by another URL/gateway instance.
                    db_tool = DbTool(
                        name=tool_data.name,
                        url=str(tool_data.url) if tool_data.url else str(gateway_create.url),
                        description=tool_data.description,
                        integration_type=tool_data.integration_type or "MCP",
                        request_type=tool_data.request_type or "SSE",
                        headers=tool_data.headers,
                        input_schema=tool_data.input_schema or {"type": "object", "properties": {}},
                        jsonpath_filter=tool_data.jsonpath_filter,
                        auth_type=None,
                        auth_value=None,
                        is_active=True,
                    )
                    db_tools_for_gateway.append(db_tool)

            db_gateway = DbGateway(
                name=gateway_create.name, url=str(gateway_create.url), description=gateway_create.description,
                transport=gateway_create.transport, capabilities=capabilities_dict,
                last_seen=datetime.now(timezone.utc), auth_type=auth_type, auth_value=encoded_auth_val,
                tools=db_tools_for_gateway
            )
            db.add(db_gateway)
            db.commit()
            db.refresh(db_gateway)
            self._active_gateways.add(db_gateway.url)
            await self._notify_gateway_added(db_gateway)
            return GatewayRead.model_validate(db_gateway)
        except (GatewayConnectionError, ValueError, RuntimeError, GatewayNameConflictError) as e:
            db.rollback()
            logger.error(f"Error registering gateway {gateway_create.name}: {e}")
            raise
        except Exception as e:
            db.rollback()
            logger.error(f"Unexpected error registering gateway {gateway_create.name}: {e}", exc_info=True)
            raise GatewayError(f"Failed to register gateway due to an unexpected error: {str(e)}")

    async def list_gateways(self, db: Session, include_inactive: bool = False) -> List[GatewayRead]:
        query = select(DbGateway)
        if not include_inactive: query = query.where(DbGateway.is_active)
        gateways = db.execute(query).scalars().all()
        return [GatewayRead.model_validate(g) for g in gateways]

    async def update_gateway(self, db: Session, gateway_id: int, gateway_update: GatewayUpdate) -> GatewayRead:
        gateway = db.get(DbGateway, gateway_id)
        if not gateway: raise GatewayNotFoundError(f"Gateway not found: {gateway_id}")
        if not gateway.is_active: raise GatewayNotFoundError(f"Gateway '{gateway.name}' exists but is inactive")

        if gateway_update.name and gateway_update.name != gateway.name:
            conflict = db.execute(select(DbGateway).where(DbGateway.name == gateway_update.name, DbGateway.id != gateway_id)).scalar_one_or_none()
            if conflict: raise GatewayNameConflictError(gateway_update.name, is_active=conflict.is_active, gateway_id=conflict.id)

        update_data = gateway_update.model_dump(exclude_unset=True)

        url_changed = "url" in update_data and update_data["url"] != gateway.url
        transport_changed = "transport" in update_data and update_data["transport"] != gateway.transport
        auth_changed = False

        if "auth_type" in update_data or "auth_value" in update_data :
            new_auth_type = update_data.get("auth_type", gateway.auth_type)
            new_auth_value = update_data.get("auth_value", gateway.auth_value)
            if new_auth_type != gateway.auth_type or new_auth_value != gateway.auth_value:
                auth_changed = True
            gateway.auth_type = new_auth_type
            gateway.auth_value = new_auth_value

        update_data.pop("auth_type", None); update_data.pop("auth_value", None)
        update_data.pop("auth_username", None); update_data.pop("auth_password", None)
        update_data.pop("auth_token", None); update_data.pop("auth_header_key", None)
        update_data.pop("auth_header_value", None)

        for key, value in update_data.items():
            setattr(gateway, key, value)

        if url_changed or transport_changed or auth_changed:
            try:
                capabilities, tools_from_remote = await self._initialize_gateway(gateway.url, gateway.auth_value, gateway.transport)
                gateway.capabilities = capabilities
                gateway.last_seen = datetime.now(timezone.utc)
                # TODO: Sophisticated tool sync on update (add new, remove old, update existing based on remote)
                # For now, just log the tools fetched, no DB changes for tools here.
                logger.info(f"Gateway {gateway.name} re-initialized, fetched {len(tools_from_remote)} tools.")
            except Exception as e:
                logger.warning(f"Failed to re-initialize updated gateway {gateway.name}: {e}")

        gateway.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(gateway)
        await self._notify_gateway_updated(gateway)
        logger.info(f"Updated gateway: {gateway.name}")
        return GatewayRead.model_validate(gateway)

    async def get_gateway(self, db: Session, gateway_id: int, include_inactive: bool = False) -> GatewayRead:
        gateway = db.get(DbGateway, gateway_id)
        if not gateway: raise GatewayNotFoundError(f"Gateway not found: {gateway_id}")
        if not gateway.is_active and not include_inactive:
            raise GatewayNotFoundError(f"Gateway '{gateway.name}' exists but is inactive")
        return GatewayRead.model_validate(gateway)

    async def toggle_gateway_status(self, db: Session, gateway_id: int, activate: bool) -> GatewayRead:
        gateway = db.get(DbGateway, gateway_id)
        if not gateway: raise GatewayNotFoundError(f"Gateway not found: {gateway_id}")
        if gateway.is_active != activate:
            gateway.is_active = activate
            gateway.updated_at = datetime.now(timezone.utc)
            if activate:
                self._active_gateways.add(gateway.url)
                try:
                    capabilities, tools_from_remote = await self._initialize_gateway(gateway.url, gateway.auth_value, gateway.transport)
                    gateway.capabilities = capabilities
                    gateway.last_seen = datetime.now(timezone.utc)
                    # TODO: Tool synchronization logic on activation
                    logger.info(f"Gateway {gateway.name} reactivated, fetched {len(tools_from_remote)} tools.")
                except Exception as e:
                    logger.warning(f"Failed to initialize reactivated gateway {gateway.name}: {e}")
                    gateway.is_active = False # Rollback activation if init fails
            else:
                self._active_gateways.discard(gateway.url)

            db.commit(); db.refresh(gateway)

            associated_tools = db.execute(select(DbTool).where(DbTool.gateway_id == gateway.id)).scalars().all()
            for tool_in_db in associated_tools:
                await self.tool_service.toggle_tool_status(db, tool_in_db.id, activate)

            if activate: await self._notify_gateway_activated(gateway)
            else: await self._notify_gateway_deactivated(gateway)
            logger.info(f"Gateway {gateway.name} {'activated' if activate else 'deactivated'}")
        return GatewayRead.model_validate(gateway)

    async def delete_gateway(self, db: Session, gateway_id: int) -> None:
        gateway = db.get(DbGateway, gateway_id)
        if not gateway: raise GatewayNotFoundError(f"Gateway not found: {gateway_id}")
        gateway_info = {"id": gateway.id, "name": gateway.name, "url": gateway.url}
        db.execute(delete(DbTool).where(DbTool.gateway_id == gateway_id))
        db.delete(gateway)
        db.commit()
        self._active_gateways.discard(gateway_info["url"])
        await self._notify_gateway_deleted(gateway_info)
        logger.info(f"Permanently deleted gateway: {gateway_info['name']}")

    async def forward_request(self, gateway: DbGateway, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if not gateway.is_active: raise GatewayConnectionError(f"Cannot forward request to inactive gateway: {gateway.name}")
        try:
            request_payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method}
            if params: request_payload["params"] = params

            auth_headers: Dict[str, str] = {}
            if gateway.auth_value:
                decoded = decode_auth(gateway.auth_value)
                if isinstance(decoded, dict): auth_headers = {str(k):str(v) for k,v in decoded.items()}

            response = await self._http_client.post(f"{gateway.url}/rpc", json=request_payload, headers=auth_headers)
            response.raise_for_status()
            result = response.json()

            # Update last_seen for the gateway instance passed in, if it's an ORM object.
            # This won't persist to DB unless the caller handles it.
            # A better place for persistent last_seen updates is after successful health checks or initializations.
            if isinstance(gateway, DbGateway):
                 gateway.last_seen = datetime.now(timezone.utc)
            # Not committing here as this method might not have a db session.

            if "error" in result: raise GatewayError(f"Gateway error: {result['error'].get('message')}")
            return result.get("result")
        except Exception as e:
            raise GatewayConnectionError(f"Failed to forward request to {gateway.name}: {str(e)}")

    async def _handle_gateway_failure(self, gateway: DbGateway) -> None:
        if GW_FAILURE_THRESHOLD == -1: return

        gateway_id = gateway.id
        count = self._gateway_failure_counts.get(gateway_id, 0) + 1
        self._gateway_failure_counts[gateway_id] = count
        logger.warning(f"Gateway {gateway.name} failed health check {count} time(s).")
        if count >= GW_FAILURE_THRESHOLD:
            logger.error(f"Gateway {gateway.name} failed {GW_FAILURE_THRESHOLD} times. Deactivating...")
            with SessionLocal() as db:
                await self.toggle_gateway_status(db, gateway_id, False)
                # db.commit() # toggle_gateway_status already commits
            self._gateway_failure_counts[gateway_id] = 0

    async def check_health_of_gateways(self, gateways: List[DbGateway]) -> bool:
        all_healthy = True
        for gateway_instance in gateways:
            if not gateway_instance.is_active: continue
            try:
                # _initialize_gateway performs a connection and capabilities check
                await self._initialize_gateway(gateway_instance.url, gateway_instance.auth_value, gateway_instance.transport)
                # If successful, update last_seen in the DB
                with SessionLocal() as db:
                    db_gw = db.get(DbGateway, gateway_instance.id)
                    if db_gw:
                        db_gw.last_seen = datetime.now(timezone.utc)
                        db.commit()
                if gateway_instance.id in self._gateway_failure_counts:
                     self._gateway_failure_counts[gateway_instance.id] = 0
            except GatewayConnectionError:
                logger.warning(f"Health check failed for gateway {gateway_instance.name} (URL: {gateway_instance.url}) due to connection error.")
                await self._handle_gateway_failure(gateway_instance)
                all_healthy = False
            except Exception as e:
                logger.error(f"Unexpected error during health check for gateway {gateway_instance.name}: {e}")
                await self._handle_gateway_failure(gateway_instance)
                all_healthy = False
        return all_healthy

    async def aggregate_capabilities(self, db: Session) -> Dict[str, Any]:
        capabilities: Dict[str, Any] = {"prompts": {"listChanged": True}, "resources": {"subscribe": True, "listChanged": True}, "tools": {"listChanged": True}, "logging": {},}
        gateways = db.execute(select(DbGateway).where(DbGateway.is_active)).scalars().all()
        for gateway_obj in gateways:
            if gateway_obj.capabilities and isinstance(gateway_obj.capabilities, dict): # Ensure capabilities is a dict
                for key, value in gateway_obj.capabilities.items():
                    if key not in capabilities: capabilities[key] = value
                    elif isinstance(value, dict) and isinstance(capabilities.get(key), dict):
                        capabilities[key].update(value)
        return capabilities

    async def subscribe_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._event_subscribers.append(queue)
        try:
            while True: event = await queue.get(); yield event
        finally: self._event_subscribers.remove(queue)

    async def _publish_event(self, event: Dict[str, Any]) -> None:
        for queue in self._event_subscribers: await queue.put(event)

    async def _notify_gateway_added(self, gateway: DbGateway) -> None:
        await self._publish_event({"type": "gateway_added", "data": GatewayRead.model_validate(gateway).model_dump(by_alias=True), "timestamp": datetime.utcnow().isoformat()})
    async def _notify_gateway_updated(self, gateway: DbGateway) -> None:
        await self._publish_event({"type": "gateway_updated", "data": GatewayRead.model_validate(gateway).model_dump(by_alias=True), "timestamp": datetime.utcnow().isoformat()})
    async def _notify_gateway_activated(self, gateway: DbGateway) -> None:
        await self._publish_event({"type": "gateway_activated", "data": {"id": gateway.id, "name": gateway.name, "is_active": True}, "timestamp": datetime.utcnow().isoformat()})
    async def _notify_gateway_deactivated(self, gateway: DbGateway) -> None:
        await self._publish_event({"type": "gateway_deactivated", "data": {"id": gateway.id, "name": gateway.name, "is_active": False}, "timestamp": datetime.utcnow().isoformat()})
    async def _notify_gateway_deleted(self, gateway_info: Dict[str, Any]) -> None:
        await self._publish_event({"type": "gateway_deleted", "data": gateway_info, "timestamp": datetime.utcnow().isoformat()})
    async def _notify_gateway_removed(self, gateway: DbGateway) -> None:
        await self._notify_gateway_deactivated(gateway)

    async def _run_health_checks(self, assume_leader_for_non_redis: bool = False) -> None:
        while True:
            is_leader = False
            if self._redis_client and self._leader_key and self._instance_id and hasattr(self._redis_client, 'get'): # Check methods for redis client
                try:
                    current_leader_bytes = await self._redis_client.get(self._leader_key)
                    current_leader = current_leader_bytes.decode() if current_leader_bytes else None
                    if current_leader == self._instance_id:
                        await self._redis_client.expire(self._leader_key, self._leader_ttl)
                        is_leader = True
                    elif not current_leader:
                        is_leader = await self._redis_client.set(self._leader_key, self._instance_id, ex=self._leader_ttl, nx=True)
                    if is_leader: logger.debug(f"Instance {self._instance_id} is leader / renewed leadership.")
                    else: logger.debug(f"Instance {self._instance_id} is not leader. Current leader: {current_leader}")
                except Exception as e: logger.error(f"Redis error during leader election: {e}")
            elif hasattr(self, '_file_lock'):
                try:
                    self._file_lock.acquire(timeout=0.1)
                    is_leader = True
                    logger.debug(f"Instance {self._instance_id if hasattr(self, '_instance_id') else 'N/A'} acquired file lock, is leader.")
                except Timeout:
                    logger.debug(f"Instance {self._instance_id if hasattr(self, '_instance_id') else 'N/A'} could not acquire file lock, not leader.")
                    is_leader = False
                except Exception as e:
                    logger.error(f"FileLock error during leader election: {e}")
                    is_leader = False
            elif assume_leader_for_non_redis:
                 is_leader = True

            if is_leader:
                logger.info("Leader instance performing health checks.")
                try:
                    with SessionLocal() as db:
                        active_gateways = db.execute(select(DbGateway).where(DbGateway.is_active)).scalars().all()
                        if active_gateways:
                            logger.debug(f"Checking health of {len(active_gateways)} active gateways.")
                            await self.check_health_of_gateways(active_gateways)
                            db.commit()
                        else:
                            logger.debug("No active gateways to health check.")
                except Exception as e:
                    logger.error(f"Error during health check cycle: {e}", exc_info=True)
                finally:
                    if hasattr(self, '_file_lock') and self._file_lock.is_locked:
                        self._file_lock.release()
                        logger.debug(f"Instance {self._instance_id if hasattr(self, '_instance_id') else 'N/A'} released file lock.")
            await asyncio.sleep(self._health_check_interval)
