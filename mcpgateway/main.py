# -*- coding: utf-8 -*-
"""
Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Mihai Criveti

MCP Gateway - Main FastAPI Application.

This module defines the core FastAPI application for the Model Context Protocol (MCP) Gateway.
It serves as the entry point for handling all HTTP and WebSocket traffic.

Features and Responsibilities:
- Initializes and orchestrates services for tools, resources, prompts, servers, gateways, and roots.
- Supports full MCP protocol operations: initialize, ping, notify, complete, and sample.
- Integrates authentication (JWT and basic), CORS, caching, and middleware.
- Serves a rich Admin UI for managing gateway entities via HTMX-based frontend.
- Exposes routes for JSON-RPC, SSE, and WebSocket transports.
- Manages application lifecycle including startup and graceful shutdown of all services.

Structure:
- Declares routers for MCP protocol operations and administration.
- Registers dependencies (e.g., DB sessions, auth handlers).
- Applies middleware including custom documentation protection.
- Configures resource caching and session registry using pluggable backends.
- Provides OpenAPI metadata and redirect handling depending on UI feature flags.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Union, Generator # Added Generator

import httpx
from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.background import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from sqlalchemy.orm import Session
from starlette.middleware.base import BaseHTTPMiddleware

from mcpgateway import __version__
from mcpgateway.admin import admin_router
from mcpgateway.cache import ResourceCache, SessionRegistry
from mcpgateway.config import jsonpath_modifier, settings
from mcpgateway.db import Base, SessionLocal, engine
from mcpgateway.handlers.sampling import SamplingHandler
from mcpgateway.schemas import (
    GatewayCreate,
    GatewayRead,
    GatewayUpdate,
    JsonPathModifier,
    PromptCreate,
    PromptRead,
    PromptUpdate,
    ResourceCreate,
    ResourceRead,
    ResourceUpdate,
    ServerCreate,
    ServerRead,
    ServerUpdate,
    ToolCreate,
    ToolRead,
    ToolUpdate,
)
from mcpgateway.services.completion_service import CompletionService
from mcpgateway.services.gateway_service import GatewayConnectionError, GatewayService, GatewayNameConflictError # Added
from mcpgateway.services.logging_service import LoggingService
from mcpgateway.services.prompt_service import (
    PromptError,
    PromptNameConflictError,
    PromptNotFoundError,
    PromptService,
)
from mcpgateway.services.resource_service import (
    ResourceError,
    ResourceNotFoundError,
    ResourceService,
    ResourceURIConflictError,
)
from mcpgateway.services.root_service import RootService
from mcpgateway.services.server_service import (
    ServerError,
    ServerNameConflictError,
    ServerNotFoundError,
    ServerService,
)
from mcpgateway.services.tool_service import (
    ToolError,
    ToolNameConflictError,
    ToolService,
    ToolNotFoundError, # Added
)
from mcpgateway.transports.sse_transport import SSETransport
from mcpgateway.transports.streamablehttp_transport import (
    SessionManagerWrapper,
    streamable_http_auth,
)
from mcpgateway.types import (
    InitializeRequest,
    InitializeResult,
    ListResourceTemplatesResult,
    LogLevel,
    ResourceContent,
    Root as RootType, # Renamed to avoid conflict with root_router
)
from mcpgateway.utils.verify_credentials import require_auth, require_auth_override
from mcpgateway.validation.jsonrpc import (
    JSONRPCError,
    INVALID_REQUEST,
    METHOD_NOT_FOUND, # Added
    validate_request,
)

# Import the admin routes from the new module
from mcpgateway.version import router as version_router

# Initialize logging service first
logging_service = LoggingService()
logger = logging_service.get_logger("mcpgateway")

# Configure root logger level
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create database tables
Base.metadata.create_all(bind=engine)

# Initialize services
tool_service = ToolService()
resource_service = ResourceService()
prompt_service = PromptService()
gateway_service = GatewayService()
root_service = RootService()
completion_service = CompletionService()
sampling_handler = SamplingHandler()
server_service = ServerService()

# Initialize session manager for Streamable HTTP transport
streamable_http_session = SessionManagerWrapper()


# Initialize session registry
session_registry = SessionRegistry(
    backend=settings.cache_type,
    redis_url=settings.redis_url if settings.cache_type == "redis" else None,
    database_url=settings.database_url if settings.cache_type == "database" else None,
    session_ttl=settings.session_ttl,
    message_ttl=settings.message_ttl,
)

# Initialize cache
resource_cache = ResourceCache(max_size=settings.resource_cache_size, ttl=settings.resource_cache_ttl)


####################
# Startup/Shutdown #
####################
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting MCP Gateway services")
    try:
        await tool_service.initialize()
        await resource_service.initialize()
        await prompt_service.initialize()
        await gateway_service.initialize()
        await root_service.initialize()
        await completion_service.initialize()
        await logging_service.initialize()
        await sampling_handler.initialize()
        await resource_cache.initialize()
        await streamable_http_session.initialize()
        await session_registry.initialize() # Added session_registry init

        logger.info("All services initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    finally:
        logger.info("Shutting down MCP Gateway services")

        _services_to_shutdown = [
            session_registry, # Shutdown session_registry first
            resource_cache, sampling_handler, logging_service, completion_service,
            root_service, gateway_service, prompt_service, # resource_service is listed later
            tool_service, streamable_http_session, resource_service # resource_service was duplicate, now only one
        ]

        for service_instance in _services_to_shutdown:
            try:
                if hasattr(service_instance, "shutdown"):
                    shutdown_method = getattr(service_instance, "shutdown")
                    if asyncio.iscoroutinefunction(shutdown_method):
                        await shutdown_method()
                    elif callable(shutdown_method): # Check if it's callable for non-async
                        shutdown_method()
            except Exception as e:
                logger.error(f"Error shutting down {service_instance.__class__.__name__}: {str(e)}")
        logger.info("Shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="A FastAPI-based MCP Gateway with federation support",
    root_path=settings.app_root_path,
    lifespan=lifespan,
)


class DocsAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        protected_paths = ["/docs", "/redoc", "/openapi.json"]
        if any(request.url.path.startswith(p) for p in protected_paths):
            try:
                token = request.headers.get("Authorization")
                cookie_token = request.cookies.get("jwt_token")
                await require_auth_override(token, cookie_token)
            except HTTPException as e:
                return JSONResponse(status_code=e.status_code, content={"detail": e.detail}, headers=e.headers if e.headers else None)
        return await call_next(request)


class MCPPathRewriteMiddleware:
    def __init__(self, app_to_wrap):
        self.app = app_to_wrap

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        auth_ok = await streamable_http_auth(scope, receive, send)
        if not auth_ok:
            return

        original_path = scope.get("path", "")
        scope["modified_path"] = original_path
        if (original_path.endswith("/mcp") and original_path != "/mcp") or \
           (original_path.endswith("/mcp/") and original_path != "/mcp/"):
            scope["path"] = "/mcp"
            await streamable_http_session.handle_streamable_http(scope, receive, send)
            return
        await self.app(scope, receive, send)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.allowed_origins else list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Content-Length"],
)
app.add_middleware(DocsAuthMiddleware)
app.add_middleware(MCPPathRewriteMiddleware)

templates = Jinja2Templates(directory=str(settings.templates_dir))
app.state.templates = templates

protocol_router = APIRouter(prefix="/protocol", tags=["Protocol"])
tool_router = APIRouter(prefix="/tools", tags=["Tools"])
resource_router = APIRouter(prefix="/resources", tags=["Resources"])
prompt_router = APIRouter(prefix="/prompts", tags=["Prompts"])
gateway_router = APIRouter(prefix="/gateways", tags=["Gateways"])
root_router_obj = APIRouter(prefix="/roots", tags=["Roots"]) # Renamed to avoid conflict
utility_router = APIRouter(tags=["Utilities"])
server_router = APIRouter(prefix="/servers", tags=["Servers"])
metrics_router = APIRouter(prefix="/metrics", tags=["Metrics"])

def get_db() -> Generator[Session, None, None]: # Corrected Generator import
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def require_api_key(api_key: str) -> None:
    if settings.auth_required:
        expected = f"{settings.basic_auth_user}:{settings.basic_auth_password}"
        if api_key != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

async def invalidate_resource_cache(uri: Optional[str] = None) -> None:
    if uri:
        resource_cache.delete(uri)
    else:
        resource_cache.clear()

@protocol_router.post("/initialize")
async def initialize(request: Request, user: str = Depends(require_auth)) -> InitializeResult:
    try:
        body = await request.json()
        logger.debug(f"Authenticated user {user} is initializing the protocol.")
        return await session_registry.handle_initialize_logic(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON in request body")

@protocol_router.post("/ping")
async def ping(request: Request, user: str = Depends(require_auth)) -> JSONResponse:
    body: Dict[str, Any] = {}
    try:
        body = await request.json()
        if body.get("method") != "ping":
            raise HTTPException(status_code=400, detail="Invalid method")
        req_id: Optional[Union[str,int]] = body.get("id")
        logger.debug(f"Authenticated user {user} sent ping request.")
        response: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "result": {}}
        return JSONResponse(content=response)
    except Exception as e:
        error_response: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": body.get("id") if "body" in locals() and isinstance(body, dict) else None,
            "error": {"code": -32603, "message": "Internal error", "data": str(e)},
        }
        return JSONResponse(status_code=500, content=error_response)

@protocol_router.post("/notifications")
async def handle_notification(request: Request, user: str = Depends(require_auth)) -> None:
    body = await request.json()
    logger.debug(f"User {user} sent a notification")
    if body.get("method") == "notifications/initialized":
        logger.info("Client initialized")
        await logging_service.notify("Client initialized", LogLevel.INFO)
    elif body.get("method") == "notifications/cancelled":
        request_id = body.get("params", {}).get("requestId")
        logger.info(f"Request cancelled: {request_id}")
        await logging_service.notify(f"Request cancelled: {request_id}", LogLevel.INFO)
    elif body.get("method") == "notifications/message":
        params = body.get("params", {})
        await logging_service.notify(
            params.get("data"),
            LogLevel(params.get("level", "info")),
            params.get("logger"),
        )

@protocol_router.post("/completion/complete")
async def handle_completion(request: Request, db: Session = Depends(get_db), user: str = Depends(require_auth)):
    body = await request.json()
    logger.debug(f"User {user} sent a completion request")
    return await completion_service.handle_completion(db, body)

@protocol_router.post("/sampling/createMessage")
async def handle_sampling(request: Request, db: Session = Depends(get_db), user: str = Depends(require_auth)):
    logger.debug(f"User {user} sent a sampling request")
    body = await request.json()
    return await sampling_handler.create_message(db, body)

@server_router.get("", response_model=List[ServerRead])
@server_router.get("/", response_model=List[ServerRead])
async def list_servers(
    include_inactive: bool = False,
    db: Session = Depends(get_db),
    user: str = Depends(require_auth),
) -> List[ServerRead]:
    logger.debug(f"User {user} requested server list")
    return await server_service.list_servers(db, include_inactive=include_inactive)

@server_router.get("/{server_id}", response_model=ServerRead)
async def get_server(server_id: int, db: Session = Depends(get_db), user: str = Depends(require_auth)) -> ServerRead:
    try:
        logger.debug(f"User {user} requested server with ID {server_id}")
        return await server_service.get_server(db, server_id)
    except ServerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@server_router.post("", response_model=ServerRead, status_code=201)
@server_router.post("/", response_model=ServerRead, status_code=201)
async def create_server(
    server: ServerCreate,
    db: Session = Depends(get_db),
    user: str = Depends(require_auth),
) -> ServerRead:
    try:
        logger.debug(f"User {user} is creating a new server")
        return await server_service.register_server(db, server)
    except ServerNameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ServerError as e:
        raise HTTPException(status_code=400, detail=str(e))

@server_router.put("/{server_id}", response_model=ServerRead)
async def update_server(
    server_id: int,
    server: ServerUpdate,
    db: Session = Depends(get_db),
    user: str = Depends(require_auth),
) -> ServerRead:
    try:
        logger.debug(f"User {user} is updating server with ID {server_id}")
        return await server_service.update_server(db, server_id, server)
    except ServerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ServerNameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ServerError as e:
        raise HTTPException(status_code=400, detail=str(e))

@server_router.post("/{server_id}/toggle", response_model=ServerRead)
async def toggle_server_status(
    server_id: int,
    activate: bool = True,
    db: Session = Depends(get_db),
    user: str = Depends(require_auth),
) -> ServerRead:
    try:
        logger.debug(f"User {user} is toggling server with ID {server_id} to {'active' if activate else 'inactive'}")
        return await server_service.toggle_server_status(db, server_id, activate)
    except ServerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ServerError as e:
        raise HTTPException(status_code=400, detail=str(e))

@server_router.delete("/{server_id}", response_model=Dict[str, str])
async def delete_server(server_id: int, db: Session = Depends(get_db), user: str = Depends(require_auth)) -> Dict[str, str]:
    try:
        logger.debug(f"User {user} is deleting server with ID {server_id}")
        await server_service.delete_server(db, server_id)
        return {"status": "success", "message": f"Server {server_id} deleted successfully"}
    except ServerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ServerError as e:
        raise HTTPException(status_code=400, detail=str(e))

@server_router.get("/{server_id}/sse")
async def sse_endpoint(request: Request, server_id: int, user: str = Depends(require_auth)):
    try:
        logger.debug(f"User {user} is establishing SSE connection for server {server_id}")
        base_url = str(request.base_url).rstrip("/")
        server_sse_url = f"{base_url}/servers/{server_id}"
        transport = SSETransport(base_url=server_sse_url)
        await transport.connect()
        await session_registry.add_session(transport.session_id, transport)
        # Pass server_id as string to session_registry.respond
        asyncio.create_task(session_registry.respond(str(server_id), user, session_id=transport.session_id, base_url=base_url))
        response = await transport.create_sse_response(request)
        tasks = BackgroundTasks()
        tasks.add_task(session_registry.remove_session, transport.session_id)
        response.background = tasks
        logger.info(f"SSE connection established: {transport.session_id}")
        return response
    except Exception as e:
        logger.error(f"SSE connection error: {e}")
        raise HTTPException(status_code=500, detail="SSE connection failed")

@server_router.post("/{server_id}/message")
async def message_endpoint(request: Request, server_id: int, user: str = Depends(require_auth)):
    try:
        logger.debug(f"User {user} sent a message to server {server_id}")
        session_id = request.query_params.get("session_id")
        if not session_id:
            logger.error("Missing session_id in message request")
            raise HTTPException(status_code=400, detail="Missing session_id")
        message = await request.json()
        await session_registry.broadcast(session_id=session_id, message=message)
        return JSONResponse(content={"status": "success"}, status_code=202)
    except ValueError as e:
        logger.error(f"Invalid message format: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Message handling error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process message")

@server_router.get("/{server_id}/tools", response_model=List[ToolRead])
async def server_get_tools(
    server_id: int, include_inactive: bool = False, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> List[ToolRead]:
    logger.debug(f"User: {user} has listed tools for the server_id: {server_id}")
    tools = await tool_service.list_server_tools(db, server_id=server_id, include_inactive=include_inactive)
    return [ToolRead.model_validate(tool.model_dump(by_alias=True)) for tool in tools] # Use model_validate

@server_router.get("/{server_id}/resources", response_model=List[ResourceRead])
async def server_get_resources(
    server_id: int, include_inactive: bool = False, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> List[ResourceRead]:
    logger.debug(f"User: {user} has listed resources for the server_id: {server_id}")
    resources = await resource_service.list_server_resources(db, server_id=server_id, include_inactive=include_inactive)
    return [ResourceRead.model_validate(res.model_dump(by_alias=True)) for res in resources] # Use model_validate

@server_router.get("/{server_id}/prompts", response_model=List[PromptRead])
async def server_get_prompts(
    server_id: int, include_inactive: bool = False, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> List[PromptRead]:
    logger.debug(f"User: {user} has listed prompts for the server_id: {server_id}")
    prompts = await prompt_service.list_server_prompts(db, server_id=server_id, include_inactive=include_inactive)
    return [PromptRead.model_validate(p.model_dump(by_alias=True)) for p in prompts] # Use model_validate

@tool_router.get("", response_model=Union[List[ToolRead], List[Any], Dict[str, Any]]) # Adjusted response_model
@tool_router.get("/", response_model=Union[List[ToolRead], List[Any], Dict[str, Any]]) # Adjusted response_model
async def list_tools(
    cursor: Optional[str] = None,
    include_inactive: bool = False,
    db: Session = Depends(get_db),
    apijsonpath: Optional[JsonPathModifier] = Body(None), # Made Optional
    _: str = Depends(require_auth),
) -> Union[List[ToolRead], List[Any], Dict[str, Any]]: # Adjusted return hint
    data = await tool_service.list_tools(db, cursor=cursor, include_inactive=include_inactive)
    if apijsonpath is None or (apijsonpath.jsonpath is None and apijsonpath.mapping is None) :
        return data
    tools_dict_list = [tool.model_dump(by_alias=True) for tool in data] # Use model_dump
    current_jsonpath = apijsonpath.jsonpath if apijsonpath.jsonpath is not None else "$[*]"
    return jsonpath_modifier(tools_dict_list, current_jsonpath, apijsonpath.mapping)

@tool_router.post("", response_model=ToolRead)
@tool_router.post("/", response_model=ToolRead)
async def create_tool(tool: ToolCreate, db: Session = Depends(get_db), user: str = Depends(require_auth)) -> ToolRead:
    try:
        logger.debug(f"User {user} is creating a new tool")
        return await tool_service.register_tool(db, tool)
    except ToolNameConflictError as e:
        if not e.is_active and e.tool_id:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Tool name already exists but is inactive. Consider activating it with ID: {e.tool_id}")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ToolError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@tool_router.get("/{tool_id}", response_model=Union[ToolRead, List[Any], Dict[str, Any]]) # Adjusted response_model
async def get_tool(
    tool_id: int,
    db: Session = Depends(get_db),
    user: str = Depends(require_auth),
    apijsonpath: Optional[JsonPathModifier] = Body(None), # Made Optional
) -> Union[ToolRead, List[Any], Dict[str, Any]]: # Adjusted return hint
    try:
        logger.debug(f"User {user} is retrieving tool with ID {tool_id}")
        data = await tool_service.get_tool(db, tool_id)
        if apijsonpath is None or (apijsonpath.jsonpath is None and apijsonpath.mapping is None):
            return data
        data_dict = data.model_dump(by_alias=True) # Use model_dump
        current_jsonpath = apijsonpath.jsonpath if apijsonpath.jsonpath is not None else "$[*]"
        current_mappings = apijsonpath.mapping
        return jsonpath_modifier(data_dict, current_jsonpath, current_mappings)
    except ToolNotFoundError as e: # Catch specific error
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e: # General fallback
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@tool_router.put("/{tool_id}", response_model=ToolRead)
async def update_tool(
    tool_id: int, tool: ToolUpdate, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> ToolRead:
    try:
        logger.debug(f"User {user} is updating tool with ID {tool_id}")
        return await tool_service.update_tool(db, tool_id, tool)
    except Exception as e: # Consider more specific exceptions
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@tool_router.delete("/{tool_id}")
async def delete_tool(tool_id: int, db: Session = Depends(get_db), user: str = Depends(require_auth)) -> Dict[str, str]:
    try:
        logger.debug(f"User {user} is deleting tool with ID {tool_id}")
        await tool_service.delete_tool(db, tool_id)
        return {"status": "success", "message": f"Tool {tool_id} permanently deleted"}
    except Exception as e: # Consider more specific exceptions
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@tool_router.post("/{tool_id}/toggle")
async def toggle_tool_status(
    tool_id: int, activate: bool = True, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> Dict[str, Any]:
    try:
        logger.debug(f"User {user} is toggling tool with ID {tool_id} to {'active' if activate else 'inactive'}")
        tool = await tool_service.toggle_tool_status(db, tool_id, activate)
        return {"status": "success", "message": f"Tool {tool_id} {'activated' if activate else 'deactivated'}", "tool": tool.model_dump()}
    except Exception as e: # Consider more specific exceptions
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@resource_router.get("/templates/list", response_model=ListResourceTemplatesResult)
async def list_resource_templates(db: Session = Depends(get_db), user: str = Depends(require_auth)) -> ListResourceTemplatesResult:
    logger.debug(f"User {user} requested resource templates")
    resource_templates = await resource_service.list_resource_templates(db)
    return ListResourceTemplatesResult(meta={}, resource_templates=resource_templates, next_cursor=None)

@resource_router.post("/{resource_id}/toggle")
async def toggle_resource_status(
    resource_id: int, activate: bool = True, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> Dict[str, Any]:
    logger.debug(f"User {user} is toggling resource with ID {resource_id} to {'active' if activate else 'inactive'}")
    try:
        resource = await resource_service.toggle_resource_status(db, resource_id, activate)
        return {"status": "success", "message": f"Resource {resource_id} {'activated' if activate else 'deactivated'}", "resource": resource.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@resource_router.get("", response_model=List[ResourceRead])
@resource_router.get("/", response_model=List[ResourceRead])
async def list_resources(
    cursor: Optional[str] = None, include_inactive: bool = False, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> List[ResourceRead]:
    logger.debug(f"User {user} requested resource list with cursor {cursor} and include_inactive={include_inactive}")
    # if cached := resource_cache.get("resource_list"): # Caching needs to consider filters
    #     return cached
    resources = await resource_service.list_resources(db, include_inactive=include_inactive)
    # resource_cache.set("resource_list", resources) # Caching needs to consider filters
    return resources

@resource_router.post("", response_model=ResourceRead)
@resource_router.post("/", response_model=ResourceRead)
async def create_resource(
    resource: ResourceCreate, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> ResourceRead:
    logger.debug(f"User {user} is creating a new resource")
    try:
        result = await resource_service.register_resource(db, resource)
        return result
    except ResourceURIConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ResourceError as e:
        raise HTTPException(status_code=400, detail=str(e))

@resource_router.get("/{uri:path}", response_model=ResourceContent) # Added response_model
async def read_resource(uri: str, db: Session = Depends(get_db), user: str = Depends(require_auth)) -> ResourceContent:
    logger.debug(f"User {user} requested resource with URI {uri}")
    # if cached := resource_cache.get(uri): # Caching should be handled carefully with auth
    #     return cached
    try:
        content: ResourceContent = await resource_service.read_resource(db, uri)
    except (ResourceNotFoundError, ResourceError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    # resource_cache.set(uri, content)
    return content

@resource_router.put("/{uri:path}", response_model=ResourceRead)
async def update_resource(
    uri: str, resource: ResourceUpdate, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> ResourceRead:
    try:
        logger.debug(f"User {user} is updating resource with URI {uri}")
        result = await resource_service.update_resource(db, uri, resource)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    await invalidate_resource_cache(uri)
    return result

@resource_router.delete("/{uri:path}")
async def delete_resource(uri: str, db: Session = Depends(get_db), user: str = Depends(require_auth)) -> Dict[str, str]:
    try:
        logger.debug(f"User {user} is deleting resource with URI {uri}")
        await resource_service.delete_resource(db, uri)
        await invalidate_resource_cache(uri)
        return {"status": "success", "message": f"Resource {uri} deleted"}
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ResourceError as e:
        raise HTTPException(status_code=400, detail=str(e))

@resource_router.post("/subscribe/{uri:path}")
async def subscribe_resource(uri: str, user: str = Depends(require_auth)) -> StreamingResponse:
    logger.debug(f"User {user} is subscribing to resource with URI {uri}")
    return StreamingResponse(resource_service.subscribe_events(uri), media_type="text/event-stream")

@prompt_router.post("/{prompt_id}/toggle")
async def toggle_prompt_status(
    prompt_id: int, activate: bool = True, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> Dict[str, Any]:
    logger.debug(f"User: {user} requested toggle for prompt {prompt_id}, activate={activate}")
    try:
        prompt = await prompt_service.toggle_prompt_status(db, prompt_id, activate)
        return {"status": "success", "message": f"Prompt {prompt_id} {'activated' if activate else 'deactivated'}", "prompt": prompt.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@prompt_router.get("", response_model=List[PromptRead])
@prompt_router.get("/", response_model=List[PromptRead])
async def list_prompts(
    cursor: Optional[str] = None, include_inactive: bool = False, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> List[PromptRead]:
    logger.debug(f"User: {user} requested prompt list with include_inactive={include_inactive}, cursor={cursor}")
    return await prompt_service.list_prompts(db, cursor=cursor, include_inactive=include_inactive)

@prompt_router.post("", response_model=PromptRead)
@prompt_router.post("/", response_model=PromptRead)
async def create_prompt(
    prompt: PromptCreate, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> PromptRead:
    logger.debug(f"User: {user} requested to create prompt: {prompt}")
    try:
        return await prompt_service.register_prompt(db, prompt)
    except PromptNameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except PromptError as e:
        raise HTTPException(status_code=400, detail=str(e))

@prompt_router.post("/{name}")
async def get_prompt(
    name: str, args: Dict[str, str] = Body(default_factory=dict), db: Session = Depends(get_db), user: str = Depends(require_auth) # Changed Body({}) to Body(default_factory=dict)
) -> Any: # Return type can be PromptResult
    logger.debug(f"User: {user} requested prompt: {name} with args={args}")
    return await prompt_service.get_prompt(db, name, args)

@prompt_router.get("/{name}")
async def get_prompt_no_args(
    name: str, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> Any: # Return type can be PromptResult
    logger.debug(f"User: {user} requested prompt: {name} with no arguments")
    return await prompt_service.get_prompt(db, name, {})

@prompt_router.put("/{name}", response_model=PromptRead)
async def update_prompt(
    name: str, prompt: PromptUpdate, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> PromptRead:
    logger.debug(f"User: {user} requested to update prompt: {name} with data={prompt}")
    try:
        return await prompt_service.update_prompt(db, name, prompt)
    except PromptNameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except PromptError as e:
        raise HTTPException(status_code=400, detail=str(e))

@prompt_router.delete("/{name}")
async def delete_prompt(name: str, db: Session = Depends(get_db), user: str = Depends(require_auth)) -> Dict[str, str]:
    logger.debug(f"User: {user} requested deletion of prompt {name}")
    try:
        await prompt_service.delete_prompt(db, name)
        return {"status": "success", "message": f"Prompt {name} deleted"}
    except PromptNotFoundError as e: # Consider returning 404
        return JSONResponse(status_code=404, content={"status": "error", "message": str(e)})
    except PromptError as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

@gateway_router.post("/{gateway_id}/toggle")
async def toggle_gateway_status(
    gateway_id: int, activate: bool = True, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> Dict[str, Any]:
    logger.debug(f"User '{user}' requested toggle for gateway {gateway_id}, activate={activate}")
    try:
        gateway = await gateway_service.toggle_gateway_status(db, gateway_id, activate)
        return {"status": "success", "message": f"Gateway {gateway_id} {'activated' if activate else 'deactivated'}", "gateway": gateway.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@gateway_router.get("", response_model=List[GatewayRead])
@gateway_router.get("/", response_model=List[GatewayRead])
async def list_gateways(
    include_inactive: bool = False, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> List[GatewayRead]:
    logger.debug(f"User '{user}' requested list of gateways with include_inactive={include_inactive}")
    return await gateway_service.list_gateways(db, include_inactive=include_inactive)

@gateway_router.post("", response_model=GatewayRead)
@gateway_router.post("/", response_model=GatewayRead)
async def register_gateway(
    gateway: GatewayCreate, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> GatewayRead:
    logger.debug(f"User '{user}' requested to register gateway: {gateway}")
    try:
        return await gateway_service.register_gateway(db, gateway)
    except GatewayConnectionError as e: # Specific error handling
        raise HTTPException(status_code=502, detail=f"Unable to connect to gateway: {str(e)}")
    except ValueError as e: # From Pydantic or other validation
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except GatewayNameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e: # General fallback
        logger.error(f"Unexpected error registering gateway: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@gateway_router.get("/{gateway_id}", response_model=GatewayRead)
async def get_gateway(gateway_id: int, db: Session = Depends(get_db), user: str = Depends(require_auth)) -> GatewayRead:
    logger.debug(f"User '{user}' requested gateway {gateway_id}")
    return await gateway_service.get_gateway(db, gateway_id)

@gateway_router.put("/{gateway_id}", response_model=GatewayRead)
async def update_gateway(
    gateway_id: int, gateway: GatewayUpdate, db: Session = Depends(get_db), user: str = Depends(require_auth)
) -> GatewayRead:
    logger.debug(f"User '{user}' requested update on gateway {gateway_id} with data={gateway}")
    return await gateway_service.update_gateway(db, gateway_id, gateway)

@gateway_router.delete("/{gateway_id}")
async def delete_gateway(gateway_id: int, db: Session = Depends(get_db), user: str = Depends(require_auth)) -> Dict[str, str]:
    logger.debug(f"User '{user}' requested deletion of gateway {gateway_id}")
    await gateway_service.delete_gateway(db, gateway_id)
    return {"status": "success", "message": f"Gateway {gateway_id} deleted"}

@root_router_obj.get("", response_model=List[RootType]) # Use aliased RootType
@root_router_obj.get("/", response_model=List[RootType]) # Use aliased RootType
async def list_roots(user: str = Depends(require_auth)) -> List[RootType]:
    logger.debug(f"User '{user}' requested list of roots")
    return await root_service.list_roots()

@root_router_obj.post("", response_model=RootType) # Use aliased RootType
@root_router_obj.post("/", response_model=RootType) # Use aliased RootType
async def add_root(root: RootType, user: str = Depends(require_auth)) -> RootType:
    logger.debug(f"User '{user}' requested to add root: {root}")
    return await root_service.add_root(str(root.uri), root.name)

@root_router_obj.delete("/{uri:path}")
async def remove_root(uri: str, user: str = Depends(require_auth)) -> Dict[str, str]:
    logger.debug(f"User '{user}' requested to remove root with URI: {uri}")
    await root_service.remove_root(uri)
    return {"status": "success", "message": f"Root {uri} removed"}

@root_router_obj.get("/changes")
async def subscribe_roots_changes(user: str = Depends(require_auth)) -> StreamingResponse:
    logger.debug(f"User '{user}' subscribed to root changes stream")
    return StreamingResponse(root_service.subscribe_changes(), media_type="text/event-stream")

@utility_router.post("/rpc/")
@utility_router.post("/rpc")
async def handle_rpc(request: Request, db: Session = Depends(get_db), user: str = Depends(require_auth)) -> JSONResponse:
    body: Dict[str, Any] = {}
    try:
        logger.debug(f"User {user} made an RPC request")
        body = await request.json()
        validate_request(body)

        method_any = body.get("method")
        if not isinstance(method_any, str) or not method_any:
            raise JSONRPCError(INVALID_REQUEST, "Method is missing or not a string", request_id=body.get("id"))
        method: str = method_any

        params_any = body.get("params", {})
        # JSON-RPC params can be an array or object. Default to object if not specified.
        params: Union[Dict[str, Any], List[Any]] = params_any if isinstance(params_any, (dict, list)) else {}

        req_id = body.get("id")
        result_data: Any = {}

        # Extract cursor parameter safely if params is a dict
        cursor: Optional[str] = None
        if isinstance(params, dict):
            cursor = params.get("cursor")

        if method == "tools/list":
            tools = await tool_service.list_tools(db, cursor=cursor)
            result_data = [t.model_dump(by_alias=True, exclude_none=True) for t in tools]
        elif method == "list_tools":
            tools = await tool_service.list_tools(db, cursor=cursor)
            result_data = [t.model_dump(by_alias=True, exclude_none=True) for t in tools]
        elif method == "initialize":
            # Ensure params for initialize is a dict for InitializeRequest
            init_params_dict = params if isinstance(params, dict) else {}
            init_req = InitializeRequest(
                protocol_version=init_params_dict.get("protocolVersion") or init_params_dict.get("protocol_version", ""),
                capabilities=init_params_dict.get("capabilities", {}),
                client_info=init_params_dict.get("clientInfo") or init_params_dict.get("client_info", {}),
            )
            # initialize function is already defined above for the /protocol/initialize route
            # Re-using its logic might be better or calling session_registry.handle_initialize_logic
            init_result = await session_registry.handle_initialize_logic(init_req.model_dump(by_alias=True))
            result_data = init_result.model_dump(by_alias=True, exclude_none=True)

        elif method == "list_gateways":
            gateways = await gateway_service.list_gateways(db, include_inactive=False)
            result_data = [g.model_dump(by_alias=True, exclude_none=True) for g in gateways]
        elif method == "list_roots":
            roots = await root_service.list_roots()
            result_data = [r.model_dump(by_alias=True, exclude_none=True) for r in roots]
        elif method == "resources/list":
            resources = await resource_service.list_resources(db)
            result_data = [r.model_dump(by_alias=True, exclude_none=True) for r in resources]
        elif method == "prompts/list":
            prompts = await prompt_service.list_prompts(db, cursor=cursor)
            result_data = [p.model_dump(by_alias=True, exclude_none=True) for p in prompts]
        elif method == "prompts/get":
            prompt_params = params if isinstance(params, dict) else {}
            name = prompt_params.get("name")
            arguments = prompt_params.get("arguments", {})
            if not name or not isinstance(name, str):
                raise JSONRPCError(INVALID_REQUEST, "Missing or invalid prompt name in parameters", request_id=req_id)
            if not isinstance(arguments, dict):
                raise JSONRPCError(INVALID_REQUEST, "Prompt arguments must be an object", request_id=req_id)

            prompt_res = await prompt_service.get_prompt(db, name, arguments)
            if hasattr(prompt_res, "model_dump"):
                result_data = prompt_res.model_dump(by_alias=True, exclude_none=True)
            else: # Should not happen if get_prompt returns PromptResult
                result_data = prompt_res
        elif method == "ping":
            result_data = {}
        else: # Default to tool invocation
            tool_params = params if isinstance(params, dict) else {}
            try:
                tool_res = await tool_service.invoke_tool(db, method, tool_params)
                if hasattr(tool_res, "model_dump"):
                    result_data = tool_res.model_dump(by_alias=True, exclude_none=True)
                else: # Should not happen if invoke_tool returns ToolResult
                    result_data = tool_res
            except ToolNotFoundError: # If tool_service can't find it, try gateway_service
                 # This part is problematic as gateway_service.forward_request has different signature
                 # and is not meant for general tool invocation. This path needs review.
                 # For now, assume tool_service is the sole handler for tool calls by name.
                raise JSONRPCError(METHOD_NOT_FOUND, f"Method not found: {method}", request_id=req_id)


        return JSONResponse(content={"jsonrpc": "2.0", "result": result_data, "id": req_id})

    except JSONRPCError as e:
        return JSONResponse(status_code=200, content=e.to_dict()) # RPC errors return 200
    except HTTPException as e: # Re-raise HTTPExceptions to let FastAPI handle them
        raise e
    except Exception as e:
        logger.error(f"RPC error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=200, # RPC errors return 200
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": "Internal error", "data": str(e)},
                "id": body.get("id") if body else None,
            }
        )

@utility_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        while True:
            try:
                data = await websocket.receive_text()
                async with httpx.AsyncClient(timeout=settings.federation_timeout, verify=not settings.skip_ssl_verify) as client:
                    response = await client.post(f"http://localhost:{settings.port}/rpc", json=json.loads(data), headers={"Content-Type": "application/json"})
                    await websocket.send_text(response.text)
            except JSONRPCError as e: # This won't be caught if error is from client.post
                await websocket.send_text(json.dumps(e.to_dict()))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}))
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client.")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}", exc_info=True)
                # Attempt to send a generic error before closing, if possible
                try:
                    await websocket.send_text(json.dumps({"jsonrpc": "2.0", "error": {"code": -32000, "message": "Internal Server Error"}, "id": None}))
                except Exception: # If sending also fails
                    pass
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                break
    except Exception as e:
        logger.error(f"WebSocket connection error: {str(e)}", exc_info=True)
        # Ensure socket is closed if accept() fails or other initial error
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass

@utility_router.get("/sse")
async def utility_sse_endpoint(request: Request, user: str = Depends(require_auth)):
    try:
        logger.debug("User %s requested SSE connection", user)
        base_url = str(request.base_url).rstrip("/")
        transport = SSETransport(base_url=base_url)
        await transport.connect()
        await session_registry.add_session(transport.session_id, transport)
        # For utility SSE, server_id is None
        asyncio.create_task(session_registry.respond(None, user, session_id=transport.session_id, base_url=base_url))
        response = await transport.create_sse_response(request)
        tasks = BackgroundTasks()
        tasks.add_task(session_registry.remove_session, transport.session_id)
        response.background = tasks
        logger.info("SSE connection established: %s", transport.session_id)
        return response
    except Exception as e:
        logger.error("SSE connection error: %s", e)
        raise HTTPException(status_code=500, detail="SSE connection failed")

@utility_router.post("/message")
async def utility_message_endpoint(request: Request, user: str = Depends(require_auth)):
    try:
        logger.debug("User %s sent a message to SSE session", user)
        session_id = request.query_params.get("session_id")
        if not session_id:
            logger.error("Missing session_id in message request")
            raise HTTPException(status_code=400, detail="Missing session_id")
        message = await request.json()
        await session_registry.broadcast(session_id=session_id, message=message)
        return JSONResponse(content={"status": "success"}, status_code=202)
    except ValueError as e:
        logger.error("Invalid message format: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Message handling error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process message")

@utility_router.post("/logging/setLevel")
async def set_log_level(request: Request, user: str = Depends(require_auth)) -> None:
    logger.debug(f"User {user} requested to set log level")
    body = await request.json()
    level_str = body.get("level", "info") # Default to info if not provided
    try:
        level = LogLevel(level_str.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid log level: {level_str}")
    await logging_service.set_level(level)
    return None # Explicit None for 204 No Content like behavior if appropriate, or return JSONResponse

@metrics_router.get("", response_model=Dict[str, Any]) # More generic return for now
async def get_metrics(db: Session = Depends(get_db), user: str = Depends(require_auth)) -> Dict[str, Any]:
    logger.debug(f"User {user} requested aggregated metrics")
    tool_metrics = await tool_service.aggregate_metrics(db)
    resource_metrics = await resource_service.aggregate_metrics(db)
    server_metrics = await server_service.aggregate_metrics(db)
    prompt_metrics = await prompt_service.aggregate_metrics(db)
    return {
        "tools": tool_metrics.model_dump(by_alias=True), # Use model_dump
        "resources": resource_metrics.model_dump(by_alias=True),
        "servers": server_metrics.model_dump(by_alias=True),
        "prompts": prompt_metrics.model_dump(by_alias=True),
    }

@metrics_router.post("/reset", response_model=Dict[str, str])
async def reset_metrics(entity: Optional[str] = None, entity_id: Optional[int] = None, db: Session = Depends(get_db), user: str = Depends(require_auth)) -> Dict[str, str]:
    logger.debug(f"User {user} requested metrics reset for entity: {entity}, id: {entity_id}")
    if entity is None:
        await tool_service.reset_metrics(db)
        await resource_service.reset_metrics(db)
        await server_service.reset_metrics(db)
        await prompt_service.reset_metrics(db)
    elif entity.lower() == "tool":
        await tool_service.reset_metrics(db, entity_id)
    elif entity.lower() == "resource": # Resource reset is global, entity_id not used by service
        await resource_service.reset_metrics(db)
    elif entity.lower() == "server": # Server reset is global
        await server_service.reset_metrics(db)
    elif entity.lower() == "prompt": # Prompt reset is global
        await prompt_service.reset_metrics(db)
    else:
        raise HTTPException(status_code=400, detail="Invalid entity type for metrics reset")
    return {"status": "success", "message": f"Metrics reset for {entity if entity else 'all entities'}"}

@app.get("/health")
async def healthcheck(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        error_message = f"Database connection error: {str(e)}"
        logger.error(error_message)
        return {"status": "unhealthy", "error": error_message}
    return {"status": "healthy"}

@app.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    try:
        await asyncio.to_thread(db.execute, text("SELECT 1"))
        return JSONResponse(content={"status": "ready"}, status_code=200)
    except Exception as e:
        error_message = f"Readiness check failed: {str(e)}"
        logger.error(error_message)
        return JSONResponse(content={"status": "not ready", "error": error_message}, status_code=503)

app.include_router(version_router)
app.include_router(protocol_router)
app.include_router(tool_router)
app.include_router(resource_router)
app.include_router(prompt_router)
app.include_router(gateway_router)
app.include_router(root_router_obj) # Use aliased router
app.include_router(utility_router)
app.include_router(server_router)
app.include_router(metrics_router)

UI_ENABLED = settings.mcpgateway_ui_enabled
ADMIN_API_ENABLED = settings.mcpgateway_admin_api_enabled
logger.info(f"Admin UI enabled: {UI_ENABLED}")
logger.info(f"Admin API enabled: {ADMIN_API_ENABLED}")

if ADMIN_API_ENABLED:
    logger.info("Including admin_router - Admin API enabled")
    app.include_router(admin_router)
else:
    logger.warning("Admin API routes not mounted - Admin API disabled via MCPGATEWAY_ADMIN_API_ENABLED=False")

app.mount("/mcp", app=streamable_http_session.handle_streamable_http)

if UI_ENABLED:
    logger.info("Mounting static files - UI enabled")
    try:
        app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
        logger.info("Static assets served from %s", settings.static_dir)
    except RuntimeError as exc:
        logger.warning("Static dir %s not found – Admin UI disabled (%s)", settings.static_dir, exc)

    @app.get("/")
    async def root_redirect_to_admin(request: Request): # Renamed to avoid conflict
        logger.debug("Redirecting root path to /admin")
        root_path = request.scope.get("root_path", "")
        return RedirectResponse(f"{root_path}/admin", status_code=303)
else:
    logger.warning("Static files not mounted - UI disabled via MCPGATEWAY_UI_ENABLED=False")
    @app.get("/")
    async def root_info_disabled_ui(): # Renamed
        logger.info("UI disabled, serving API info at root path")
        return {"name": settings.app_name, "version": "1.0.0", "description": f"{settings.app_name} API - UI is disabled", "ui_enabled": False, "admin_api_enabled": ADMIN_API_ENABLED}

app.post("/initialize")(initialize)
app.post("/notifications")(handle_notification)
