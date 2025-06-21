# -*- coding: utf-8 -*-
"""

Copyright 2025
SPDX-License-Identifier: Apache-2.0
Authors: Mihai Criveti

Tests for server service implementation.
"""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from mcpgateway.db import Prompt as DbPrompt
from mcpgateway.db import Resource as DbResource
from mcpgateway.db import Server as DbServer
from mcpgateway.db import Tool as DbTool
from mcpgateway.schemas import ServerCreate, ServerRead, ServerUpdate
from mcpgateway.services.server_service import (
    ServerError,
    ServerNameConflictError,
    ServerNotFoundError,
    ServerService,
)


@pytest.fixture
def server_service() -> ServerService:
    """Create a server service instance."""
    return ServerService()


@pytest.fixture
def mock_server() -> MagicMock:
    """Create a mock server model."""
    server = MagicMock(spec=DbServer)
    server.id = 1
    server.name = "test_server"
    server.description = "A test server"
    server.icon = "server-icon"
    server.created_at = "2023-01-01T00:00:00"
    server.updated_at = "2023-01-01T00:00:00"
    server.is_active = True

    # Set up associated items
    tool1 = MagicMock(spec=DbTool)
    tool1.id = 101

    resource1 = MagicMock(spec=DbResource)
    resource1.id = 201

    prompt1 = MagicMock(spec=DbPrompt)
    prompt1.id = 301

    server.tools = [tool1]
    server.resources = [resource1]
    server.prompts = [prompt1]

    # Set up metrics
    server.metrics = []
    server.execution_count = 0
    server.successful_executions = 0
    server.failed_executions = 0
    server.failure_rate = 0.0
    server.min_response_time = None
    server.max_response_time = None
    server.avg_response_time = None
    server.last_execution_time = None

    return server


@pytest.fixture
def mock_tool() -> MagicMock:
    """Create a mock tool."""
    tool = MagicMock(spec=DbTool)
    tool.id = 101
    tool.name = "test_tool"
    return tool


@pytest.fixture
def mock_resource() -> MagicMock:
    """Create a mock resource."""
    resource = MagicMock(spec=DbResource)
    resource.id = 201
    resource.name = "test_resource"
    return resource


@pytest.fixture
def mock_prompt() -> MagicMock:
    """Create a mock prompt."""
    prompt = MagicMock(spec=DbPrompt)
    prompt.id = 301
    prompt.name = "test_prompt"
    return prompt


class TestServerService:
    """Tests for the ServerService class."""

    @pytest.mark.asyncio
    async def test_register_server(
        self,
        server_service: ServerService,
        test_db: MagicMock,
        mock_tool: MagicMock,
        mock_resource: MagicMock,
        mock_prompt: MagicMock,
    ) -> None:
        """Test successful server registration."""
        # Set up DB behavior
        mock_scalar = Mock()
        mock_scalar.scalar_one_or_none.return_value = None
        test_db.execute = Mock(return_value=mock_scalar)
        test_db.add = Mock()
        test_db.commit = Mock()
        test_db.refresh = Mock()

        # Set up DB.get to return associated objects
        test_db.get = Mock(
            side_effect=lambda cls, id: {
                (DbTool, 101): mock_tool,
                (DbResource, 201): mock_resource,
                (DbPrompt, 301): mock_prompt,
            }.get((cls, id))
        )

        # Set up service methods
        server_service._notify_server_added = AsyncMock() # type: ignore[method-assign]
        server_service._convert_server_to_read = Mock( # type: ignore[method-assign]
            return_value=ServerRead(
                id=1,
                name="test_server",
                description="A test server",
                icon="server-icon",
                created_at="2023-01-01T00:00:00",
                updated_at="2023-01-01T00:00:00",
                is_active=True,
                associated_tools=[101],
                associated_resources=[201],
                associated_prompts=[301],
                metrics={
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "failure_rate": 0.0,
                    "min_response_time": None,
                    "max_response_time": None,
                    "avg_response_time": None,
                    "last_execution_time": None,
                },
            )
        )

        # Create server request
        server_create = ServerCreate(name="test_server", description="A test server", icon="server-icon", associated_tools=["101"], associated_resources=["201"], associated_prompts=["301"])

        # Call method
        result = await server_service.register_server(test_db, server_create)

        # Verify DB operations
        test_db.add.assert_called_once()
        test_db.commit.assert_called_once()
        test_db.refresh.assert_called_once()

        # Verify notification
        server_service._notify_server_added.assert_called_once()

        # Verify result
        assert result.name == "test_server"
        assert result.description == "A test server"
        assert result.icon == "server-icon"
        assert result.is_active is True
        assert 101 in result.associated_tools
        assert 201 in result.associated_resources
        assert 301 in result.associated_prompts

    @pytest.mark.asyncio
    async def test_register_server_name_conflict(
        self, server_service: ServerService, mock_server: MagicMock, test_db: MagicMock
    ) -> None:
        """Test server registration with name conflict."""
        # Mock DB to return existing server
        mock_scalar = Mock()
        mock_scalar.scalar_one_or_none.return_value = mock_server
        test_db.execute = Mock(return_value=mock_scalar)

        # Create server request with conflicting name
        server_create = ServerCreate(name="test_server", description="A new server", icon="new-icon")  # Same name as mock_server

        # Should raise conflict error
        with pytest.raises(ServerNameConflictError) as exc_info:
            await server_service.register_server(test_db, server_create)

        assert "Server already exists with name" in str(exc_info.value)
        assert exc_info.value.name == "test_server"
        assert exc_info.value.is_active == mock_server.is_active
        assert exc_info.value.server_id == mock_server.id

    @pytest.mark.asyncio
    async def test_register_server_invalid_associated_tool(
        self, server_service: ServerService, test_db: MagicMock
    ) -> None:
        """Test server registration with non-existent associated tool."""
        # Set up DB behavior
        mock_scalar = Mock()
        mock_scalar.scalar_one_or_none.return_value = None
        test_db.execute = Mock(return_value=mock_scalar)

        # Set up DB.get to return None for tool (not found)
        test_db.get = Mock(return_value=None)
        test_db.rollback = Mock()

        # Create server request with non-existent tool
        server_create = ServerCreate(name="test_server", description="A test server", associated_tools=["999"])  # Non-existent tool ID

        # Should raise error about non-existent tool
        with pytest.raises(ServerError) as exc_info:
            await server_service.register_server(test_db, server_create)

        assert "Tool with id 999 does not exist" in str(exc_info.value)

        # Verify rollback
        test_db.rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_servers(
        self, server_service: ServerService, mock_server: MagicMock, test_db: MagicMock
    ) -> None:
        """Test listing servers."""
        # Mock DB to return a list of servers
        mock_scalar_result = MagicMock()
        mock_scalar_result.all.return_value = [mock_server]
        mock_execute = Mock(return_value=mock_scalar_result)
        test_db.execute = mock_execute

        # Set up conversion
        server_read = ServerRead(
            id=1,
            name="test_server",
            description="A test server",
            icon="server-icon",
            created_at="2023-01-01T00:00:00",
            updated_at="2023-01-01T00:00:00",
            is_active=True,
            associated_tools=[101],
            associated_resources=[201],
            associated_prompts=[301],
            metrics={
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "failure_rate": 0.0,
                "min_response_time": None,
                "max_response_time": None,
                "avg_response_time": None,
                "last_execution_time": None,
            },
        )
        server_service._convert_server_to_read = Mock(return_value=server_read) # type: ignore[method-assign]

        # Call method
        result = await server_service.list_servers(test_db)

        # Verify DB query
        test_db.execute.assert_called_once()

        # Verify result
        assert len(result) == 1
        assert result[0] == server_read
        server_service._convert_server_to_read.assert_called_once_with(mock_server)

    @pytest.mark.asyncio
    async def test_get_server(
        self, server_service: ServerService, mock_server: MagicMock, test_db: MagicMock
    ) -> None:
        """Test getting a server by ID."""
        # Mock DB get to return server
        test_db.get = Mock(return_value=mock_server)

        # Set up conversion
        server_read = ServerRead(
            id=1,
            name="test_server",
            description="A test server",
            icon="server-icon",
            created_at="2023-01-01T00:00:00",
            updated_at="2023-01-01T00:00:00",
            is_active=True,
            associated_tools=[101],
            associated_resources=[201],
            associated_prompts=[301],
            metrics={
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "failure_rate": 0.0,
                "min_response_time": None,
                "max_response_time": None,
                "avg_response_time": None,
                "last_execution_time": None,
            },
        )
        server_service._convert_server_to_read = Mock(return_value=server_read) # type: ignore[method-assign]

        # Call method
        result = await server_service.get_server(test_db, 1)

        # Verify DB query
        test_db.get.assert_called_once_with(DbServer, 1)

        # Verify result
        assert result == server_read
        server_service._convert_server_to_read.assert_called_once_with(mock_server)

    @pytest.mark.asyncio
    async def test_get_server_not_found(self, server_service: ServerService, test_db: MagicMock) -> None:
        """Test getting a non-existent server."""
        # Mock DB get to return None
        test_db.get = Mock(return_value=None)

        # Should raise NotFoundError
        with pytest.raises(ServerNotFoundError) as exc_info:
            await server_service.get_server(test_db, 999)

        assert "Server not found: 999" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_server(
        self,
        server_service: ServerService,
        mock_server: MagicMock,
        test_db: MagicMock,
        mock_tool: MagicMock,
        mock_resource: MagicMock,
        mock_prompt: MagicMock,
    ) -> None:
        """Test updating a server."""
        # Mock DB get to return server
        test_db.get = Mock(
            side_effect=lambda cls, id: (
                mock_server
                if cls == DbServer and id == 1
                else {
                    (DbTool, 102): mock_tool,
                    (DbResource, 202): mock_resource,
                    (DbPrompt, 302): mock_prompt,
                }.get((cls, id))
            )
        )

        # Mock DB to check for name conflicts
        mock_scalar = Mock()
        mock_scalar.scalar_one_or_none.return_value = None
        test_db.execute = Mock(return_value=mock_scalar)

        test_db.commit = Mock()
        test_db.refresh = Mock()

        # Set up service methods
        server_service._notify_server_updated = AsyncMock() # type: ignore[method-assign]
        server_service._convert_server_to_read = Mock( # type: ignore[method-assign]
            return_value=ServerRead(
                id=1,
                name="updated_server",
                description="An updated server",
                icon="updated-icon",
                created_at="2023-01-01T00:00:00",
                updated_at="2023-01-01T00:00:00",
                is_active=True,
                associated_tools=[102],
                associated_resources=[202],
                associated_prompts=[302],
                metrics={
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "failure_rate": 0.0,
                    "min_response_time": None,
                    "max_response_time": None,
                    "avg_response_time": None,
                    "last_execution_time": None,
                },
            )
        )

        # Create update request
        server_update = ServerUpdate(name="updated_server", description="An updated server", icon="updated-icon", associated_tools=["102"], associated_resources=["202"], associated_prompts=["302"])

        # Call method
        result = await server_service.update_server(test_db, 1, server_update)

        # Verify DB operations
        test_db.commit.assert_called_once()
        test_db.refresh.assert_called_once()

        # Verify server properties were updated
        assert mock_server.name == "updated_server"
        assert mock_server.description == "An updated server"
        assert mock_server.icon == "updated-icon"

        # Verify notification
        server_service._notify_server_updated.assert_called_once()

        # Verify result
        assert result.name == "updated_server"
        assert result.description == "An updated server"
        assert result.icon == "updated-icon"
        assert 102 in result.associated_tools
        assert 202 in result.associated_resources
        assert 302 in result.associated_prompts

    @pytest.mark.asyncio
    async def test_update_server_not_found(self, server_service: ServerService, test_db: MagicMock) -> None:
        """Test updating a non-existent server."""
        # Mock DB get to return None
        test_db.get = Mock(return_value=None)

        # Create update request
        server_update = ServerUpdate(name="updated_server", description="An updated server")

        # Should raise NotFoundError
        with pytest.raises(ServerNotFoundError) as exc_info:
            await server_service.update_server(test_db, 999, server_update)

        assert "Server not found: 999" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_server_name_conflict(
        self, server_service: ServerService, mock_server: MagicMock, test_db: MagicMock
    ) -> None:
        """Test updating a server with a name that conflicts with another server."""
        # Create a second server (the one being updated)
        server1 = mock_server

        # Create a conflicting server
        server2 = MagicMock(spec=DbServer)
        server2.id = 2
        server2.name = "existing_server"
        server2.is_active = True

        # Mock DB get to return server1
        test_db.get = Mock(return_value=server1)

        # Mock DB to check for name conflicts and return server2
        mock_scalar = Mock()
        mock_scalar.scalar_one_or_none.return_value = server2
        test_db.execute = Mock(return_value=mock_scalar)

        test_db.rollback = Mock()

        # Create update request with conflicting name
        server_update = ServerUpdate(
            name="existing_server",  # Name that conflicts with server2
        )

        # Should raise conflict error
        with pytest.raises(ServerNameConflictError) as exc_info:
            await server_service.update_server(test_db, 1, server_update)

        assert "Server already exists with name" in str(exc_info.value)
        assert exc_info.value.name == "existing_server"
        assert exc_info.value.is_active == server2.is_active
        assert exc_info.value.server_id == server2.id

    @pytest.mark.asyncio
    async def test_toggle_server_status(
        self, server_service: ServerService, mock_server: MagicMock, test_db: MagicMock
    ) -> None:
        """Test toggling server active status."""
        # Mock DB get to return server
        test_db.get = Mock(return_value=mock_server)
        test_db.commit = Mock()
        test_db.refresh = Mock()

        # Set up service methods
        server_service._notify_server_activated = AsyncMock() # type: ignore[method-assign]
        server_service._notify_server_deactivated = AsyncMock() # type: ignore[method-assign]
        server_service._convert_server_to_read = Mock( # type: ignore[method-assign]
            return_value=ServerRead(
                id=1,
                name="test_server",
                description="A test server",
                icon="server-icon",
                created_at="2023-01-01T00:00:00",
                updated_at="2023-01-01T00:00:00",
                is_active=False,  # Deactivated
                associated_tools=[101],
                associated_resources=[201],
                associated_prompts=[301],
                metrics={
                    "total_executions": 0,
                    "successful_executions": 0,
                    "failed_executions": 0,
                    "failure_rate": 0.0,
                    "min_response_time": None,
                    "max_response_time": None,
                    "avg_response_time": None,
                    "last_execution_time": None,
                },
            )
        )

        # Deactivate the server (it's active by default)
        result = await server_service.toggle_server_status(test_db, 1, activate=False)

        # Verify DB operations
        test_db.get.assert_called_once_with(DbServer, 1)
        test_db.commit.assert_called_once()
        test_db.refresh.assert_called_once()

        # Verify properties were updated
        assert mock_server.is_active is False

        # Verify notification
        server_service._notify_server_deactivated.assert_called_once()
        server_service._notify_server_activated.assert_not_called()

        # Verify result
        assert result.is_active is False

    @pytest.mark.asyncio
    async def test_delete_server(
        self, server_service: ServerService, mock_server: MagicMock, test_db: MagicMock
    ) -> None:
        """Test deleting a server."""
        # Mock DB get to return server
        test_db.get = Mock(return_value=mock_server)
        test_db.delete = Mock()
        test_db.commit = Mock()

        # Set up service methods
        server_service._notify_server_deleted = AsyncMock() # type: ignore[method-assign]

        # Call method
        await server_service.delete_server(test_db, 1)

        # Verify DB operations
        test_db.get.assert_called_once_with(DbServer, 1)
        test_db.delete.assert_called_once_with(mock_server)
        test_db.commit.assert_called_once()

        # Verify notification
        server_service._notify_server_deleted.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_server_not_found(self, server_service: ServerService, test_db: MagicMock) -> None:
        """Test deleting a non-existent server."""
        # Mock DB get to return None
        test_db.get = Mock(return_value=None)

        # Should raise NotFoundError
        with pytest.raises(ServerNotFoundError) as exc_info:
            await server_service.delete_server(test_db, 999)

        assert "Server not found: 999" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_reset_metrics(self, server_service: ServerService, test_db: MagicMock) -> None:
        """Test resetting metrics."""
        # Mock DB operations
        test_db.execute = Mock()
        test_db.commit = Mock()

        # Call method
        await server_service.reset_metrics(test_db)

        # Verify DB operations
        test_db.execute.assert_called_once()
        test_db.commit.assert_called_once()
