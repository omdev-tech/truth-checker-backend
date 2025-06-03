"""Test configuration and common fixtures."""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient

from truth_checker.infrastructure.mcp.factory import MCPProviderFactory
from truth_checker.infrastructure.mcp.wikipedia_adapter import WikipediaMCPAdapter


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create and provide an event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def mock_http_client() -> AsyncClient:
    """Provide a mock HTTP client for testing."""
    async with AsyncClient() as client:
        yield client


@pytest_asyncio.fixture
async def wikipedia_adapter(mock_http_client: AsyncClient) -> WikipediaMCPAdapter:
    """Provide a Wikipedia adapter instance for testing."""
    adapter = WikipediaMCPAdapter(
        base_url="http://test-mcp-server",
        timeout=1.0,
        cache_ttl=60,
        cache_maxsize=100,
    )
    adapter._client = mock_http_client
    yield adapter
    await adapter.shutdown()


@pytest.fixture
def mcp_factory() -> MCPProviderFactory:
    """Provide a factory instance for testing."""
    factory = MCPProviderFactory()
    factory.register_provider("wikipedia", WikipediaMCPAdapter)
    return factory 