"""Tests for the MCP provider factory."""

import pytest
import pytest_asyncio
from httpx import AsyncClient

from truth_checker.domain.ports.mcp_provider import MCPProvider, MCPSearchResult, MCPValidationResult
from truth_checker.infrastructure.mcp.factory import MCPProviderFactory
from truth_checker.infrastructure.mcp.wikipedia_adapter import WikipediaMCPAdapter


class TestMCPProvider(MCPProvider):
    """Test MCP provider implementation."""

    def __init__(
        self,
        base_url: str = "http://test.com",
        timeout: float = 10.0,
        provider_name: str = "Test",
    ):
        """Initialize test provider."""
        self._name = provider_name
        self._initialized = False
        self._language = "en"

    async def initialize(self) -> None:
        """Initialize the provider."""
        self._initialized = True

    async def search(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.5,
    ) -> list[MCPSearchResult]:
        """Search for articles."""
        return [
            MCPSearchResult(
                title="Test Result",
                summary="Test summary",
                url="http://test.com",
                relevance_score=0.9,
            )
        ]

    async def get_article_summary(self, title: str) -> str:
        """Get article summary."""
        return "Test summary"

    async def get_article_content(self, title: str) -> dict:
        """Get article content."""
        return {"text": "Test content"}

    async def validate_fact(self, statement: str) -> MCPValidationResult:
        """Validate a fact."""
        return MCPValidationResult(
            is_valid=True,
            confidence=0.9,
            supporting_evidence=["Test evidence"],
            source_urls=["http://test.com"],
        )

    async def set_language(self, language_code: str) -> None:
        """Set language."""
        self._language = language_code

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._initialized = False

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self._name

    @property
    def supported_languages(self) -> list[str]:
        """Get supported languages."""
        return ["en"]

    @property
    def is_available(self) -> bool:
        """Check if available."""
        return self._initialized

    @property
    def capabilities(self) -> dict[str, bool]:
        """Get capabilities."""
        return {
            "search": True,
            "validation": True,
        }


@pytest_asyncio.fixture
async def mock_http_client(monkeypatch) -> AsyncClient:
    """Create a mock HTTP client."""
    client = AsyncClient()
    return client


@pytest_asyncio.fixture
async def mcp_factory() -> MCPProviderFactory:
    """Create an MCP provider factory."""
    return MCPProviderFactory()


@pytest.mark.asyncio
async def test_register_provider(mcp_factory: MCPProviderFactory):
    """Test provider registration."""
    # Register test provider
    mcp_factory.register_provider("test", TestMCPProvider)
    
    # Create provider instance
    provider = await mcp_factory.create_provider("test")
    assert isinstance(provider, TestMCPProvider)
    assert provider.provider_name == "Test"
    assert provider.is_available


@pytest.mark.asyncio
async def test_register_duplicate_provider(mcp_factory: MCPProviderFactory):
    """Test error on duplicate provider registration."""
    mcp_factory.register_provider("test", TestMCPProvider)
    
    with pytest.raises(ValueError):
        mcp_factory.register_provider("test", TestMCPProvider)


@pytest.mark.asyncio
async def test_create_unknown_provider(mcp_factory: MCPProviderFactory):
    """Test error when creating unknown provider."""
    with pytest.raises(ValueError):
        await mcp_factory.create_provider("unknown")


@pytest.mark.asyncio
async def test_get_provider(mcp_factory: MCPProviderFactory):
    """Test provider retrieval."""
    # Register and create provider
    mcp_factory.register_provider("test", TestMCPProvider)
    provider = await mcp_factory.create_provider("test")
    
    # Get provider
    retrieved = mcp_factory.get_provider("test")
    assert retrieved is provider
    assert retrieved.is_available


@pytest.mark.asyncio
async def test_get_nonexistent_provider(mcp_factory: MCPProviderFactory):
    """Test retrieval of non-existent provider."""
    assert mcp_factory.get_provider("unknown") is None


@pytest.mark.asyncio
async def test_shutdown_provider(mcp_factory: MCPProviderFactory):
    """Test provider shutdown."""
    # Register and create provider
    mcp_factory.register_provider("test", TestMCPProvider)
    provider = await mcp_factory.create_provider("test")
    assert provider.is_available
    
    # Shutdown provider
    await mcp_factory.shutdown_provider("test")
    assert not provider.is_available
    assert mcp_factory.get_provider("test") is None


@pytest.mark.asyncio
async def test_shutdown_all_providers(mcp_factory: MCPProviderFactory):
    """Test shutdown of all providers."""
    # Register and create multiple providers
    mcp_factory.register_provider("test1", TestMCPProvider)
    mcp_factory.register_provider("test2", TestMCPProvider)
    
    provider1 = await mcp_factory.create_provider("test1")
    provider2 = await mcp_factory.create_provider("test2")
    
    assert provider1.is_available
    assert provider2.is_available
    
    # Shutdown all providers
    await mcp_factory.shutdown_all()
    
    assert not provider1.is_available
    assert not provider2.is_available
    assert not mcp_factory._active_providers


@pytest.mark.asyncio
async def test_available_providers(mcp_factory: MCPProviderFactory):
    """Test available providers property."""
    # Register providers
    mcp_factory.register_provider("test1", TestMCPProvider)
    mcp_factory.register_provider("test2", TestMCPProvider)
    
    # Create only one provider
    await mcp_factory.create_provider("test1")
    
    # Check availability
    available = mcp_factory.available_providers
    assert available["test1"] is True
    assert available["test2"] is False


@pytest.mark.asyncio
async def test_provider_initialization_failure(mcp_factory: MCPProviderFactory):
    """Test handling of provider initialization failure."""
    class FailingProvider(TestMCPProvider):
        async def initialize(self) -> None:
            raise RuntimeError("Initialization failed")

    mcp_factory.register_provider("failing", FailingProvider)
    
    with pytest.raises(RuntimeError):
        await mcp_factory.create_provider("failing")


@pytest.mark.asyncio
async def test_wikipedia_provider_registration():
    """Test that Wikipedia provider is registered by default."""
    factory = MCPProviderFactory()
    
    # Create Wikipedia provider
    provider = await factory.create_provider("wikipedia")
    assert isinstance(provider, WikipediaMCPAdapter)
    assert provider.provider_name == "Wikipedia"
    
    # Clean up
    await factory.shutdown_all() 