"""Tests for the AI provider factory."""

import pytest
import pytest_asyncio
from httpx import AsyncClient

from truth_checker.domain.ports.ai_provider import AIProvider
from truth_checker.infrastructure.ai.chatgpt_adapter import ChatGPTAdapter
from truth_checker.infrastructure.ai.factory import AIProviderFactory


class TestProvider(AIProvider):
    """Test provider implementation."""

    def __init__(self, config: dict = None, provider_name: str = "Test"):
        """Initialize test provider."""
        self._name = provider_name
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the provider."""
        self._initialized = True

    async def analyze_statement(self, text: str):
        """Analyze a statement."""
        pass

    async def verify_facts(self, facts, evidence):
        """Verify facts."""
        pass

    async def generate_explanation(self, verification_result):
        """Generate explanation."""
        pass

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._initialized = False

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self._name

    @property
    def capabilities(self) -> dict:
        """Get capabilities."""
        return {}

    @property
    def is_available(self) -> bool:
        """Check if available."""
        return self._initialized


@pytest_asyncio.fixture
async def mock_http_client(monkeypatch) -> AsyncClient:
    """Create a mock HTTP client."""
    client = AsyncClient()
    return client


@pytest_asyncio.fixture
async def ai_factory() -> AIProviderFactory:
    """Create an AI provider factory."""
    return AIProviderFactory()


@pytest.mark.asyncio
async def test_create_provider(ai_factory: AIProviderFactory):
    """Test provider creation."""
    # Register test provider
    ai_factory.register_provider_type("test", TestProvider)

    # Create test provider
    provider = await ai_factory.create_provider(
        "test",
        config={"key": "value"},
        provider_name="TestAI",
    )
    assert isinstance(provider, TestProvider)
    assert provider.provider_name == "TestAI"
    assert provider.is_available


@pytest.mark.asyncio
async def test_get_provider(ai_factory: AIProviderFactory):
    """Test provider retrieval."""
    # Register and create test provider
    ai_factory.register_provider_type("test", TestProvider)
    provider = await ai_factory.create_provider(
        "test",
        provider_name="TestAI",
    )
    
    # Get the provider
    retrieved = ai_factory.get_provider("TestAI")
    assert retrieved is provider


@pytest.mark.asyncio
async def test_list_providers(ai_factory: AIProviderFactory):
    """Test provider listing."""
    # Register test provider
    ai_factory.register_provider_type("test", TestProvider)

    # Create providers
    await ai_factory.create_provider(
        "test",
        provider_name="TestAI1",
    )
    await ai_factory.create_provider(
        "test",
        provider_name="TestAI2",
    )
    
    # List providers
    providers = ai_factory.list_providers()
    assert len(providers) == 2
    assert "TestAI1" in providers
    assert "TestAI2" in providers


@pytest.mark.asyncio
async def test_register_provider_type(ai_factory: AIProviderFactory):
    """Test provider type registration."""
    # Register the provider type
    ai_factory.register_provider_type("test", TestProvider)
    
    # Create a provider of the new type
    provider = await ai_factory.create_provider("test")
    assert isinstance(provider, TestProvider)


@pytest.mark.asyncio
async def test_remove_provider(ai_factory: AIProviderFactory):
    """Test provider removal."""
    # Register and create test provider
    ai_factory.register_provider_type("test", TestProvider)
    provider = await ai_factory.create_provider(
        "test",
        provider_name="TestAI",
    )
    
    # Remove the provider
    await ai_factory.remove_provider("TestAI")
    assert ai_factory.get_provider("TestAI") is None


@pytest.mark.asyncio
async def test_shutdown(ai_factory: AIProviderFactory):
    """Test factory shutdown."""
    # Register test provider
    ai_factory.register_provider_type("test", TestProvider)

    # Create providers
    await ai_factory.create_provider(
        "test",
        provider_name="TestAI1",
    )
    await ai_factory.create_provider(
        "test",
        provider_name="TestAI2",
    )
    
    # Shutdown factory
    await ai_factory.shutdown()
    assert len(ai_factory.list_providers()) == 0


@pytest.mark.asyncio
async def test_unknown_provider_type(ai_factory: AIProviderFactory):
    """Test error handling for unknown provider type."""
    with pytest.raises(ValueError):
        await ai_factory.create_provider("unknown") 