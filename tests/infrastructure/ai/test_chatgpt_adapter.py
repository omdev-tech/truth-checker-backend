"""Tests for the ChatGPT adapter."""

import json
from typing import Dict, List

import pytest
import pytest_asyncio
from httpx import AsyncClient, Response

from truth_checker.domain.ports.ai_provider import AIAnalysisResult, AIVerificationResult
from truth_checker.infrastructure.ai.chatgpt_adapter import ChatGPTAdapter, ChatGPTConfig


class MockResponse:
    """Mock response for testing."""

    def __init__(self, status_code: int, json_data: dict = None):
        """Initialize mock response."""
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self) -> dict:
        """Get JSON data."""
        return self._json_data

    def raise_for_status(self) -> None:
        """Check response status."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@pytest_asyncio.fixture
async def mock_http_client(monkeypatch) -> AsyncClient:
    """Create a mock HTTP client."""
    client = AsyncClient()

    # Mock successful response
    success_response = MockResponse(
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "is_factual_claim": True,
                            "confidence": 0.9,
                            "extracted_facts": ["Test fact"],
                            "verification_results": [{"fact": "Test fact", "verified": True}],
                            "overall_confidence": 0.9,
                            "explanation": "Test explanation",
                            "sources": ["Test source"],
                        })
                    }
                }
            ]
        },
    )

    # Mock the post method
    async def mock_post(*args, **kwargs) -> MockResponse:
        # Parse the request to determine the response
        if "analyze" in kwargs.get("json", {}).get("messages", [{}])[0].get("content", ""):
            # Analysis response
            return MockResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({
                                    "is_factual_claim": True,
                                    "confidence": 0.9,
                                    "extracted_facts": ["Test fact"],
                                })
                            }
                        }
                    ]
                },
            )
        elif "verify" in kwargs.get("json", {}).get("messages", [{}])[0].get("content", ""):
            # Verification response
            return MockResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({
                                    "verification_results": [{"fact": "Test fact", "verified": True}],
                                    "overall_confidence": 0.9,
                                    "explanation": "Test explanation",
                                    "sources": ["Test source"],
                                })
                            }
                        }
                    ]
                },
            )
        else:
            # Default test response
            return success_response

    client.post = mock_post
    return client


@pytest_asyncio.fixture
async def chatgpt_adapter(mock_http_client: AsyncClient) -> ChatGPTAdapter:
    """Create a ChatGPT adapter with mock client."""
    config = ChatGPTConfig(api_key="test_key")
    adapter = ChatGPTAdapter(config=config)
    adapter._client = mock_http_client
    return adapter


@pytest.mark.asyncio
async def test_initialize_success(chatgpt_adapter: ChatGPTAdapter):
    """Test successful initialization."""
    await chatgpt_adapter.initialize()
    assert chatgpt_adapter.is_available


@pytest.mark.asyncio
async def test_analyze_statement(chatgpt_adapter: ChatGPTAdapter):
    """Test statement analysis."""
    await chatgpt_adapter.initialize()
    
    result = await chatgpt_adapter.analyze_statement("Test statement")
    assert isinstance(result, AIAnalysisResult)
    assert result.is_factual_claim
    assert result.confidence > 0
    assert len(result.extracted_facts) > 0


@pytest.mark.asyncio
async def test_verify_facts(chatgpt_adapter: ChatGPTAdapter):
    """Test fact verification."""
    await chatgpt_adapter.initialize()
    
    result = await chatgpt_adapter.verify_facts(
        facts=["Test fact"],
        evidence=["Test evidence"],
    )
    assert isinstance(result, AIVerificationResult)
    assert result.confidence > 0
    assert len(result.verification_results) > 0
    assert result.explanation


@pytest.mark.asyncio
async def test_generate_explanation(chatgpt_adapter: ChatGPTAdapter):
    """Test explanation generation."""
    await chatgpt_adapter.initialize()
    
    verification = AIVerificationResult(
        facts=["Test fact"],
        verification_results=[{"fact": "Test fact", "verified": True}],
        confidence=0.9,
        explanation="Original explanation",
        source_references=["Test source"],
    )
    
    explanation = await chatgpt_adapter.generate_explanation(verification)
    assert isinstance(explanation, str)
    assert len(explanation) > 0


@pytest.mark.asyncio
async def test_provider_properties(chatgpt_adapter: ChatGPTAdapter):
    """Test provider properties."""
    assert chatgpt_adapter.provider_name == "ChatGPT"
    assert isinstance(chatgpt_adapter.capabilities, dict)
    assert "fact_extraction" in chatgpt_adapter.capabilities


@pytest.mark.asyncio
async def test_shutdown(chatgpt_adapter: ChatGPTAdapter):
    """Test provider shutdown."""
    await chatgpt_adapter.initialize()
    assert chatgpt_adapter.is_available
    
    await chatgpt_adapter.shutdown()
    assert not chatgpt_adapter.is_available


@pytest.mark.asyncio
async def test_initialization_failure(mock_http_client: AsyncClient):
    """Test initialization failure."""
    # Create adapter with invalid config
    config = ChatGPTConfig(api_key="invalid_key")
    adapter = ChatGPTAdapter(config=config)

    # Mock failed response
    async def mock_failed_post(*args, **kwargs) -> MockResponse:
        return MockResponse(401)

    mock_http_client.post = mock_failed_post
    adapter._client = mock_http_client

    # Test initialization failure
    with pytest.raises(ConnectionError):
        await adapter.initialize()
    assert not adapter.is_available 