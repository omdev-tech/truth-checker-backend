"""Tests for the FastAPI application."""

import json
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

from truth_checker.api.app import app
from truth_checker.domain.ports.ai_provider import AIAnalysisResult, AIVerificationResult
from truth_checker.domain.ports.mcp_provider import MCPSearchResult
from truth_checker.infrastructure.ai.chatgpt_adapter import ChatGPTAdapter
from truth_checker.infrastructure.mcp.wikipedia_adapter import WikipediaMCPAdapter


class TestProvider:
    """Test provider implementation."""

    def __init__(self, name: str = "Test"):
        """Initialize test provider."""
        self._name = name
        self._initialized = True

    async def initialize(self) -> None:
        """Initialize the provider."""
        pass

    async def analyze_statement(self, text: str) -> AIAnalysisResult:
        """Analyze a statement."""
        return AIAnalysisResult(
            statement=text,
            is_factual_claim=True,
            confidence=0.9,
            extracted_facts=["Test fact"],
            metadata={"source": "test"},
        )

    async def verify_facts(
        self,
        facts: List[str],
        evidence: List[str],
    ) -> AIVerificationResult:
        """Verify facts."""
        return AIVerificationResult(
            facts=facts,
            verification_results=[{"fact": fact, "verified": True} for fact in facts],
            confidence=0.9,
            explanation="Test explanation",
            source_references=["Test source"],
        )

    async def search(self, query: str) -> List[MCPSearchResult]:
        """Search for content."""
        return [
            MCPSearchResult(
                title="Test Result",
                content="Test content",
                url="http://test.com",
                relevance=0.9,
            )
        ]

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        pass

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


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
async def mock_providers(monkeypatch):
    """Create mock providers."""
    # Create test providers
    ai_provider = TestProvider("TestAI")
    mcp_provider = TestProvider("TestMCP")

    # Clear existing providers
    app.ai_factory._provider_types.clear()
    app.ai_factory._providers.clear()
    await app.mcp_factory.shutdown()  # This will remove all providers

    # Mock provider factories
    app.ai_factory.register_provider_type("test_ai", lambda *args, **kwargs: ai_provider)
    app.mcp_factory.register_provider("TestMCP", mcp_provider)

    # Create providers
    await app.ai_factory.create_provider("test_ai", provider_name="TestAI")


def test_health_check(test_client: TestClient, mock_providers):
    """Test health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "TestAI" in data["ai_providers"]
    assert "TestMCP" in data["mcp_providers"]


def test_analyze_text(test_client: TestClient, mock_providers):
    """Test text analysis endpoint."""
    response = test_client.post(
        "/analyze",
        json={
            "text": "Test statement",
            "provider": "TestAI",
        },
    )
    assert response.status_code == 200
    
    data = response.json()
    assert data["analysis"]["is_factual_claim"]
    assert data["analysis"]["confidence"] > 0
    assert len(data["analysis"]["extracted_facts"]) > 0


def test_verify_facts(test_client: TestClient, mock_providers):
    """Test fact verification endpoint."""
    response = test_client.post(
        "/verify",
        json={
            "facts": ["Test fact"],
            "provider": "TestAI",
            "mcp_provider": "TestMCP",
        },
    )
    assert response.status_code == 200
    
    data = response.json()
    assert data["verification"]["confidence"] > 0
    assert len(data["verification"]["verification_results"]) > 0
    assert data["verification"]["explanation"]


def test_analyze_unknown_provider(test_client: TestClient, mock_providers):
    """Test error handling for unknown AI provider."""
    response = test_client.post(
        "/analyze",
        json={
            "text": "Test statement",
            "provider": "Unknown",
        },
    )
    assert response.status_code == 404
    assert "provider not found" in response.json()["detail"]


def test_verify_unknown_providers(test_client: TestClient, mock_providers):
    """Test error handling for unknown providers."""
    # Test unknown AI provider
    response = test_client.post(
        "/verify",
        json={
            "facts": ["Test fact"],
            "provider": "Unknown",
            "mcp_provider": "TestMCP",
        },
    )
    assert response.status_code == 404
    assert "ai provider not found" in response.json()["detail"]

    # Test unknown MCP provider
    response = test_client.post(
        "/verify",
        json={
            "facts": ["Test fact"],
            "provider": "TestAI",
            "mcp_provider": "Unknown",
        },
    )
    assert response.status_code == 404
    assert "mcp provider not found" in response.json()["detail"]


def test_upload_file(test_client: TestClient):
    """Test file upload endpoint."""
    response = test_client.post(
        "/upload",
        files={"file": ("test.txt", b"test content")},
    )
    assert response.status_code == 501
    assert "Phase 2" in response.json()["detail"] 