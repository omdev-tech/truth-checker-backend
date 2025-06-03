"""Tests for the Wikipedia MCP adapter."""

import json
from typing import Dict, List

import pytest
import pytest_asyncio
from httpx import AsyncClient, HTTPStatusError, Request, Response

from truth_checker.domain.ports.mcp_provider import MCPSearchResult, MCPValidationResult
from truth_checker.infrastructure.mcp.wikipedia_adapter import WikipediaMCPAdapter


def create_response(status_code: int, json_data: dict = None, text: str = None) -> Response:
    """Create a Response object with a proper request."""
    request = Request("GET", "http://test-mcp-server/health")
    content = (
        json.dumps(json_data).encode() if json_data is not None
        else text.encode() if text is not None
        else b""
    )
    headers = (
        {"content-type": "application/json"} if json_data is not None
        else {"content-type": "text/plain"}
    )
    return Response(
        status_code=status_code,
        headers=headers,
        content=content,
        request=request,
    )


@pytest_asyncio.fixture
async def wikipedia_adapter(mock_http_client: AsyncClient) -> WikipediaMCPAdapter:
    """Create a Wikipedia adapter with the mock client."""
    adapter = WikipediaMCPAdapter(
        base_url="http://test-mcp-server",
        timeout=1.0,
    )
    adapter._client = mock_http_client
    return adapter


@pytest.mark.asyncio
async def test_initialize_success(wikipedia_adapter: WikipediaMCPAdapter, mock_http_client: AsyncClient):
    """Test successful initialization."""
    # Create a new response with proper request
    response = create_response(200)
    
    # Mock the get method
    async def mock_get(*args, **kwargs):
        return response
    
    mock_http_client.get = mock_get
    
    # Set the client before initializing
    wikipedia_adapter._client = mock_http_client
    
    # Initialize should succeed
    await wikipedia_adapter.initialize()
    assert wikipedia_adapter.is_available


@pytest.mark.asyncio
async def test_initialize_failure(wikipedia_adapter: WikipediaMCPAdapter, mock_http_client: AsyncClient):
    """Test initialization failure."""
    # Create a failing response
    response = create_response(500)
    
    # Mock the get method to raise an error
    async def mock_get(*args, **kwargs):
        raise HTTPStatusError("Server error", request=response.request, response=response)
    
    mock_http_client.get = mock_get
    
    # Set the client before initializing
    wikipedia_adapter._client = mock_http_client
    
    with pytest.raises(ConnectionError):
        await wikipedia_adapter.initialize()
    assert not wikipedia_adapter.is_available


@pytest.mark.asyncio
async def test_search(wikipedia_adapter: WikipediaMCPAdapter, mock_http_client: AsyncClient):
    """Test article search functionality."""
    mock_results = [
        {
            "title": "Test Article",
            "summary": "Test summary",
            "url": "http://test.com",
            "score": 0.95,
            "metadata": {"key": "value"},
        }
    ]

    async def mock_post(url: str, **kwargs):
        if url.endswith("/search"):
            return create_response(200, json_data=mock_results)
        return create_response(404)

    mock_http_client.post = mock_post
    results = await wikipedia_adapter.search("test query")

    assert len(results) == 1
    assert results[0].title == "Test Article"
    assert results[0].summary == "Test summary"
    assert results[0].url == "http://test.com"
    assert results[0].relevance_score == 0.95
    assert results[0].metadata == {"key": "value"}


@pytest.mark.asyncio
async def test_get_article_summary(wikipedia_adapter: WikipediaMCPAdapter, mock_http_client: AsyncClient):
    """Test article summary retrieval."""
    mock_summary = {"summary": "Test article summary"}

    async def mock_post(url: str, **kwargs):
        if url.endswith("/summary"):
            return create_response(200, json_data=mock_summary)
        return create_response(404)

    mock_http_client.post = mock_post
    summary = await wikipedia_adapter.get_article_summary("Test Article")

    assert summary == "Test article summary"


@pytest.mark.asyncio
async def test_get_article_content(wikipedia_adapter: WikipediaMCPAdapter, mock_http_client: AsyncClient):
    """Test article content retrieval."""
    mock_content = {
        "text": "Full article content",
        "metadata": {"sections": ["Introduction", "Content"]},
    }

    async def mock_post(url: str, **kwargs):
        if url.endswith("/page"):
            return create_response(200, json_data=mock_content)
        return create_response(404)

    mock_http_client.post = mock_post
    content = await wikipedia_adapter.get_article_content("Test Article")

    assert content["text"] == "Full article content"
    assert content["metadata"]["sections"] == ["Introduction", "Content"]


@pytest.mark.asyncio
async def test_validate_fact(wikipedia_adapter: WikipediaMCPAdapter, mock_http_client: AsyncClient):
    """Test fact validation."""
    mock_search_results = [
        {
            "title": "Relevant Article",
            "summary": "Contains the fact",
            "url": "http://test.com",
            "score": 0.9,
            "metadata": {},
        }
    ]

    mock_content = {
        "text": "This is a test fact that should be validated",
        "metadata": {},
    }

    async def mock_post(url: str, **kwargs) -> Response:
        if url.endswith("/search"):
            return create_response(200, json_data=mock_search_results)
        elif url.endswith("/page"):
            return create_response(200, json_data=mock_content)
        return create_response(404)

    mock_http_client.post = mock_post

    result = await wikipedia_adapter.validate_fact("test fact")

    assert result.is_valid
    assert result.confidence > 0.5
    assert len(result.supporting_evidence) > 0
    assert len(result.source_urls) > 0


@pytest.mark.asyncio
async def test_set_language(wikipedia_adapter: WikipediaMCPAdapter, mock_http_client: AsyncClient):
    """Test language setting."""
    async def mock_post(url: str, **kwargs):
        if url.endswith("/set_lang"):
            return create_response(200)
        return create_response(404)

    mock_http_client.post = mock_post
    await wikipedia_adapter.set_language("es")
    assert "es" in wikipedia_adapter.supported_languages


@pytest.mark.asyncio
async def test_provider_properties(wikipedia_adapter: WikipediaMCPAdapter, mock_http_client: AsyncClient):
    """Test provider property getters."""
    assert wikipedia_adapter.provider_name == "Wikipedia"
    assert len(wikipedia_adapter.supported_languages) > 0
    assert not wikipedia_adapter.is_available  # Not initialized yet
    assert "search" in wikipedia_adapter.capabilities
    assert "validation" in wikipedia_adapter.capabilities


@pytest.mark.asyncio
async def test_error_handling(wikipedia_adapter: WikipediaMCPAdapter, mock_http_client: AsyncClient):
    """Test error handling for various operations."""
    async def mock_post(url: str, **kwargs):
        response = create_response(500)
        response.raise_for_status()
        return response

    mock_http_client.post = mock_post

    with pytest.raises(HTTPStatusError):
        await wikipedia_adapter.search("test") 