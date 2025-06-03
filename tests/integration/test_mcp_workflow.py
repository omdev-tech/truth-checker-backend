"""Integration tests for MCP workflow."""

import json
from typing import Dict

import pytest
import pytest_asyncio
from httpx import AsyncClient, HTTPStatusError, Request, Response

from truth_checker.infrastructure.mcp.factory import mcp_factory
from truth_checker.infrastructure.mcp.wikipedia_adapter import WikipediaMCPAdapter


def create_response(status_code: int, json_data: dict = None, text: str = None) -> Response:
    """Create a Response object with a proper request."""
    request = Request("GET", "http://test-mcp-server")
    response = Response(status_code, request=request)
    if json_data is not None:
        response._content = json.dumps(json_data).encode()
    if text is not None:
        response._content = text.encode()
    return response


@pytest_asyncio.fixture
async def mock_provider(mock_http_client: AsyncClient) -> WikipediaMCPAdapter:
    """Create a mock provider with the mock client."""
    provider = WikipediaMCPAdapter(
        base_url="http://test-mcp-server",
        timeout=1.0,
    )
    provider._client = mock_http_client
    provider._initialized = True
    return provider


@pytest.mark.asyncio
async def test_complete_fact_checking_workflow(mock_provider: WikipediaMCPAdapter, mock_http_client: AsyncClient):
    """Test complete fact-checking workflow using Wikipedia MCP."""
    # Mock responses for each step
    mock_responses = {
        "/health": create_response(200),
        "/search": create_response(
            200,
            json_data=[
                {
                    "title": "Python (programming language)",
                    "summary": "Python is a high-level programming language.",
                    "url": "http://test.com/python",
                    "score": 0.95,
                    "metadata": {},
                }
            ],
        ),
        "/page": create_response(
            200,
            json_data={
                "text": "Python is a high-level programming language created by Guido van Rossum.",
                "metadata": {"sections": ["Overview", "History"]},
            },
        ),
        "/summary": create_response(
            200,
            json_data={"summary": "Python is a high-level programming language."},
        ),
    }

    async def mock_request(url: str, **kwargs) -> Response:
        for endpoint, response in mock_responses.items():
            if url.endswith(endpoint):
                return response
        return create_response(404)

    mock_http_client.get = mock_request
    mock_http_client.post = mock_request

    try:
        # Test the complete workflow
        fact = "Python is a high-level programming language"

        # 1. Search for relevant articles
        search_results = await mock_provider.search(fact)
        assert len(search_results) == 1
        assert search_results[0].title == "Python (programming language)"

        # 2. Get article summary
        summary = await mock_provider.get_article_summary(search_results[0].title)
        assert "Python" in summary
        assert "programming language" in summary

        # 3. Validate the fact
        validation = await mock_provider.validate_fact(fact)
        assert validation.is_valid
        assert validation.confidence > 0.5
        assert len(validation.supporting_evidence) > 0
        assert validation.source_urls == ["http://test.com/python"]

    finally:
        # Clean up
        await mock_provider.shutdown()


@pytest.mark.asyncio
async def test_multilingual_fact_checking(mock_provider: WikipediaMCPAdapter, mock_http_client: AsyncClient):
    """Test fact-checking in different languages."""
    # Mock responses for Spanish language
    mock_responses = {
        "/health": create_response(200),
        "/set_lang": create_response(200),
        "/search": create_response(
            200,
            json_data=[
                {
                    "title": "Python (lenguaje de programación)",
                    "summary": "Python es un lenguaje de programación.",
                    "url": "http://test.com/python/es",
                    "score": 0.95,
                    "metadata": {"language": "es"},
                }
            ],
        ),
        "/page": create_response(
            200,
            json_data={
                "text": "Python es un lenguaje de programación de alto nivel.",
                "metadata": {"language": "es"},
            },
        ),
    }

    async def mock_request(url: str, **kwargs) -> Response:
        for endpoint, response in mock_responses.items():
            if url.endswith(endpoint):
                return response
        return create_response(404)

    mock_http_client.get = mock_request
    mock_http_client.post = mock_request

    try:
        # Set language to Spanish
        await mock_provider.set_language("es")

        # Test Spanish fact validation
        fact = "Python es un lenguaje de programación"
        validation = await mock_provider.validate_fact(fact)

        assert validation.is_valid
        assert validation.confidence > 0.5
        assert len(validation.supporting_evidence) > 0
        assert any("es un lenguaje" in evidence for evidence in validation.supporting_evidence)

    finally:
        # Clean up
        await mock_provider.shutdown()


@pytest.mark.asyncio
async def test_error_recovery(mock_provider: WikipediaMCPAdapter, mock_http_client: AsyncClient):
    """Test error recovery and fallback behavior."""
    # Simulate intermittent failures
    failure_count = 0

    async def mock_request(url: str, **kwargs) -> Response:
        nonlocal failure_count

        if url.endswith("/health"):
            return create_response(200)

        if failure_count < 2:  # Fail first two requests
            failure_count += 1
            response = create_response(500, text="Server error")
            raise HTTPStatusError("Server error", request=response.request, response=response)

        # Succeed after failures
        if url.endswith("/search"):
            return create_response(
                200,
                json_data=[
                    {
                        "title": "Test Article",
                        "summary": "Test content",
                        "url": "http://test.com",
                        "score": 0.8,
                        "metadata": {},
                    }
                ],
            )
        return create_response(404)

    mock_http_client.get = mock_request
    mock_http_client.post = mock_request

    try:
        # First attempt should fail
        with pytest.raises(HTTPStatusError):
            await mock_provider.search("test")

        # Second attempt should fail
        with pytest.raises(HTTPStatusError):
            await mock_provider.search("test")

        # Third attempt should succeed
        results = await mock_provider.search("test")
        assert len(results) == 1
        assert results[0].title == "Test Article"

    finally:
        # Clean up
        await mock_provider.shutdown() 