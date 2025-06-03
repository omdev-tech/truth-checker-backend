"""Tests for fact-checking service."""

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from truth_checker.domain.services.fact_checking_service import FactCheckingService, FactCheckResult
from truth_checker.infrastructure.ai.chatgpt_provider import ChatGPTProvider


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = AsyncMock()
    client.chat = AsyncMock()
    client.chat.completions = AsyncMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest_asyncio.fixture
async def ai_provider(mock_openai_client):
    """Create a ChatGPT provider for testing."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        provider = ChatGPTProvider()
        provider._client = mock_openai_client
        provider._initialized = True
        return provider


@pytest.fixture
def mcp_client():
    """Create a mock MCP client."""
    client = AsyncMock()
    
    # Mock search results
    client.search.return_value = ["Paris", "France"]
    
    # Mock article summaries
    client.summary.side_effect = [
        "Paris is the capital and largest city of France.",
        "France is a country in Western Europe with Paris as its capital."
    ]
    
    return client


@pytest.mark.asyncio
async def test_check_fact(ai_provider, mcp_client, mock_openai_client):
    """Test fact checking with AI and MCP."""
    service = FactCheckingService(ai_provider, mcp_client)
    
    # Mock AI responses
    analysis_response = AsyncMock()
    analysis_response.choices = [
        AsyncMock(
            message=AsyncMock(
                content=json.dumps({
                    "claims": [
                        {
                            "claim": "Paris is the capital of France",
                            "search_query": "Paris France capital"
                        }
                    ]
                })
            )
        )
    ]
    
    validation_response = AsyncMock()
    validation_response.choices = [
        AsyncMock(
            message=AsyncMock(
                content=json.dumps({
                    "verdict": "true",
                    "confidence": 0.99,
                    "analysis": "Multiple sources confirm Paris as France's capital",
                    "evidence_used": [
                        "Paris is the capital and largest city of France.",
                        "France is a country in Western Europe with Paris as its capital."
                    ]
                })
            )
        )
    ]
    
    explanation_response = AsyncMock()
    explanation_response.choices = [
        AsyncMock(
            message=AsyncMock(
                content="This statement is true. Multiple reliable sources confirm that Paris is indeed the capital of France."
            )
        )
    ]
    
    # Set up mock responses
    mock_openai_client.chat.completions.create.side_effect = [
        analysis_response,
        validation_response,
        explanation_response
    ]
    
    result = await service.check_fact("Paris is the capital of France")
    
    # Verify result
    assert isinstance(result, FactCheckResult)
    assert result.is_valid is True
    assert result.confidence > 0.9
    assert "Paris" in result.explanation
    assert "capital" in result.explanation
    assert len(result.evidence) == 2
    
    # Verify MCP client calls
    mcp_client.search.assert_called_once_with("Paris France capital")
    assert mcp_client.summary.call_count == 2
    
    # Verify AI provider calls
    assert mock_openai_client.chat.completions.create.call_count == 3  # analyze, validate, explain


@pytest.mark.asyncio
async def test_check_fact_with_false_statement(ai_provider, mcp_client, mock_openai_client):
    """Test fact checking with a false statement."""
    service = FactCheckingService(ai_provider, mcp_client)
    
    # Mock different evidence for false statement
    mcp_client.summary.side_effect = [
        "The Earth is an oblate spheroid, slightly flattened at the poles.",
        "Scientific evidence confirms the Earth is roughly spherical in shape."
    ]
    
    # Mock AI responses for false statement
    analysis_response = AsyncMock()
    analysis_response.choices = [
        AsyncMock(
            message=AsyncMock(
                content=json.dumps({
                    "claims": [
                        {
                            "claim": "The Earth is flat",
                            "search_query": "Earth shape flat sphere"
                        }
                    ]
                })
            )
        )
    ]
    
    validation_response = AsyncMock()
    validation_response.choices = [
        AsyncMock(
            message=AsyncMock(
                content=json.dumps({
                    "verdict": "false",
                    "confidence": 0.99,
                    "analysis": "Scientific evidence contradicts the flat Earth claim",
                    "evidence_used": [
                        "The Earth is an oblate spheroid, slightly flattened at the poles.",
                        "Scientific evidence confirms the Earth is roughly spherical in shape."
                    ]
                })
            )
        )
    ]
    
    explanation_response = AsyncMock()
    explanation_response.choices = [
        AsyncMock(
            message=AsyncMock(
                content="This statement is false. Scientific evidence clearly shows that the Earth is roughly spherical, specifically an oblate spheroid."
            )
        )
    ]
    
    # Set up mock responses
    mock_openai_client.chat.completions.create.side_effect = [
        analysis_response,
        validation_response,
        explanation_response
    ]
    
    result = await service.check_fact("The Earth is flat")
    
    # Verify result
    assert isinstance(result, FactCheckResult)
    assert result.is_valid is False
    assert result.confidence > 0.9
    assert "spherical" in result.explanation.lower()
    assert len(result.evidence) == 2
    
    # Verify MCP client calls
    mcp_client.search.assert_called_once_with("Earth shape flat sphere")
    assert mcp_client.summary.call_count == 2 