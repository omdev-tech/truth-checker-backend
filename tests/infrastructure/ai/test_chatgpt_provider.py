"""Tests for ChatGPT provider implementation."""

import json
import pytest
from unittest.mock import AsyncMock, patch

from truth_checker.domain.ports.ai_provider import AIResponse
from truth_checker.infrastructure.ai.chatgpt_provider import ChatGPTProvider


@pytest.fixture
async def provider():
    """Create a ChatGPT provider for testing."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        provider = ChatGPTProvider()
        await provider.initialize()
        yield provider
        await provider.shutdown()


@pytest.mark.asyncio
async def test_analyze_statement(provider):
    """Test statement analysis."""
    # Mock OpenAI response
    mock_response = AsyncMock()
    mock_response.choices = [
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
    
    with patch.object(provider._client.chat.completions, 'create', return_value=mock_response):
        result = await provider.analyze_statement("Paris is the capital of France")
        
        assert isinstance(result, AIResponse)
        content = json.loads(result.content)
        assert "claims" in content
        assert len(content["claims"]) == 1
        assert content["claims"][0]["claim"] == "Paris is the capital of France"


@pytest.mark.asyncio
async def test_validate_with_evidence(provider):
    """Test evidence validation."""
    evidence = [
        "Paris is the capital and largest city of France.",
        "As the capital, Paris is the seat of France's government."
    ]
    
    mock_response = AsyncMock()
    mock_response.choices = [
        AsyncMock(
            message=AsyncMock(
                content=json.dumps({
                    "verdict": "true",
                    "confidence": 0.99,
                    "analysis": "Multiple sources confirm Paris as France's capital",
                    "evidence_used": evidence
                })
            )
        )
    ]
    
    with patch.object(provider._client.chat.completions, 'create', return_value=mock_response):
        result = await provider.validate_with_evidence(
            "Paris is the capital of France",
            evidence
        )
        
        assert isinstance(result, AIResponse)
        content = json.loads(result.content)
        assert content["verdict"] == "true"
        assert content["confidence"] > 0.9
        assert len(content["evidence_used"]) == 2


@pytest.mark.asyncio
async def test_generate_explanation(provider):
    """Test explanation generation."""
    evidence = [
        "Paris is the capital and largest city of France.",
        "As the capital, Paris is the seat of France's government."
    ]
    
    mock_response = AsyncMock()
    mock_response.choices = [
        AsyncMock(
            message=AsyncMock(
                content="This statement is true. Multiple reliable sources confirm that Paris is indeed the capital of France, serving as the seat of the French government."
            )
        )
    ]
    
    with patch.object(provider._client.chat.completions, 'create', return_value=mock_response):
        explanation = await provider.generate_explanation(
            "Paris is the capital of France",
            True,
            evidence
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) <= 500
        assert "Paris" in explanation
        assert "capital" in explanation 