"""Tests for the ElevenLabs STT adapter."""

import os
import tempfile
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from truth_checker.domain.ports.stt_provider import AudioFormat, TranscriptionResult
from truth_checker.infrastructure.stt.elevenlabs_adapter import (
    ElevenLabsAdapter,
    ElevenLabsConfig,
)


@pytest.fixture
def mock_response():
    """Create a mock HTTP response."""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json = MagicMock(
        return_value={
            "text": "Test transcription",
            "confidence": 0.95,
            "language": "en",
            "duration": 2.5,
            "metadata": {},
        }
    )
    return response


@pytest.fixture
def mock_client(mock_response):
    """Create a mock HTTP client."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=mock_response)
    client.post = AsyncMock(return_value=mock_response)
    client.delete = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest_asyncio.fixture
async def adapter(mock_client):
    """Create an ElevenLabs adapter instance."""
    with patch("httpx.AsyncClient", return_value=mock_client):
        adapter = ElevenLabsAdapter(
            config=ElevenLabsConfig(api_key="test_key"),
        )
        await adapter.initialize()
        yield adapter
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_initialization():
    """Test adapter initialization."""
    adapter = ElevenLabsAdapter(
        config=ElevenLabsConfig(api_key="test_key"),
    )
    assert not adapter.is_available

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.get.return_value = AsyncMock(
            status_code=200,
            raise_for_status=MagicMock(),
        )
        await adapter.initialize()
        assert adapter.is_available

        await adapter.shutdown()
        assert not adapter.is_available


@pytest.mark.asyncio
async def test_initialization_failure():
    """Test adapter initialization failure."""
    adapter = ElevenLabsAdapter(
        config=ElevenLabsConfig(api_key="test_key"),
    )

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.get.side_effect = httpx.RequestError("Test error")
        with pytest.raises(ConnectionError):
            await adapter.initialize()
        assert not adapter.is_available


@pytest.mark.asyncio
async def test_transcribe_file(adapter, mock_client):
    """Test file transcription."""
    # Create temporary test file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(b"test audio content")
        temp_file.flush()

        try:
            result = await adapter.transcribe_file(temp_file.name)
            assert isinstance(result, TranscriptionResult)
            assert result.text == "Test transcription"
            assert result.confidence == 0.95
            assert result.language == "en"
            assert result.start_time == 0.0
            assert result.end_time == 2.5
        finally:
            os.unlink(temp_file.name)


@pytest.mark.asyncio
async def test_transcribe_file_not_found(adapter):
    """Test file transcription with non-existent file."""
    with pytest.raises(FileNotFoundError):
        await adapter.transcribe_file("nonexistent.wav")


@pytest.mark.asyncio
async def test_transcribe_file_invalid_format(adapter):
    """Test file transcription with invalid format."""
    with tempfile.NamedTemporaryFile(suffix=".invalid", delete=False) as temp_file:
        temp_file.write(b"test content")
        temp_file.flush()

        try:
            with pytest.raises(ValueError):
                await adapter.transcribe_file(temp_file.name)
        finally:
            os.unlink(temp_file.name)


@pytest.mark.asyncio
async def test_transcribe_stream(adapter, mock_client):
    """Test stream transcription."""
    async def mock_stream() -> AsyncIterator[bytes]:
        yield b"chunk1"
        yield b"chunk2"

    results = []
    async for result in adapter.transcribe_stream(mock_stream()):
        results.append(result)

    assert len(results) > 0
    for result in results:
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Test transcription"
        assert result.confidence == 0.95
        assert result.language == "en"


@pytest.mark.asyncio
async def test_transcribe_stream_error(adapter, mock_client):
    """Test stream transcription error."""
    mock_client.post.side_effect = httpx.RequestError("Test error")

    async def mock_stream() -> AsyncIterator[bytes]:
        yield b"chunk1"

    with pytest.raises(RuntimeError):
        async for _ in adapter.transcribe_stream(mock_stream()):
            pass


def test_provider_properties(adapter):
    """Test provider properties."""
    assert adapter.provider_name == "ElevenLabs"
    assert AudioFormat.WAV in adapter.supported_audio_formats
    assert len(adapter.supported_languages) > 0
    assert adapter.is_available
    assert adapter.capabilities["file_transcription"]
    assert adapter.capabilities["stream_transcription"] 