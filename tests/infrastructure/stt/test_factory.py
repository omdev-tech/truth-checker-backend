"""Tests for the STT provider factory."""

from typing import AsyncIterator, Dict, Optional

import pytest
import pytest_asyncio

from truth_checker.domain.ports.stt_provider import (
    AudioFormat,
    STTProvider,
    TranscriptionResult,
    VideoFormat,
)
from truth_checker.infrastructure.stt.elevenlabs_adapter import ElevenLabsAdapter
from truth_checker.infrastructure.stt.factory import STTProviderFactory


class TestSTTProvider(STTProvider):
    """Test STT provider implementation."""

    def __init__(self, name: str = "test"):
        """Initialize test provider."""
        self._name = name
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the provider."""
        self._initialized = True

    async def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Mock file transcription."""
        return TranscriptionResult(
            text="Test transcription",
            confidence=0.95,
            language=language or "en",
            start_time=0.0,
            end_time=2.5,
            metadata={},
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        chunk_size: int = 4096,
        language: Optional[str] = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Mock stream transcription."""
        async for _ in audio_stream:
            yield TranscriptionResult(
                text="Test transcription",
                confidence=0.95,
                language=language or "en",
                start_time=0.0,
                end_time=2.5,
                metadata={},
            )

    async def shutdown(self) -> None:
        """Shutdown the provider."""
        self._initialized = False

    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self._name

    @property
    def supported_audio_formats(self) -> list[AudioFormat]:
        """Get supported audio formats."""
        return [AudioFormat.WAV]

    @property
    def supported_video_formats(self) -> list[VideoFormat]:
        """Get supported video formats."""
        return [VideoFormat.MP4]

    @property
    def supported_languages(self) -> list[str]:
        """Get supported languages."""
        return ["en"]

    @property
    def is_available(self) -> bool:
        """Check if provider is available."""
        return self._initialized

    @property
    def capabilities(self) -> Dict[str, bool]:
        """Get provider capabilities."""
        return {
            "file_transcription": True,
            "stream_transcription": True,
        }


@pytest_asyncio.fixture
async def factory():
    """Create a factory instance."""
    factory = STTProviderFactory()
    yield factory
    await factory.shutdown_all()


@pytest.mark.asyncio
async def test_register_provider(factory):
    """Test provider registration."""
    factory.register_provider("test", TestSTTProvider)
    assert "test" in factory.list_providers()

    with pytest.raises(ValueError):
        factory.register_provider("test", TestSTTProvider)


@pytest.mark.asyncio
async def test_create_provider(factory):
    """Test provider creation."""
    factory.register_provider("test", TestSTTProvider)
    provider = await factory.create_provider("test")
    assert isinstance(provider, TestSTTProvider)
    assert provider.is_available

    # Test provider reuse
    provider2 = await factory.create_provider("test")
    assert provider is provider2

    with pytest.raises(ValueError):
        await factory.create_provider("unknown")


@pytest.mark.asyncio
async def test_get_provider(factory):
    """Test provider retrieval."""
    with pytest.raises(ValueError):
        await factory.get_provider("test")

    factory.register_provider("test", TestSTTProvider)
    provider = await factory.create_provider("test")
    retrieved = await factory.get_provider("test")
    assert provider is retrieved


@pytest.mark.asyncio
async def test_provider_lifecycle(factory):
    """Test provider lifecycle management."""
    factory.register_provider("test", TestSTTProvider)
    provider = await factory.create_provider("test")
    assert provider.is_available
    assert "test" in factory.list_active_providers()

    await factory.shutdown_provider("test")
    assert not provider.is_available
    assert "test" not in factory.list_active_providers()

    # Test shutdown of non-existent provider
    await factory.shutdown_provider("unknown")


@pytest.mark.asyncio
async def test_shutdown_all(factory):
    """Test shutdown of all providers."""
    factory.register_provider("test1", TestSTTProvider)
    factory.register_provider("test2", TestSTTProvider)

    provider1 = await factory.create_provider("test1")
    provider2 = await factory.create_provider("test2")

    assert provider1.is_available
    assert provider2.is_available

    await factory.shutdown_all()

    assert not provider1.is_available
    assert not provider2.is_available
    assert not factory.list_active_providers()


@pytest.mark.asyncio
async def test_provider_availability(factory):
    """Test provider availability checking."""
    factory.register_provider("test", TestSTTProvider)
    assert not factory.is_provider_available("test")

    provider = await factory.create_provider("test")
    assert factory.is_provider_available("test")

    await factory.shutdown_provider("test")
    assert not factory.is_provider_available("test")


@pytest.mark.asyncio
async def test_elevenlabs_provider(factory):
    """Test built-in ElevenLabs provider."""
    assert "elevenlabs" in factory.list_providers()
    
    config = {"api_key": "test_key"}
    provider = await factory.create_provider("elevenlabs", config)
    assert isinstance(provider, ElevenLabsAdapter)
    assert provider.is_available

    await factory.shutdown_provider("elevenlabs") 