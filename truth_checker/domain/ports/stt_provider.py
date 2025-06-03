"""Port interface for Speech-to-Text (STT) providers."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncIterator, Dict, Optional

from pydantic import BaseModel


class AudioFormat(str, Enum):
    """Supported audio formats."""
    
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    OGG = "ogg"
    WEBM = "webm"


class VideoFormat(str, Enum):
    """Supported video formats."""
    
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"


class TranscriptionResult(BaseModel):
    """Result of audio/video transcription."""
    
    text: str
    confidence: float
    language: str
    start_time: float  # In seconds
    end_time: float  # In seconds
    metadata: Dict = {}


class STTProvider(ABC):
    """Abstract interface for Speech-to-Text providers.
    
    This port defines how STT services will interact with our system.
    Concrete implementations will be provided for different STT providers
    (ElevenLabs, etc.) in the infrastructure layer.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the STT provider and its resources."""
        pass

    @abstractmethod
    async def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio/video file.

        Args:
            file_path: Path to the audio/video file
            language: Optional language code (ISO 639-1)

        Returns:
            Transcription result
        """
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        chunk_size: int = 4096,  # 4KB chunks
        language: Optional[str] = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Transcribe an audio stream in real-time.

        Args:
            audio_stream: Async iterator providing audio chunks
            chunk_size: Size of each audio chunk in bytes
            language: Optional language code (ISO 639-1)

        Returns:
            Async iterator of transcription results
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the provider and clean up resources."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass

    @property
    @abstractmethod
    def supported_audio_formats(self) -> list[AudioFormat]:
        """Get list of supported audio formats."""
        pass

    @property
    @abstractmethod
    def supported_video_formats(self) -> list[VideoFormat]:
        """Get list of supported video formats."""
        pass

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> Dict[str, bool]:
        """Get the provider capabilities."""
        pass 