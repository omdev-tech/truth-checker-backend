"""Domain service for Speech-to-Text operations."""

import logging
from typing import Optional

from ..ports.stt_provider import STTProvider, AudioFormat, VideoFormat, TranscriptionResult

logger = logging.getLogger(__name__)


class STTService:
    """Domain service for Speech-to-Text operations.
    
    This service coordinates STT operations using the STT provider port.
    It provides a clean interface for the application layer while
    delegating actual transcription to infrastructure adapters.
    """
    
    def __init__(self, stt_provider: STTProvider):
        """Initialize service with STT provider.
        
        Args:
            stt_provider: STT provider port implementation
        """
        self._stt_provider = stt_provider
    
    async def transcribe_file(
        self,
        file_path: str,
        format_type: AudioFormat | VideoFormat,
        language: Optional[str] = None,
        fast_mode: bool = True
    ) -> TranscriptionResult:
        """Transcribe an audio or video file.
        
        Args:
            file_path: Path to the audio/video file
            format_type: Audio or video format
            language: Optional language code (ISO 639-1)
            fast_mode: Whether to use fast processing mode
            
        Returns:
            Transcription result
        """
        logger.info(f"🎙️ Starting transcription: {file_path}")
        logger.info(f"📊 Parameters: format={format_type}, language={language}, fast_mode={fast_mode}")
        
        try:
            # Initialize provider if needed
            await self._stt_provider.initialize()
            
            # Transcribe file
            result = await self._stt_provider.transcribe_file(
                file_path=file_path,
                language=language
            )
            
            logger.info(f"✅ Transcription completed: {len(result.text)} chars, confidence={result.confidence}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            raise e
    
    async def transcribe_stream(
        self,
        audio_stream,
        chunk_size: int = 4096,
        language: Optional[str] = None
    ):
        """Transcribe an audio stream in real-time.
        
        Args:
            audio_stream: Async iterator providing audio chunks
            chunk_size: Size of each audio chunk in bytes
            language: Optional language code (ISO 639-1)
            
        Yields:
            Transcription results
        """
        logger.info(f"🔴 Starting real-time transcription: chunk_size={chunk_size}, language={language}")
        
        try:
            # Initialize provider if needed
            await self._stt_provider.initialize()
            
            # Transcribe stream
            async for result in self._stt_provider.transcribe_stream(
                audio_stream=audio_stream,
                chunk_size=chunk_size,
                language=language
            ):
                logger.debug(f"📝 Stream chunk transcribed: {len(result.text)} chars")
                yield result
                
        except Exception as e:
            logger.error(f"❌ Stream transcription failed: {e}")
            raise e
    
    async def shutdown(self) -> None:
        """Shutdown the STT service."""
        logger.info("🔄 Shutting down STT service...")
        try:
            await self._stt_provider.shutdown()
            logger.info("✅ STT service shutdown completed")
        except Exception as e:
            logger.error(f"❌ STT service shutdown failed: {e}")
            raise e
    
    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._stt_provider.provider_name
    
    @property
    def supported_audio_formats(self) -> list[AudioFormat]:
        """Get list of supported audio formats."""
        return self._stt_provider.supported_audio_formats
    
    @property
    def supported_video_formats(self) -> list[VideoFormat]:
        """Get list of supported video formats."""
        return self._stt_provider.supported_video_formats
    
    @property
    def supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return self._stt_provider.supported_languages
    
    @property
    def is_available(self) -> bool:
        """Check if the provider is available."""
        return self._stt_provider.is_available
    
    @property
    def capabilities(self) -> dict:
        """Get the provider capabilities."""
        return self._stt_provider.capabilities 