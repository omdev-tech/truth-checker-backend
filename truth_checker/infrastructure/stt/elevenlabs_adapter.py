"""ElevenLabs implementation of the STT provider interface."""

import asyncio
import os
from typing import AsyncIterator, Dict, Optional

import httpx
from pydantic import BaseModel

from ...domain.ports.stt_provider import (
    AudioFormat,
    STTProvider,
    TranscriptionResult,
    VideoFormat,
)


class ElevenLabsConfig(BaseModel):
    """Configuration for ElevenLabs adapter."""
    
    api_key: str
    model: str = "eleven-english-v2"  # Default model
    timeout: float = 30.0
    base_url: str = "https://api.elevenlabs.io/v1"


class ElevenLabsAdapter(STTProvider):
    """ElevenLabs implementation of the STT provider interface."""

    def __init__(
        self,
        config: Optional[ElevenLabsConfig] = None,
        provider_name: str = "ElevenLabs",
    ):
        """Initialize the adapter."""
        self._config = config or ElevenLabsConfig(api_key="")
        self._name = provider_name
        self._client = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the HTTP client and verify API access."""
        try:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    base_url=self._config.base_url,
                    timeout=self._config.timeout,
                    headers={
                        "xi-api-key": self._config.api_key,
                        # Don't set Content-Type for multipart requests - httpx will handle this
                    },
                )

            # Test connection using the correct health endpoint
            response = await self._client.get("/health")
            response.raise_for_status()
            self._initialized = True
        except Exception as e:
            self._initialized = False
            if self._client:
                await self._client.aclose()
                self._client = None
            raise ConnectionError(f"Failed to initialize ElevenLabs provider: {e}")

    async def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe an audio/video file."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        # Verify file exists and is supported format
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = os.path.splitext(file_path)[1].lower()[1:]
        if extension not in [f.value for f in AudioFormat] + [f.value for f in VideoFormat]:
            raise ValueError(f"Unsupported file format: {extension}")

        try:
            # Upload file for transcription using the correct ElevenLabs API
            with open(file_path, "rb") as f:
                files = {"file": f}
                data = {
                    "model_id": "scribe_v1",  # Use the correct model ID
                    "timestamps_granularity": "word",
                    "diarize": False,
                    "tag_audio_events": False
                }
                
                if language:
                    data["language_code"] = language

                response = await self._client.post(
                    "/speech-to-text",  # Correct endpoint
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                result = response.json()

            # Parse the response according to the actual ElevenLabs API format
            # Extract plain text from words array
            text = result.get("text", "")
            if not text and result.get("words"):
                # Reconstruct text from words if main text field is missing
                text = "".join([word["text"] for word in result["words"]])

            # Calculate duration from last word's end time
            duration = 0.0
            if result.get("words"):
                last_word = result["words"][-1]
                duration = last_word.get("end", 0.0)

            return TranscriptionResult(
                text=text,
                confidence=result.get("language_probability", 1.0),  # Use language probability as confidence
                language=result.get("language_code", language or "en"),
                start_time=0.0,
                end_time=duration,
                metadata={
                    "words": result.get("words", []),
                    "model": "scribe_v1"
                },
            )
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        chunk_size: int = 4096,
        language: Optional[str] = None,
    ) -> AsyncIterator[TranscriptionResult]:
        """Transcribe an audio stream in real-time."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        try:
            # Initialize streaming session
            response = await self._client.post(
                "/speech-to-text/stream",
                json={
                    "model": self._config.model,
                    "language": language or "en",
                },
            )
            response.raise_for_status()
            session = response.json()["session_id"]

            # Process audio chunks
            buffer = bytearray()
            current_time = 0.0

            async for chunk in audio_stream:
                buffer.extend(chunk)

                # Process when we have enough data
                if len(buffer) >= chunk_size:
                    # Send chunk for transcription
                    response = await self._client.post(
                        f"/speech-to-text/stream/{session}",
                        content=bytes(buffer[:chunk_size]),
                    )
                    response.raise_for_status()
                    result = response.json()

                    # Clear processed data
                    buffer = buffer[chunk_size:]

                    if result.get("text"):
                        yield TranscriptionResult(
                            text=result["text"],
                            confidence=result["confidence"],
                            language=result.get("language", "en"),
                            start_time=current_time,
                            end_time=current_time + result["duration"],
                            metadata=result.get("metadata", {}),
                        )
                        current_time += result["duration"]

            # Process any remaining audio
            if buffer:
                response = await self._client.post(
                    f"/speech-to-text/stream/{session}",
                    content=bytes(buffer),
                )
                response.raise_for_status()
                result = response.json()

                if result.get("text"):
                    yield TranscriptionResult(
                        text=result["text"],
                        confidence=result["confidence"],
                        language=result.get("language", "en"),
                        start_time=current_time,
                        end_time=current_time + result["duration"],
                        metadata=result.get("metadata", {}),
                    )

            # Close streaming session
            await self._client.delete(f"/speech-to-text/stream/{session}")

        except Exception as e:
            raise RuntimeError(f"Streaming transcription failed: {e}")
        finally:
            # Ensure session is closed even if error occurs
            if "session" in locals():
                try:
                    await self._client.delete(f"/speech-to-text/stream/{session}")
                except Exception:
                    pass

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._name

    @property
    def supported_audio_formats(self) -> list[AudioFormat]:
        """Get list of supported audio formats."""
        return [
            AudioFormat.WAV,
            AudioFormat.MP3,
            AudioFormat.M4A,
            AudioFormat.OGG,
        ]

    @property
    def supported_video_formats(self) -> list[VideoFormat]:
        """Get list of supported video formats."""
        return [
            VideoFormat.MP4,
            VideoFormat.WEBM,
        ]

    @property
    def supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return ["en", "es", "fr", "de", "it"]  # Add more as supported

    @property
    def is_available(self) -> bool:
        """Check if the provider is available."""
        return self._initialized and self._client is not None

    @property
    def capabilities(self) -> Dict[str, bool]:
        """Get the provider capabilities."""
        return {
            "file_transcription": True,
            "stream_transcription": True,
            "real_time_processing": True,
            "multilingual": True,
        } 