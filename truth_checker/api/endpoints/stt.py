"""STT endpoints for file upload and WebSocket streaming."""

import asyncio
import os
import tempfile
from typing import AsyncIterator, Optional

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel

from ...domain.ports.stt_provider import AudioFormat, TranscriptionResult, VideoFormat
from ...infrastructure.stt.factory import STTProviderFactory

router = APIRouter(prefix="/stt", tags=["stt"])


class TranscriptionRequest(BaseModel):
    """Request model for transcription."""

    provider: str = "elevenlabs"
    language: Optional[str] = None


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""

    text: str
    confidence: float
    language: str
    duration: float
    metadata: dict = {}


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    request: TranscriptionRequest = TranscriptionRequest(),
) -> TranscriptionResponse:
    """Transcribe an audio/video file.

    Args:
        file: Audio/video file to transcribe
        request: Transcription request parameters

    Returns:
        Transcription result
    """
    # Verify file format
    extension = os.path.splitext(file.filename)[1].lower()[1:]
    supported_formats = [f.value for f in AudioFormat] + [f.value for f in VideoFormat]
    if extension not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(supported_formats)}",
        )

    # Save file temporarily
    with tempfile.NamedTemporaryFile(suffix=f".{extension}", delete=False) as temp_file:
        try:
            # Write uploaded file to temp file
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()

            # Get STT provider
            factory = STTProviderFactory()
            try:
                provider = await factory.create_provider(request.provider)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            # Transcribe file
            try:
                result = await provider.transcribe_file(
                    temp_file.name,
                    language=request.language,
                )
                return TranscriptionResponse(
                    text=result.text,
                    confidence=result.confidence,
                    language=result.language,
                    duration=result.end_time - result.start_time,
                    metadata=result.metadata,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                await factory.shutdown_provider(request.provider)

        finally:
            # Clean up temp file
            os.unlink(temp_file.name)


async def process_audio_stream(
    websocket: WebSocket,
    chunk_size: int = 4096,
    provider_name: str = "elevenlabs",
    language: Optional[str] = None,
) -> AsyncIterator[TranscriptionResult]:
    """Process audio stream from WebSocket.

    Args:
        websocket: WebSocket connection
        chunk_size: Size of audio chunks to process
        provider_name: Name of STT provider to use
        language: Optional language code

    Yields:
        Transcription results
    """
    # Initialize STT provider
    factory = STTProviderFactory()
    provider = await factory.create_provider(provider_name)

    try:
        # Create async iterator for audio chunks
        async def audio_stream() -> AsyncIterator[bytes]:
            while True:
                try:
                    chunk = await websocket.receive_bytes()
                    yield chunk
                except WebSocketDisconnect:
                    break

        # Process audio stream
        async for result in provider.transcribe_stream(
            audio_stream(),
            chunk_size=chunk_size,
            language=language,
        ):
            yield result

    finally:
        await factory.shutdown_provider(provider_name)


@router.websocket("/stream")
async def stream_audio(
    websocket: WebSocket,
    provider: str = "elevenlabs",
    language: Optional[str] = None,
):
    """WebSocket endpoint for real-time audio streaming.

    Args:
        websocket: WebSocket connection
        provider: STT provider to use
        language: Optional language code
    """
    await websocket.accept()

    try:
        # Process audio stream and send results
        async for result in process_audio_stream(
            websocket,
            provider_name=provider,
            language=language,
        ):
            await websocket.send_json(
                {
                    "text": result.text,
                    "confidence": result.confidence,
                    "language": result.language,
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                    "metadata": result.metadata,
                }
            )

    except WebSocketDisconnect:
        pass  # Client disconnected
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close() 