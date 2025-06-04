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
from ...infrastructure.ai.factory import AIProviderFactory

router = APIRouter(prefix="/stt", tags=["stt"])


class TranscriptionRequest(BaseModel):
    """Request model for transcription."""

    provider: str = "elevenlabs"
    language: Optional[str] = None


class ChunkProcessingRequest(BaseModel):
    """Request model for chunk processing (transcription + fact-checking)."""

    provider: str = "elevenlabs"
    language: Optional[str] = None
    fast_mode: bool = True  # Use fast processing for real-time feedback
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""

    text: str
    confidence: float
    language: str
    duration: float
    metadata: dict = {}


class ChunkProcessingResponse(BaseModel):
    """Response model for chunk processing."""

    transcription: TranscriptionResponse
    fact_check: dict
    processing_time: float
    chunk_info: dict


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


@router.post("/transcribe-chunk", response_model=ChunkProcessingResponse)
async def transcribe_and_fact_check_chunk(
    file: UploadFile = File(...),
    request: ChunkProcessingRequest = ChunkProcessingRequest(),
) -> ChunkProcessingResponse:
    """Transcribe and fact-check an audio/video chunk for real-time processing.

    Args:
        file: Audio/video file chunk to process
        request: Chunk processing request parameters

    Returns:
        Combined transcription and fact-check results
    """
    import time
    start_time = time.time()
    
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
            stt_factory = STTProviderFactory()
            ai_factory = AIProviderFactory()
            
            try:
                stt_provider = await stt_factory.create_provider(request.provider)
                
                # Create AI provider with fast mode if requested
                ai_config = {"fast_mode": request.fast_mode} if request.fast_mode else {}
                ai_provider = await ai_factory.create_provider("chatgpt", ai_config)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            # Transcribe file
            try:
                transcription_result = await stt_provider.transcribe_file(
                    temp_file.name,
                    language=request.language,
                )
                
                transcription_response = TranscriptionResponse(
                    text=transcription_result.text,
                    confidence=transcription_result.confidence,
                    language=transcription_result.language,
                    duration=transcription_result.end_time - transcription_result.start_time,
                    metadata=transcription_result.metadata,
                )
                
                # Fact-check the transcription if there's text
                fact_check_result = {"status": "no_text", "claims": [], "overall_confidence": 0.0}
                
                if transcription_result.text and len(transcription_result.text.strip()) > 10:
                    try:
                        # Extract claims from transcription
                        claims = await ai_provider.analyze_text(transcription_result.text)
                        
                        if claims:
                            # Verify claims (simplified for fast processing)
                            verified_claims = []
                            total_confidence = 0.0
                            
                            for claim in claims[:3]:  # Limit to first 3 claims for speed
                                verification = await ai_provider.verify_claim(claim)
                                verified_claims.append({
                                    "text": claim.text,
                                    "status": verification.status.value if verification.status else "unknown",
                                    "confidence": verification.confidence.value if verification.confidence else "low",
                                    "explanation": verification.explanation
                                })
                                
                                # Calculate confidence score
                                confidence_map = {"high": 1.0, "medium": 0.6, "low": 0.3, "insufficient": 0.1}
                                total_confidence += confidence_map.get(verification.confidence.value if verification.confidence else "low", 0.3)
                            
                            avg_confidence = total_confidence / len(verified_claims) if verified_claims else 0.0
                            
                            # Determine overall status
                            statuses = [claim["status"] for claim in verified_claims]
                            if any(status in ["false", "misleading"] for status in statuses):
                                overall_status = "false"
                            elif any(status == "disputed" for status in statuses):
                                overall_status = "uncertain"
                            elif any(status in ["true", "partially_true"] for status in statuses):
                                overall_status = "true"
                            else:
                                overall_status = "not_checkable"
                            
                            fact_check_result = {
                                "status": overall_status,
                                "claims": verified_claims,
                                "overall_confidence": avg_confidence,
                                "total_claims": len(claims),
                                "processed_claims": len(verified_claims)
                            }
                        else:
                            fact_check_result = {"status": "not_checkable", "claims": [], "overall_confidence": 0.0}
                            
                    except Exception as e:
                        fact_check_result = {
                            "status": "error", 
                            "error": str(e),
                            "claims": [], 
                            "overall_confidence": 0.0
                        }
                
                processing_time = time.time() - start_time
                
                return ChunkProcessingResponse(
                    transcription=transcription_response,
                    fact_check=fact_check_result,
                    processing_time=processing_time,
                    chunk_info={
                        "start_time": request.start_time,
                        "end_time": request.end_time,
                        "fast_mode": request.fast_mode,
                        "provider": request.provider
                    }
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
            finally:
                await stt_factory.shutdown_provider(request.provider)
                await ai_factory.shutdown()

        finally:
            # Clean up temp file
            os.unlink(temp_file.name) 