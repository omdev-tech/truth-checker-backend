"""STT endpoints for file upload and WebSocket streaming."""

import asyncio
import os
import tempfile
import subprocess
import logging
from typing import AsyncIterator, Optional
from urllib.parse import urlparse, parse_qs

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

# Set up logger for this module
logger = logging.getLogger(__name__)


class TranscriptionRequest(BaseModel):
    """Request model for transcription."""

    provider: str = "elevenlabs"
    language: Optional[str] = None


class StreamProcessingRequest(BaseModel):
    """Request model for stream URL processing."""

    url: str
    stream_type: str  # 'youtube', 'twitch', 'direct-url'
    start_time: float = 0.0
    duration: float = 300.0  # 5 minutes default
    provider: str = "elevenlabs"
    language: Optional[str] = None
    fast_mode: bool = True


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""

    text: str
    confidence: float
    language: str
    duration: float
    metadata: dict


class ChunkProcessingRequest(BaseModel):
    """Request model for chunk processing (transcription + fact-checking)."""

    provider: str = "elevenlabs"
    language: Optional[str] = None
    fast_mode: bool = True  # Use fast processing for real-time feedback
    start_time: Optional[float] = None
    end_time: Optional[float] = None


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
            stt_provider = None
            ai_provider = None
            
            try:
                stt_provider = await stt_factory.create_provider(request.provider)
                
                # Create AI provider with fast mode if requested
                ai_config = {"fast_mode": request.fast_mode} if request.fast_mode else {}
                ai_provider = await ai_factory.create_provider("chatgpt", **ai_config)
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
                        # Don't fail the entire request if fact-checking fails
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
                # Clean up providers
                if stt_provider:
                    await stt_factory.shutdown_provider(request.provider)
                if ai_provider:
                    await ai_factory.shutdown()

        finally:
            # Clean up temp file
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name) 


def extract_youtube_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/live\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def extract_twitch_channel(url: str) -> Optional[str]:
    """Extract Twitch channel name from URL."""
    import re
    match = re.search(r'twitch\.tv\/([a-zA-Z0-9_]+)', url)
    return match.group(1) if match else None


async def get_video_duration(url: str, stream_type: str) -> float:
    """Get the actual duration of a video using yt-dlp.
    
    Args:
        url: Video URL
        stream_type: Type of stream ('youtube', 'twitch', 'direct-url')
        
    Returns:
        Duration in seconds
    """
    try:
        if stream_type == 'youtube':
            video_id = extract_youtube_video_id(url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")
            
            # Use yt-dlp to get video information
            cmd = [
                'yt-dlp',
                '--quiet',
                '--no-warnings',
                '--print', 'duration',
                f'https://www.youtube.com/watch?v={video_id}'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.warning(f"⚠️ Could not get video duration: {stderr.decode()}")
                return 3600.0  # Default to 1 hour if unknown
            
            duration = float(stdout.decode().strip())
            logger.info(f"🎬 Video duration: {duration}s ({duration/60:.1f} minutes)")
            return duration
            
        else:
            # For other types, assume long duration
            return 3600.0
            
    except Exception as e:
        logger.warning(f"⚠️ Could not determine video duration: {e}")
        return 3600.0  # Default to 1 hour


async def download_stream_audio(
    url: str, 
    stream_type: str, 
    start_time: float = 0.0, 
    duration: float = 300.0
) -> str:
    """Download audio from stream URL using yt-dlp and ffmpeg.
    
    Args:
        url: Stream URL
        stream_type: Type of stream ('youtube', 'twitch', 'direct-url')
        start_time: Start time in seconds
        duration: Duration in seconds
        
    Returns:
        Path to downloaded audio file
    """
    temp_audio_file = None
    
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_audio_file = f.name
        
        if stream_type == 'youtube':
            video_id = extract_youtube_video_id(url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")
            
            # Get actual video duration first
            video_duration = await get_video_duration(url, stream_type)
            
            # Check if the requested segment is beyond video duration
            if start_time >= video_duration:
                logger.warning(f"⚠️ Segment start ({start_time}s) is beyond video duration ({video_duration}s)")
                raise ValueError(f"Segment beyond video duration: {start_time}s >= {video_duration}s")
            
            # Adjust duration if segment extends beyond video end
            if start_time + duration > video_duration:
                adjusted_duration = video_duration - start_time
                logger.info(f"📐 Adjusting segment duration from {duration}s to {adjusted_duration}s")
                duration = adjusted_duration
            
            # Use yt-dlp to extract audio stream URL
            cmd = [
                'yt-dlp',
                '--quiet',
                '--no-warnings',
                '--get-url',
                '--format', 'bestaudio/best',
                f'https://www.youtube.com/watch?v={video_id}'
            ]
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    raise RuntimeError(f"yt-dlp failed: {stderr.decode()}")
                
                audio_url = stdout.decode().strip()
                
            except FileNotFoundError:
                raise RuntimeError("yt-dlp not found. Please install yt-dlp: pip install yt-dlp")
            
        elif stream_type == 'twitch':
            channel = extract_twitch_channel(url)
            if not channel:
                raise ValueError("Invalid Twitch URL")
            
            # For Twitch, we'll use a direct stream URL approach
            # Note: This is simplified - real implementation would need Twitch API
            audio_url = f"https://twitch.tv/{channel}/stream"
            
        elif stream_type == 'direct-url':
            audio_url = url
            
        else:
            raise ValueError(f"Unsupported stream type: {stream_type}")
        
        # Use ffmpeg to extract audio segment
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', audio_url,
            '-ss', str(start_time),
            '-t', str(duration),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',  # Overwrite output file
            temp_audio_file
        ]
        
        logger.info(f"🎵 Running ffmpeg: extracting {duration}s from {start_time}s")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"💥 ffmpeg failed: {stderr.decode()}")
                raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")
                
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg")
        
        # Verify file was created and has meaningful content
        if not os.path.exists(temp_audio_file):
            raise RuntimeError("Audio file was not created")
            
        file_size = os.path.getsize(temp_audio_file)
        
        # Check for minimum file size (empty WAV files are typically ~78 bytes)
        MIN_AUDIO_SIZE = 1000  # 1KB minimum
        if file_size < MIN_AUDIO_SIZE:
            logger.error(f"❌ Audio file too small: {file_size} bytes (minimum: {MIN_AUDIO_SIZE} bytes)")
            raise RuntimeError(f"Audio file too small: {file_size} bytes - likely beyond video duration")
        
        logger.info(f"✅ Audio extracted successfully: {file_size} bytes")
        return temp_audio_file
        
    except Exception as e:
        # Clean up on error
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.unlink(temp_audio_file)
        raise e


@router.post("/process-stream", response_model=ChunkProcessingResponse)
async def process_stream_segment(
    request: StreamProcessingRequest,
) -> ChunkProcessingResponse:
    """Process a segment from a stream URL (YouTube, Twitch, etc.).
    
    Args:
        request: Stream processing request with URL and parameters
        
    Returns:
        Combined transcription and fact-check results
    """
    import time
    start_time = time.time()
    
    temp_audio_file = None
    
    try:
        # Download audio from stream
        logger.info(f"📡 Starting stream processing for {request.stream_type}")
        logger.info(f"🔗 URL: {request.url}")
        logger.info(f"⏱️ Segment: {request.start_time}s - {request.start_time + request.duration}s")
        logger.info(f"🎙️ Provider: {request.provider}")
        
        print(f"📡 Downloading audio from {request.stream_type} stream: {request.url}")
        print(f"⏱️ Segment: {request.start_time}s - {request.start_time + request.duration}s")
        
        try:
            temp_audio_file = await download_stream_audio(
                request.url,
                request.stream_type,
                request.start_time,
                request.duration
            )
        except ValueError as e:
            # Handle segments beyond video duration gracefully
            if "beyond video duration" in str(e):
                logger.info(f"⏭️ Segment beyond video duration - returning empty result")
                return ChunkProcessingResponse(
                    transcription=TranscriptionResponse(
                        text="",
                        confidence=0.0,
                        language="en",
                        duration=0.0,
                        metadata={"skip_reason": "beyond_video_duration"}
                    ),
                    fact_check={"status": "no_content", "claims": [], "overall_confidence": 0.0},
                    processing_time=time.time() - start_time,
                    chunk_info={
                        "url": request.url,
                        "stream_type": request.stream_type,
                        "start_time": request.start_time,
                        "duration": request.duration,
                        "skip_reason": "beyond_video_duration",
                        "fast_mode": request.fast_mode,
                        "provider": request.provider
                    }
                )
            else:
                raise e
        
        # Log audio file details
        file_size = os.path.getsize(temp_audio_file)
        logger.info(f"✅ Audio downloaded: {temp_audio_file}")
        logger.info(f"📊 Audio file size: {file_size} bytes")
        
        print(f"✅ Audio downloaded: {temp_audio_file} ({file_size} bytes)")
        
        # Get STT and AI providers
        stt_factory = STTProviderFactory()
        ai_factory = AIProviderFactory()
        stt_provider = None
        ai_provider = None
        
        try:
            logger.info(f"🏭 Creating STT provider: {request.provider}")
            stt_provider = await stt_factory.create_provider(request.provider)
            
            # Create AI provider with fast mode if requested
            ai_config = {"fast_mode": request.fast_mode} if request.fast_mode else {}
            ai_provider = await ai_factory.create_provider("chatgpt", **ai_config)
            logger.info(f"🤖 AI provider created with fast_mode: {request.fast_mode}")
            
        except ValueError as e:
            logger.error(f"❌ Provider creation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        
        # Transcribe the downloaded audio
        try:
            logger.info(f"🎵 Starting transcription with {request.provider}")
            
            transcription_result = await stt_provider.transcribe_file(
                temp_audio_file,
                language=request.language,
            )
            
            logger.info(f"✅ Transcription completed: {len(transcription_result.text)} characters")
            
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
                    # Don't fail the entire request if fact-checking fails
                    fact_check_result = {
                        "status": "error", 
                        "error": str(e),
                        "claims": [], 
                        "overall_confidence": 0.0
                    }
            
            processing_time = time.time() - start_time
            
            print(f"✅ Stream segment processed in {processing_time:.2f}s")
            
            return ChunkProcessingResponse(
                transcription=transcription_response,
                fact_check=fact_check_result,
                processing_time=processing_time,
                chunk_info={
                    "url": request.url,
                    "stream_type": request.stream_type,
                    "start_time": request.start_time,
                    "duration": request.duration,
                    "fast_mode": request.fast_mode,
                    "provider": request.provider
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        finally:
            # Clean up providers
            if stt_provider:
                await stt_factory.shutdown_provider(request.provider)
            if ai_provider:
                await ai_factory.shutdown()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream processing failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.unlink(temp_audio_file) 


@router.post("/video-info")
async def get_video_info(
    request: StreamProcessingRequest,
):
    """Get video information including duration.
    
    Args:
        request: Stream processing request with URL and stream type
        
    Returns:
        Video information including duration
    """
    try:
        duration = await get_video_duration(request.url, request.stream_type)
        
        # Calculate optimal number of segments
        chunk_duration = request.duration
        total_segments = max(1, int((duration + chunk_duration - 1) // chunk_duration))
        
        return {
            "duration": duration,
            "duration_formatted": f"{duration/60:.1f} minutes",
            "chunk_duration": chunk_duration,
            "total_segments": total_segments,
            "stream_type": request.stream_type,
            "url": request.url
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get video info: {str(e)}") 