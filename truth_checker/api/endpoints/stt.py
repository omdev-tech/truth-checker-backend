"""Speech-to-text API endpoints."""

import asyncio
import logging
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from ...domain.models.claim import Claim
from ...domain.models.verification import VerificationResult, VerificationStatus, ConfidenceLevel
from ...domain.ports.live_stream_detector import LiveBroadcastStatus, DetectionMethod
from ...domain.ports.stt_provider import TranscriptionResult, AudioFormat, VideoFormat
from ...domain.services.fact_checking_service import FactCheckingService
from ...domain.services.stt_service import STTService
from ...domain.services.stream_analysis_service import StreamAnalysisService
from ...domain.services.youtube_auth_service import YouTubeAuthService
from ...infrastructure.dependencies import (
    get_fact_checking_service,
    get_stt_service,
    get_stream_analysis_service,
    get_youtube_auth_service
)
from ...infrastructure.ai.factory import AIProviderFactory
from ...infrastructure.mcp.factory import MCPProviderFactory, mcp_factory

router = APIRouter(prefix="/stt", tags=["stt"])

# Set up logger for this module
logger = logging.getLogger(__name__)


def float_to_confidence_level(confidence: float) -> ConfidenceLevel:
    """Convert float confidence to ConfidenceLevel enum."""
    if confidence >= 0.9:
        return ConfidenceLevel.HIGH
    elif confidence >= 0.7:
        return ConfidenceLevel.MEDIUM
    elif confidence >= 0.5:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.INSUFFICIENT


def convert_fact_check_to_frontend_format(fact_check_result) -> dict:
    """Convert FactCheckResult to frontend expected format."""
    if hasattr(fact_check_result, 'to_dict'):
        result_dict = fact_check_result.to_dict()
    else:
        result_dict = fact_check_result
    
    # Convert claims to frontend format
    frontend_claims = []
    if 'claims' in result_dict and result_dict['claims']:
        for claim in result_dict['claims']:
            if isinstance(claim, dict):
                # Convert confidence number to string
                claim_confidence = claim.get('confidence', 0.5)
                if isinstance(claim_confidence, (int, float)):
                    if claim_confidence >= 0.9:
                        conf_str = 'high'
                    elif claim_confidence >= 0.7:
                        conf_str = 'medium'
                    elif claim_confidence >= 0.5:
                        conf_str = 'low'
                    else:
                        conf_str = 'insufficient'
                else:
                    conf_str = str(claim_confidence).lower()
                
                frontend_claims.append({
                    'text': claim.get('claim', claim.get('text', 'Unknown claim')),
                    'status': claim.get('status', 'uncertain'),
                    'confidence': conf_str,
                    'explanation': claim.get('explanation', 'No explanation available')
                })
    
    # Determine overall status based on claims
    overall_status = 'no_text'
    if frontend_claims:
        verified_count = sum(1 for claim in frontend_claims if claim['status'] in ['verified', 'true'])
        total_claims = len(frontend_claims)
        
        if verified_count == total_claims:
            overall_status = 'true'
        elif verified_count == 0:
            overall_status = 'disputed'
        else:
            overall_status = 'partially_true'
    
    return {
        'status': overall_status,
        'claims': frontend_claims,
        'overall_confidence': result_dict.get('confidence_score', 0.0),
        'total_claims': len(frontend_claims),
        'processed_claims': len(frontend_claims),
        'error': None if frontend_claims else 'No claims extracted'
    }


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    providers: Dict[str, str]


class TranscriptionRequest(BaseModel):
    """Request model for transcription."""

    provider: str = "elevenlabs"
    language: Optional[str] = None
    fast_mode: bool = True


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

    chunk_index: int
    total_chunks: int
    start_time: float
    end_time: float
    provider: str = "elevenlabs"
    language: Optional[str] = None
    fast_mode: bool = True  # Use fast processing for real-time feedback


class StreamFactCheckResponse(BaseModel):
    """Response model for stream fact-checking (same as TextCheckResponse)."""
    
    claims: List[Claim] = Field(..., description="Extracted claims")
    results: List[VerificationResult] = Field(..., description="Verification results")


class ChunkProcessingResponse(BaseModel):
    """Response model for chunk processing."""

    chunk_index: int
    transcription: TranscriptionResponse
    fact_check: dict  # Keep as dict to match frontend expectations
    processing_time: float
    start_time: float
    end_time: float
    duration: float


class LiveStatusRequest(BaseModel):
    """Request model for live status detection."""
    url: str
    stream_type: str = "youtube"
    manual_live_override: bool = False


class LiveStatusResponse(BaseModel):
    """Response model for live status detection."""
    is_live: bool
    live_broadcast_content: str  # "live", "upcoming", "none"
    method: str  # Detection method used
    video_id: Optional[str] = None
    title: Optional[str] = None
    concurrent_viewers: Optional[int] = None
    error: Optional[str] = None


@router.get("/health", response_model=HealthStatus)
async def health_check(
    stt_service: STTService = Depends(get_stt_service)
):
    """Health check endpoint."""
    # Check STT provider
    try:
        provider_status = "available"
    except Exception as e:
        logger.error(f"STT service error: {e}")
        provider_status = "unavailable"
    
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        providers={
            "stt": provider_status,
            "fact_checker": "available",
            "stream_analysis": "available"
        }
    )


@router.post("/transcribe", response_model=ChunkProcessingResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    provider: str = Form("elevenlabs"),
    language: Optional[str] = Form(None),
    fast_mode: bool = Form(True),
    stt_service: STTService = Depends(get_stt_service),
    fact_checker: FactCheckingService = Depends(get_fact_checking_service)
):
    """Transcribe uploaded audio/video file and perform fact-checking."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        import tempfile
        import os
        import time
        
        start_time = time.time()  # Start timing
        
        # Create temporary file with appropriate extension
        file_ext = os.path.splitext(file.filename)[1] if file.filename else '.tmp'
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as f:
            temp_file = f.name
            content = await file.read()
            f.write(content)
        
        # Determine format
        if file.content_type and file.content_type.startswith('video/'):
            format_type = VideoFormat.MP4  # Default video format
        else:
            format_type = AudioFormat.WAV  # Default audio format
        
        logger.info(f"🎙️ Starting transcription for file: {file.filename}")
        
        # Process file
        result = await stt_service.transcribe_file(
            temp_file,
            format_type,
            language=language,
            fast_mode=fast_mode
        )
        
        logger.info(f"📝 Transcription completed: {len(result.text)} characters")
        
        # Fact-check transcription if there's content
        if result.text.strip():
            logger.info("🔍 Starting fact-check process...")
            fact_check_result = await fact_checker.fact_check(result.text)
            logger.info(f"✅ Fact-check completed: {fact_check_result.overall_assessment}")
            fact_check_dict = convert_fact_check_to_frontend_format(fact_check_result)
        else:
            # Empty transcription - create empty fact-check result
            fact_check_dict = {
                'status': 'no_text',
                'claims': [],
                'overall_confidence': 0.0,
                'total_claims': 0,
                'processed_claims': 0,
                'error': 'No content to fact-check'
            }
            logger.info("⚠️ No transcribable content found")
        
        processing_time = time.time() - start_time  # Calculate processing time
        
        return ChunkProcessingResponse(
            chunk_index=0,  # Single file, so index 0
            transcription=TranscriptionResponse(
                text=result.text,
                confidence=result.confidence,
                language=result.language,
                duration=result.end_time - result.start_time,
                metadata=result.metadata,
            ),
            fact_check=fact_check_dict,
            processing_time=processing_time,
            start_time=0.0,
            end_time=result.end_time - result.start_time,
            duration=result.end_time - result.start_time
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


@router.post("/process-chunk", response_model=ChunkProcessingResponse)
async def process_chunk(
    request: ChunkProcessingRequest,
    file: UploadFile = File(...),
    stt_service: STTService = Depends(get_stt_service),
    fact_checker: FactCheckingService = Depends(get_fact_checking_service)
):
    """Process a single chunk with transcription and fact-checking."""
    import time
    start_time = time.time()
    
    temp_file = None
    
    try:
        # Save uploaded chunk temporarily
        import tempfile
        import os
        
        file_ext = os.path.splitext(file.filename)[1] if file.filename else '.wav'
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as f:
            temp_file = f.name
            content = await file.read()
            f.write(content)
        
        # Determine format
        if file.content_type and file.content_type.startswith('video/'):
            format_type = VideoFormat.MP4
        else:
            format_type = AudioFormat.WAV
        
        print(f"📝 Processing chunk {request.chunk_index + 1}/{request.total_chunks}")
        print(f"⏱️ Time range: {request.start_time:.1f}s - {request.end_time:.1f}s")
        
        # Transcribe chunk
        transcription_result = await stt_service.transcribe_file(
            temp_file,
            format_type,
            language=request.language,
            fast_mode=request.fast_mode
        )
        
        print(f"📝 Transcription: {transcription_result.text[:100]}...")
        
        # Fact-check transcription
        if transcription_result.text.strip():
            fact_check_result = await fact_checker.fact_check(transcription_result.text)
            print(f"✅ Fact-check completed: {fact_check_result.overall_assessment}")
            fact_check_dict = convert_fact_check_to_frontend_format(fact_check_result)
        else:
            # Empty transcription
            fact_check_dict = {
                'status': 'no_text',
                'claims': [],
                'overall_confidence': 0.0,
                'total_claims': 0,
                'processed_claims': 0,
                'error': 'No content to fact-check'
            }
            print("⚠️ No transcribable content in this chunk")
        
        processing_time = time.time() - start_time
        
        return ChunkProcessingResponse(
            chunk_index=request.chunk_index,
            transcription=TranscriptionResponse(
                text=transcription_result.text,
                confidence=transcription_result.confidence,
                language=transcription_result.language,
                duration=transcription_result.end_time - transcription_result.start_time,
                metadata=transcription_result.metadata,
            ),
            fact_check=fact_check_dict,
            processing_time=processing_time,
            start_time=request.start_time,
            end_time=request.end_time,
            duration=request.end_time - request.start_time
        )
        
    except Exception as e:
        logger.error(f"Chunk processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


@router.post("/transcribe-chunk", response_model=ChunkProcessingResponse)
async def transcribe_chunk(
    file: UploadFile = File(...),
    provider: str = Form("elevenlabs"),
    language: Optional[str] = Form(None),
    fast_mode: bool = Form(True),
    start_time: float = Form(0.0),
    end_time: float = Form(0.0),
    chunk_index: int = Form(0),
    total_chunks: int = Form(1),
    stt_service: STTService = Depends(get_stt_service)
):
    """Alias endpoint for transcribe-chunk (compatibility with frontend API).
    
    This endpoint processes a chunk with transcription and fact-checking,
    using the same robust logic as stream processing.
    """
    import time
    start_time_processing = time.time()
    
    temp_file = None
    
    try:
        # Save uploaded chunk temporarily
        import tempfile
        import os
        
        file_ext = os.path.splitext(file.filename)[1] if file.filename else '.wav'
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as f:
            temp_file = f.name
            content = await file.read()
            f.write(content)
        
        # Determine format
        if file.content_type and file.content_type.startswith('video/'):
            format_type = VideoFormat.MP4
        else:
            format_type = AudioFormat.WAV
        
        print(f"📝 Processing chunk {chunk_index + 1}/{total_chunks}")
        print(f"⏱️ Time range: {start_time:.1f}s - {end_time:.1f}s")
        
        # Transcribe chunk
        transcription_result = await stt_service.transcribe_file(
            temp_file,
            format_type,
            language=language,
            fast_mode=fast_mode
        )
        
        print(f"📝 Transcription: {transcription_result.text[:100]}...")
        
        # Fact-check transcription if there's content (using same logic as stream processing)
        if transcription_result.text.strip():
            print("🔍 Starting fact-check...")
            
            try:
                # Use the same approach as the working /fact-check/text endpoint
                logger.info("Getting AI provider...")
                ai_factory = AIProviderFactory()
                ai_provider = ai_factory.get_provider("chatgpt")
                if ai_provider is None:
                    logger.info("AI provider not found, creating new one...")
                    ai_provider = await ai_factory.create_provider("chatgpt")
                logger.info("AI provider ready")
                
                # Get or create MCP provider for fact verification
                logger.info("Getting MCP provider...")
                mcp_provider = mcp_factory.get_provider("wikipedia")
                if mcp_provider is None:
                    logger.info("MCP provider not found, creating new one...")
                    mcp_provider = await mcp_factory.create_provider("wikipedia")
                logger.info("MCP provider ready")
                
                # Set language if specified
                if language and language != "en":
                    logger.info(f"Setting language to {language}")
                    await mcp_provider.set_language(language)
                
                # Extract claims from text
                logger.info("Extracting claims from text...")
                claims = await ai_provider.analyze_text(transcription_result.text, None)
                logger.info(f"Found {len(claims)} claims")
                
                # Verify each claim (same logic as working endpoint)
                results = []
                for i, claim in enumerate(claims):
                    logger.info(f"Verifying claim {i+1}/{len(claims)}: {claim.text}")
                    
                    # Get AI verification
                    logger.info("Getting AI verification...")
                    ai_result = await ai_provider.verify_claim(claim, None)
                    logger.info(f"AI verification complete: {ai_result.status}")
                    
                    # Get MCP validation
                    logger.info("Getting MCP validation...")
                    mcp_result = await mcp_provider.validate_fact(claim.text)
                    logger.info(f"MCP validation complete: {mcp_result.is_valid}")
                    
                    # Combine results (prefer MCP when available)
                    if mcp_result.is_valid:
                        # Create new result with combined data
                        combined_sources = ai_result.sources + mcp_result.source_urls
                        final_result = VerificationResult(
                            claim_text=ai_result.claim_text,
                            status=ai_result.status,
                            confidence=float_to_confidence_level(mcp_result.confidence),
                            explanation=ai_result.explanation,
                            sources=combined_sources,
                            timestamp=ai_result.timestamp,
                            metadata=ai_result.metadata
                        )
                        results.append(final_result)
                    else:
                        results.append(ai_result)
                
                # Convert to frontend format
                claims_data = []
                total_confidence = 0.0
                status_counts = {
                    'true': 0, 'false': 0, 'uncertain': 0, 'unverifiable': 0,
                    'partially_true': 0, 'disputed': 0, 'misleading': 0
                }
                
                for i, (claim, result) in enumerate(zip(claims, results)):
                    # Convert VerificationStatus to string
                    status_str = result.status.value.lower() if hasattr(result.status, 'value') else str(result.status).lower()
                    if status_str == 'unverifiable':
                        status_str = 'uncertain'
                    
                    # Convert ConfidenceLevel to percentage string
                    confidence_str = result.confidence.value.lower() if hasattr(result.confidence, 'value') else str(result.confidence).lower()
                    
                    claims_data.append({
                        'text': claim.text,
                        'status': status_str,
                        'confidence': confidence_str,
                        'explanation': result.explanation or f"This claim was assessed as {status_str}"
                    })
                    
                    # Count for overall status (handle all possible status types)
                    if status_str in status_counts:
                        status_counts[status_str] += 1
                    else:
                        # Handle any unknown status types as uncertain
                        status_counts['uncertain'] += 1
                    
                    # Convert confidence to float for averaging
                    confidence_float = 0.5  # default
                    if confidence_str == 'high':
                        confidence_float = 0.9
                    elif confidence_str == 'medium':
                        confidence_float = 0.7
                    elif confidence_str == 'low':
                        confidence_float = 0.5
                    else:
                        confidence_float = 0.3
                    
                    total_confidence += confidence_float
                
                # Calculate overall confidence and status
                overall_confidence = total_confidence / len(claims) if len(claims) > 0 else 0.0
                
                # Determine overall status using weighted scoring system
                total_claims = len(claims)
                if total_claims == 0:
                    overall_status = 'no_text'
                else:
                    # Calculate weighted scores for different status types
                    positive_score = status_counts.get('true', 0) * 1.0 + status_counts.get('partially_true', 0) * 0.7
                    negative_score = status_counts.get('false', 0) * 1.0 + status_counts.get('misleading', 0) * 0.8
                    neutral_score = (
                        status_counts.get('uncertain', 0) * 0.3 + 
                        status_counts.get('unverifiable', 0) * 0.3 + 
                        status_counts.get('disputed', 0) * 0.5
                    )
                    
                    # Determine overall status based on weighted scores
                    if positive_score > negative_score and positive_score > neutral_score:
                        overall_status = 'true'
                    elif negative_score > positive_score and negative_score > neutral_score:
                        overall_status = 'false'
                    elif total_claims == status_counts.get('uncertain', 0) + status_counts.get('unverifiable', 0):
                        overall_status = 'not_checkable'
                    else:
                        overall_status = 'uncertain'
                
                fact_check_response = {
                    'status': overall_status,
                    'claims': claims_data,
                    'overall_confidence': overall_confidence,
                    'total_claims': len(claims),
                    'processed_claims': len(claims)
                }
                
                print(f"✅ Fact-check completed: {len(claims)} claims, {len(results)} results")
                print(f"📊 Overall: {overall_status} ({overall_confidence:.1%} confidence)")
                
            except Exception as e:
                logger.error(f"❌ Error in chunk fact-checking: {type(e).__name__}: {str(e)}", exc_info=True)
                # Fallback response on error
                fact_check_response = {
                    'status': 'error',
                    'claims': [{
                        'text': transcription_result.text[:100] + ('...' if len(transcription_result.text) > 100 else ''),
                        'status': 'error',
                        'confidence': 'insufficient',
                        'explanation': f"Error occurred during fact-checking: {str(e)}"
                    }],
                    'overall_confidence': 0.0,
                    'total_claims': 1,
                    'processed_claims': 0,
                    'error': str(e)
                }
                print(f"⚠️ Fact-check failed, using fallback response")
        else:
            # Empty transcription
            fact_check_response = {
                'status': 'no_text',
                'claims': [],
                'overall_confidence': 0.0,
                'total_claims': 0,
                'processed_claims': 0
            }
            print("⚠️ No transcribable content in this chunk")
        
        processing_time = time.time() - start_time_processing
        
        print(f"✅ Chunk processed in {processing_time:.2f}s")
        
        return ChunkProcessingResponse(
            chunk_index=chunk_index,
            transcription=TranscriptionResponse(
                text=transcription_result.text,
                confidence=transcription_result.confidence,
                language=transcription_result.language,
                duration=transcription_result.end_time - transcription_result.start_time,
                metadata=transcription_result.metadata,
            ),
            fact_check=fact_check_response,
            processing_time=processing_time,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time
        )
        
    except Exception as e:
        logger.error(f"Chunk processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


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


async def get_video_duration(url: str, stream_type: str, youtube_auth_service: YouTubeAuthService) -> float:
    """Get the actual duration of a video using yt-dlp with proper authentication.

    Args:
        url: Video URL
        stream_type: Type of stream ('youtube', 'twitch', 'direct-url')
        youtube_auth_service: YouTube authentication service

    Returns:
        Duration in seconds (returns -1 for live streams)
    """
    try:
        if stream_type == 'youtube':
            video_id = extract_youtube_video_id(url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")
            
            # Ensure we have valid YouTube authentication
            auth_result = await youtube_auth_service.ensure_valid_authentication()
            if not auth_result.success:
                logger.warning(f"⚠️ YouTube authentication failed: {auth_result.error_message}")
                logger.info("🔄 Proceeding without authentication - may have limited access")
            
            # Use yt-dlp with authentication to get video information
            cmd = ['yt-dlp', '--quiet', '--no-warnings'] + youtube_auth_service.get_yt_dlp_args() + [
                '--print', 'duration',
                f'https://www.youtube.com/watch?v={video_id}'
            ]
            
            logger.debug(f"🎬 Running yt-dlp duration check: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode()
                logger.warning(f"⚠️ Could not get video duration: {error_msg}")
                
                # Check if it's a bot detection error
                if "Sign in to confirm you're not a bot" in error_msg:
                    logger.error("🤖 YouTube bot detection triggered - authentication service may need refresh")
                    # Try to refresh authentication
                    await youtube_auth_service.authenticate()
                    raise RuntimeError(f"YouTube authentication required: {error_msg}")
                
                return 3600.0  # Default to 1 hour if unknown
            
            duration_str = stdout.decode().strip()
            
            # Handle live streams that return "NA" or similar
            if duration_str.upper() in ['NA', 'N/A', '', 'NONE', 'NULL']:
                logger.info(f"🔴 Live stream detected via duration: {duration_str}")
                return -1.0  # Special value for live streams
            
            try:
                duration = float(duration_str)
                logger.info(f"🎬 Video duration: {duration}s ({duration/60:.1f} minutes)")
                return duration
            except ValueError:
                logger.warning(f"⚠️ Could not parse duration: '{duration_str}' - treating as live stream")
                return -1.0  # Treat unparseable duration as live stream
            
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
    duration: float = 300.0,
    youtube_auth_service: YouTubeAuthService = None
) -> str:
    """Download audio from stream URL using yt-dlp with proper authentication.

    Args:
        url: Stream URL
        stream_type: Type of stream ('youtube', 'twitch', 'direct-url')
        start_time: Start time in seconds
        duration: Duration in seconds
        youtube_auth_service: YouTube authentication service

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
            video_duration = await get_video_duration(url, stream_type, youtube_auth_service)
            
            # Only check duration limits for non-live streams (video_duration > 0)
            if video_duration > 0:
                # Check if the requested segment is beyond video duration
                if start_time >= video_duration:
                    logger.warning(f"⚠️ Segment start ({start_time}s) is beyond video duration ({video_duration}s)")
                    raise ValueError(f"Segment beyond video duration: {start_time}s >= {video_duration}s")
                
                # Adjust duration if segment extends beyond video end
                if start_time + duration > video_duration:
                    adjusted_duration = video_duration - start_time
                    logger.info(f"📐 Adjusting segment duration from {duration}s to {adjusted_duration}s")
                    duration = adjusted_duration
            else:
                # Live stream detected (video_duration == -1.0)
                logger.info(f"🔴 Live stream detected - skipping duration validation")
                logger.info(f"🎵 Processing live audio: {duration}s from current position")
            
            # Ensure we have valid YouTube authentication
            if youtube_auth_service:
                auth_result = await youtube_auth_service.ensure_valid_authentication()
                if not auth_result.success:
                    logger.warning(f"⚠️ YouTube authentication failed: {auth_result.error_message}")
                    logger.info("🔄 Proceeding without authentication - may have limited access")
                yt_dlp_auth_args = youtube_auth_service.get_yt_dlp_args()
            else:
                logger.warning("⚠️ No YouTube authentication service provided - using legacy config")
                yt_dlp_auth_args = []
            
            # Use yt-dlp with authentication to extract audio stream URL
            cmd = ['yt-dlp', '--quiet', '--no-warnings'] + yt_dlp_auth_args + [
                '--get-url',
                '--format', 'bestaudio/best',
                f'https://www.youtube.com/watch?v={video_id}'
            ]
            
            logger.debug(f"🎵 Running yt-dlp for audio extraction: {' '.join(cmd)}")
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    error_msg = stderr.decode()
                    logger.error(f"💥 yt-dlp failed: {error_msg}")
                    
                    # Provide specific guidance for bot detection
                    if "Sign in to confirm you're not a bot" in error_msg:
                        logger.error("🤖 YouTube bot detection triggered")
                        if youtube_auth_service:
                            logger.info("🔄 Attempting to refresh authentication...")
                            auth_result = await youtube_auth_service.authenticate()
                            if auth_result.success:
                                logger.info("✅ Authentication refreshed, please retry")
                            else:
                                logger.error("❌ Authentication refresh failed")
                        logger.error("🔧 To fix this, configure cookies in your .env file:")
                        logger.error("   YT_DLP_COOKIES_FROM_BROWSER=firefox")
                        logger.error("   (or chrome, safari, etc.)")
                        
                    raise RuntimeError(f"yt-dlp failed: {error_msg}")
                
                audio_url = stdout.decode().strip()
                logger.info(f"✅ Audio stream URL extracted successfully")
                
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
    stt_service: STTService = Depends(get_stt_service),
    youtube_auth_service: YouTubeAuthService = Depends(get_youtube_auth_service)
) -> ChunkProcessingResponse:
    """Process a segment from a stream URL (YouTube, Twitch, etc.).

    Args:
        request: Stream processing request with URL and parameters
        stt_service: Speech-to-text service
        youtube_auth_service: YouTube authentication service
        
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
                request.duration,
                youtube_auth_service
            )
        except ValueError as e:
            # Handle segments beyond video duration gracefully
            if "beyond video duration" in str(e):
                logger.info(f"⏭️ Segment beyond video duration - returning empty result")
                return ChunkProcessingResponse(
                    chunk_index=0,
                    transcription=TranscriptionResponse(
                        text="",
                        confidence=0.0,
                        language="en",
                        duration=0.0,
                        metadata={"skip_reason": "beyond_video_duration"}
                    ),
                    fact_check={},
                    processing_time=time.time() - start_time,
                    start_time=request.start_time,
                    end_time=request.start_time + request.duration,
                    duration=request.duration
                )
            else:
                raise e
        
        # Log audio file details
        file_size = os.path.getsize(temp_audio_file)
        logger.info(f"✅ Audio downloaded: {temp_audio_file}")
        logger.info(f"📊 Audio file size: {file_size} bytes")
        
        print(f"✅ Audio downloaded: {temp_audio_file} ({file_size} bytes)")
        
        # Transcribe audio
        print("🎙️ Starting transcription...")
        transcription_result = await stt_service.transcribe_file(
            temp_audio_file,
            AudioFormat.WAV,
            language=request.language,
            fast_mode=request.fast_mode
        )
        
        print(f"📝 Transcription: {transcription_result.text[:100]}...")
        
        # Fact-check transcription if there's content
        if transcription_result.text.strip():
            print("🔍 Starting fact-check...")
            
            try:
                # Use the same approach as the working /fact-check/text endpoint
                logger.info("Getting AI provider...")
                ai_factory = AIProviderFactory()
                ai_provider = ai_factory.get_provider("chatgpt")
                if ai_provider is None:
                    logger.info("AI provider not found, creating new one...")
                    ai_provider = await ai_factory.create_provider("chatgpt")
                logger.info("AI provider ready")
                
                # Get or create MCP provider for fact verification
                logger.info("Getting MCP provider...")
                mcp_provider = mcp_factory.get_provider("wikipedia")
                if mcp_provider is None:
                    logger.info("MCP provider not found, creating new one...")
                    mcp_provider = await mcp_factory.create_provider("wikipedia")
                logger.info("MCP provider ready")
                
                # Set language if specified
                if request.language and request.language != "en":
                    logger.info(f"Setting language to {request.language}")
                    await mcp_provider.set_language(request.language)
                
                # Extract claims from text
                logger.info("Extracting claims from text...")
                claims = await ai_provider.analyze_text(transcription_result.text, None)
                logger.info(f"Found {len(claims)} claims")
                
                # Verify each claim (same logic as working endpoint)
                results = []
                for i, claim in enumerate(claims):
                    logger.info(f"Verifying claim {i+1}/{len(claims)}: {claim.text}")
                    
                    # Get AI verification
                    logger.info("Getting AI verification...")
                    ai_result = await ai_provider.verify_claim(claim, None)
                    logger.info(f"AI verification complete: {ai_result.status}")
                    
                    # Get MCP validation
                    logger.info("Getting MCP validation...")
                    mcp_result = await mcp_provider.validate_fact(claim.text)
                    logger.info(f"MCP validation complete: {mcp_result.is_valid}")
                    
                    # Combine results (prefer MCP when available)
                    if mcp_result.is_valid:
                        # Create new result with combined data
                        combined_sources = ai_result.sources + mcp_result.source_urls
                        final_result = VerificationResult(
                            claim_text=ai_result.claim_text,
                            status=ai_result.status,
                            confidence=float_to_confidence_level(mcp_result.confidence),
                            explanation=ai_result.explanation,
                            sources=combined_sources,
                            timestamp=ai_result.timestamp,
                            metadata=ai_result.metadata
                        )
                        results.append(final_result)
                    else:
                        results.append(ai_result)
                
                # Convert to frontend format
                claims_data = []
                total_confidence = 0.0
                status_counts = {
                    'true': 0, 'false': 0, 'uncertain': 0, 'unverifiable': 0,
                    'partially_true': 0, 'disputed': 0, 'misleading': 0
                }
                
                for i, (claim, result) in enumerate(zip(claims, results)):
                    # Convert VerificationStatus to string
                    status_str = result.status.value.lower() if hasattr(result.status, 'value') else str(result.status).lower()
                    if status_str == 'unverifiable':
                        status_str = 'uncertain'
                    
                    # Convert ConfidenceLevel to percentage string
                    confidence_str = result.confidence.value.lower() if hasattr(result.confidence, 'value') else str(result.confidence).lower()
                    
                    claims_data.append({
                        'text': claim.text,
                        'status': status_str,
                        'confidence': confidence_str,
                        'explanation': result.explanation or f"This claim was assessed as {status_str}"
                    })
                    
                    # Count for overall status (handle all possible status types)
                    if status_str in status_counts:
                        status_counts[status_str] += 1
                    else:
                        # Handle any unknown status types as uncertain
                        status_counts['uncertain'] += 1
                    
                    # Convert confidence to float for averaging
                    confidence_float = 0.5  # default
                    if confidence_str == 'high':
                        confidence_float = 0.9
                    elif confidence_str == 'medium':
                        confidence_float = 0.7
                    elif confidence_str == 'low':
                        confidence_float = 0.5
                    else:
                        confidence_float = 0.3
                    
                    total_confidence += confidence_float
                
                # Calculate overall confidence and status
                overall_confidence = total_confidence / len(claims) if len(claims) > 0 else 0.0
                
                # Determine overall status using weighted scoring system
                total_claims = len(claims)
                if total_claims == 0:
                    overall_status = 'no_text'
                else:
                    # Calculate weighted scores for different status types
                    positive_score = status_counts.get('true', 0) * 1.0 + status_counts.get('partially_true', 0) * 0.7
                    negative_score = status_counts.get('false', 0) * 1.0 + status_counts.get('misleading', 0) * 0.8
                    neutral_score = (
                        status_counts.get('uncertain', 0) * 0.3 + 
                        status_counts.get('unverifiable', 0) * 0.3 + 
                        status_counts.get('disputed', 0) * 0.5
                    )
                    
                    # Determine overall status based on weighted scores
                    if positive_score > negative_score and positive_score > neutral_score:
                        overall_status = 'true'
                    elif negative_score > positive_score and negative_score > neutral_score:
                        overall_status = 'false'
                    elif total_claims == status_counts.get('uncertain', 0) + status_counts.get('unverifiable', 0):
                        overall_status = 'not_checkable'
                    else:
                        overall_status = 'uncertain'
                
                fact_check_response = {
                    'status': overall_status,
                    'claims': claims_data,
                    'overall_confidence': overall_confidence,
                    'total_claims': len(claims),
                    'processed_claims': len(claims)
                }
                
                print(f"✅ Fact-check completed: {len(claims)} claims, {len(results)} results")
                print(f"📊 Overall: {overall_status} ({overall_confidence:.1%} confidence)")
                
            except Exception as e:
                logger.error(f"❌ Error in stream fact-checking: {type(e).__name__}: {str(e)}", exc_info=True)
                # Fallback response on error
                fact_check_response = {
                    'status': 'error',
                    'claims': [{
                        'text': transcription_result.text[:100] + ('...' if len(transcription_result.text) > 100 else ''),
                        'status': 'error',
                        'confidence': 'insufficient',
                        'explanation': f"Error occurred during fact-checking: {str(e)}"
                    }],
                    'overall_confidence': 0.0,
                    'total_claims': 1,
                    'processed_claims': 0,
                    'error': str(e)
                }
                print(f"⚠️ Fact-check failed, using fallback response")
        else:
            # Empty transcription
            fact_check_response = {
                'status': 'no_text',
                'claims': [],
                'overall_confidence': 0.0,
                'total_claims': 0,
                'processed_claims': 0
            }
            print("⚠️ No transcribable content in this segment")
        
        processing_time = time.time() - start_time
        
        print(f"✅ Stream segment processed in {processing_time:.2f}s")
        
        return ChunkProcessingResponse(
            chunk_index=0,  # Single segment
            transcription=TranscriptionResponse(
                text=transcription_result.text,
                confidence=transcription_result.confidence,
                language=transcription_result.language,
                duration=transcription_result.end_time - transcription_result.start_time,
                metadata=transcription_result.metadata,
            ),
            fact_check=fact_check_response,
            processing_time=processing_time,
            start_time=request.start_time,
            end_time=request.start_time + request.duration,
            duration=request.duration
        )
        
    except Exception as e:
        logger.error(f"Stream processing error: {e}")
        
        # Special handling for duration-related errors
        if "beyond video duration" in str(e):
            raise HTTPException(
                status_code=400, 
                detail=f"Segment beyond video duration: {str(e)}"
            )
        
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary audio file
        if temp_audio_file and os.path.exists(temp_audio_file):
            os.unlink(temp_audio_file)


@router.post("/check-live-status", response_model=LiveStatusResponse)
async def check_live_status(
    request: LiveStatusRequest,
    stream_analysis_service: StreamAnalysisService = Depends(get_stream_analysis_service)
) -> LiveStatusResponse:
    """Check if a stream URL is currently live using authoritative APIs.
    
    This endpoint provides the most reliable way to detect live streams
    by using official APIs (YouTube Data API, etc.) rather than URL parsing.
    """
    try:
        # Use domain service for stream analysis
        stream_info = await stream_analysis_service.analyze_stream(
            url=request.url,
            stream_type=request.stream_type,
            manual_live_override=request.manual_live_override
        )
        
        # Convert domain model to API response
        return LiveStatusResponse(
            is_live=stream_info.is_live,
            live_broadcast_content=stream_info.broadcast_status.value,
            method=stream_info.detection_method.value,
            video_id=stream_info.video_id,
            title=stream_info.title,
            concurrent_viewers=stream_info.concurrent_viewers,
            error=stream_info.error_message
        )
            
    except Exception as e:
        logger.error(f"Live status check error: {e}")
        return LiveStatusResponse(
            is_live=False,
            live_broadcast_content='none',
            method='error_fallback',
            video_id=None,
            error=str(e)
        )


@router.post("/video-info")
async def get_video_info(
    request: StreamProcessingRequest,
    stream_analysis_service: StreamAnalysisService = Depends(get_stream_analysis_service),
    youtube_auth_service: YouTubeAuthService = Depends(get_youtube_auth_service)
):
    """Get video information including duration and live status.

    Args:
        request: Stream processing request with URL and stream type
        stream_analysis_service: Stream analysis service
        youtube_auth_service: YouTube authentication service
        
    Returns:
        Video information including duration, live status, and metadata
    """
    try:
        # Get live status using domain service first
        stream_info = await stream_analysis_service.analyze_stream(
            url=request.url,
            stream_type=request.stream_type
        )
        
        # Get duration with authentication
        duration = await get_video_duration(request.url, request.stream_type, youtube_auth_service)
        
        # If stream analysis says it's live OR duration detection says it's live
        is_live_stream = (
            stream_analysis_service.is_live_processing_recommended(stream_info) or
            duration == -1.0
        )
        
        if is_live_stream:
            logger.info(f"🔴 Live stream confirmed - using live processing mode")
            duration = -1.0  # Ensure duration is set to live indicator
            chunk_duration = stream_analysis_service.get_recommended_chunk_duration(stream_info)
            
            response = {
                "duration": -1,  # Special value for live streams
                "duration_formatted": "Live Stream",
                "chunk_duration": chunk_duration,
                "total_segments": -1,  # Infinite segments for live
                "stream_type": request.stream_type,
                "url": request.url,
                "is_live": True,
                "live_status": {
                    "is_live": stream_info.is_live,
                    "live_broadcast_content": stream_info.broadcast_status.value,
                    "method": stream_info.detection_method.value,
                    "title": stream_info.title,
                    "concurrent_viewers": stream_info.concurrent_viewers,
                    "error": stream_info.error_message
                },
                "note": "Live stream detected - use real-time processing mode",
                "recommended_chunk_duration": chunk_duration,
                "processing_mode": "live",
                "authentication_status": {
                    "is_authenticated": youtube_auth_service.is_authenticated,
                    "method": youtube_auth_service.current_method.value if youtube_auth_service.current_method else None,
                    "provider": youtube_auth_service.provider_name
                }
            }
            
        else:
            # Regular video processing
            if duration <= 0:
                duration = 3600.0  # Fallback for regular videos
            
            chunk_duration = request.duration
            total_segments = max(1, int((duration + chunk_duration - 1) // chunk_duration))
            
            response = {
                "duration": duration,
                "duration_formatted": f"{duration/60:.1f} minutes",
                "chunk_duration": chunk_duration,
                "total_segments": total_segments,
                "stream_type": request.stream_type,
                "url": request.url,
                "is_live": False,
                "live_status": {
                    "is_live": stream_info.is_live,
                    "live_broadcast_content": stream_info.broadcast_status.value,
                    "method": stream_info.detection_method.value,
                    "title": stream_info.title,
                    "concurrent_viewers": stream_info.concurrent_viewers,
                    "error": stream_info.error_message
                },
                "processing_mode": "regular",
                "authentication_status": {
                    "is_authenticated": youtube_auth_service.is_authenticated,
                    "method": youtube_auth_service.current_method.value if youtube_auth_service.current_method else None,
                    "provider": youtube_auth_service.provider_name
                }
            }
        
        logger.info(f"📊 Video info response: duration={response['duration']}, is_live={response['is_live']}, mode={response['processing_mode']}")
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get video info: {str(e)}")


@router.get("/youtube-auth-status")
async def youtube_auth_status(
    youtube_auth_service: YouTubeAuthService = Depends(get_youtube_auth_service)
):
    """Check YouTube authentication status for monitoring."""
    try:
        # Test current authentication
        is_working = await youtube_auth_service.test_authentication()
        
        # Get available methods
        available_methods = youtube_auth_service.get_available_auth_methods()
        
        # If not working, try to authenticate
        if not is_working:
            auth_result = await youtube_auth_service.authenticate()
            is_working = auth_result.success
        
        return {
            "status": "healthy" if is_working else "degraded",
            "authenticated": is_working,
            "current_method": youtube_auth_service.current_method.value if youtube_auth_service.current_method else None,
            "available_methods": [method.value for method in available_methods],
            "provider": youtube_auth_service.provider_name,
            "timestamp": datetime.now().isoformat(),
            "details": {
                "can_access_youtube": is_working,
                "fallback_methods_available": len(available_methods) > 1
            }
        }
        
    except Exception as e:
        logger.error(f"YouTube auth status check failed: {e}")
        return {
            "status": "error",
            "authenticated": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 