"""API endpoints for live stream processing."""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...domain.services.live_stream_processing_service import LiveStreamProcessingService
from ...infrastructure.dependencies import get_live_stream_processing_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/live-stream", tags=["live-stream"])


# Request/Response Models
class StartLiveStreamRequest(BaseModel):
    """Request to start live stream processing."""
    url: str = Field(..., description="Live stream URL")
    stream_type: str = Field(..., description="Type of stream (youtube, twitch, etc.)")
    segment_duration: float = Field(30.0, description="Duration of each segment in seconds")
    overlap_duration: float = Field(5.0, description="Overlap between segments in seconds")


class LiveStreamSessionResponse(BaseModel):
    """Response for live stream session operations."""
    session_id: str
    stream_url: str
    stream_type: str
    started_at: str
    is_active: bool
    configuration: Dict[str, Any]
    statistics: Dict[str, Any]
    stream_metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class LiveStreamSegmentResponse(BaseModel):
    """Response for live stream segment."""
    segment_id: int
    stream_url: str
    start_time: float
    end_time: float
    duration: float
    status: str
    created_at: str
    transcription: Optional[Dict[str, Any]] = None
    fact_check: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class LiveStreamStatusResponse(BaseModel):
    """Response for live stream status."""
    active_sessions: Dict[str, LiveStreamSessionResponse]
    total_active_sessions: int
    service_available: bool
    max_concurrent_sessions: int


@router.post("/start", response_model=LiveStreamSessionResponse)
async def start_live_stream(
    request: StartLiveStreamRequest,
    background_tasks: BackgroundTasks,
    live_stream_service: LiveStreamProcessingService = Depends(get_live_stream_processing_service)
) -> LiveStreamSessionResponse:
    """Start continuous live stream processing.
    
    Args:
        request: Live stream start request
        background_tasks: FastAPI background tasks
        live_stream_service: Live stream processing service
        
    Returns:
        Created live stream session
    """
    logger.info(f"🔴 Starting live stream processing: {request.stream_type} - {request.url[:50]}...")
    
    try:
        # Generate unique session ID
        session_id = f"live_{int(time.time())}_{hash(request.url) % 10000}"
        
        # Start live stream processing
        session = await live_stream_service.start_live_stream_processing(
            session_id=session_id,
            stream_url=request.url,
            stream_type=request.stream_type,
            segment_duration=request.segment_duration,
            overlap_duration=request.overlap_duration
        )
        
        logger.info(f"✅ Live stream processing started: {session_id}")
        
        # Convert to response model
        return LiveStreamSessionResponse(
            session_id=session.session_id,
            stream_url=session.stream_url,
            stream_type=session.stream_type,
            started_at=session.started_at.isoformat(),
            is_active=session.is_active,
            configuration={
                'segment_duration': session.segment_duration,
                'overlap_duration': session.overlap_duration,
                'segment_interval': session.segment_interval
            },
            statistics={
                'total_segments': session.total_segments,
                'completed_segments': session.completed_segments,
                'processing_segments': session.processing_segments,
                'error_segments': session.error_segments,
                'success_rate': session.success_rate
            },
            stream_metadata=session.stream_metadata,
            error_message=session.error_message
        )
        
    except ValueError as e:
        logger.error(f"❌ Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"❌ Service unavailable: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Failed to start live stream processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")


@router.post("/stop/{session_id}", response_model=LiveStreamSessionResponse)
async def stop_live_stream(
    session_id: str,
    live_stream_service: LiveStreamProcessingService = Depends(get_live_stream_processing_service)
) -> LiveStreamSessionResponse:
    """Stop live stream processing for a specific session.
    
    Args:
        session_id: ID of the session to stop
        live_stream_service: Live stream processing service
        
    Returns:
        Stopped live stream session
    """
    logger.info(f"🛑 Stopping live stream processing: {session_id}")
    
    try:
        session = await live_stream_service.stop_live_stream_processing(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        logger.info(f"✅ Live stream processing stopped: {session_id}")
        
        # Convert to response model
        return LiveStreamSessionResponse(
            session_id=session.session_id,
            stream_url=session.stream_url,
            stream_type=session.stream_type,
            started_at=session.started_at.isoformat(),
            is_active=session.is_active,
            configuration={
                'segment_duration': session.segment_duration,
                'overlap_duration': session.overlap_duration,
                'segment_interval': session.segment_interval
            },
            statistics={
                'total_segments': session.total_segments,
                'completed_segments': session.completed_segments,
                'processing_segments': session.processing_segments,
                'error_segments': session.error_segments,
                'success_rate': session.success_rate
            },
            stream_metadata=session.stream_metadata,
            error_message=session.error_message
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to stop live stream processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop processing: {str(e)}")


@router.get("/status/{session_id}", response_model=LiveStreamSessionResponse)
async def get_session_status(
    session_id: str,
    live_stream_service: LiveStreamProcessingService = Depends(get_live_stream_processing_service)
) -> LiveStreamSessionResponse:
    """Get status of a live stream processing session.
    
    Args:
        session_id: ID of the session
        live_stream_service: Live stream processing service
        
    Returns:
        Session status
    """
    try:
        session = await live_stream_service.get_session_status(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Convert to response model
        return LiveStreamSessionResponse(
            session_id=session.session_id,
            stream_url=session.stream_url,
            stream_type=session.stream_type,
            started_at=session.started_at.isoformat(),
            is_active=session.is_active,
            configuration={
                'segment_duration': session.segment_duration,
                'overlap_duration': session.overlap_duration,
                'segment_interval': session.segment_interval
            },
            statistics={
                'total_segments': session.total_segments,
                'completed_segments': session.completed_segments,
                'processing_segments': session.processing_segments,
                'error_segments': session.error_segments,
                'success_rate': session.success_rate
            },
            stream_metadata=session.stream_metadata,
            error_message=session.error_message
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get session status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/status", response_model=LiveStreamStatusResponse)
async def get_live_stream_status(
    live_stream_service: LiveStreamProcessingService = Depends(get_live_stream_processing_service)
) -> LiveStreamStatusResponse:
    """Get overall live stream processing status.
    
    Args:
        live_stream_service: Live stream processing service
        
    Returns:
        Overall status
    """
    try:
        active_sessions = await live_stream_service.list_active_sessions()
        
        # Convert sessions to response models
        session_responses = {}
        for session_id, session in active_sessions.items():
            session_responses[session_id] = LiveStreamSessionResponse(
                session_id=session.session_id,
                stream_url=session.stream_url,
                stream_type=session.stream_type,
                started_at=session.started_at.isoformat(),
                is_active=session.is_active,
                configuration={
                    'segment_duration': session.segment_duration,
                    'overlap_duration': session.overlap_duration,
                    'segment_interval': session.segment_interval
                },
                statistics={
                    'total_segments': session.total_segments,
                    'completed_segments': session.completed_segments,
                    'processing_segments': session.processing_segments,
                    'error_segments': session.error_segments,
                    'success_rate': session.success_rate
                },
                stream_metadata=session.stream_metadata,
                error_message=session.error_message
            )
        
        return LiveStreamStatusResponse(
            active_sessions=session_responses,
            total_active_sessions=len(active_sessions),
            service_available=live_stream_service.is_available,
            max_concurrent_sessions=live_stream_service.max_concurrent_sessions
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get live stream status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/results/{session_id}")
async def get_live_stream_results(
    session_id: str,
    live_stream_service: LiveStreamProcessingService = Depends(get_live_stream_processing_service)
):
    """Get live stream processing results as Server-Sent Events.
    
    Args:
        session_id: ID of the session
        live_stream_service: Live stream processing service
        
    Returns:
        Streaming response with segment results
    """
    logger.info(f"📡 Starting result stream for session: {session_id}")
    
    async def generate_results():
        """Generate Server-Sent Events for live stream results."""
        try:
            async for segment in live_stream_service.get_processing_results(session_id):
                # Convert segment to response format
                segment_response = LiveStreamSegmentResponse(
                    segment_id=segment.segment_id,
                    stream_url=segment.stream_url,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    duration=segment.duration,
                    status=segment.status.value,
                    created_at=segment.created_at.isoformat(),
                    transcription={
                        'text': segment.transcription_text,
                        'confidence': segment.transcription_confidence
                    } if segment.transcription_text else None,
                    fact_check=segment.fact_check_result,
                    processing_time=segment.processing_time,
                    error_message=segment.error_message,
                    metadata=segment.metadata or {}
                )
                
                # Send as Server-Sent Event
                yield f"data: {segment_response.model_dump_json()}\n\n"
                
        except Exception as e:
            logger.error(f"❌ Result stream failed for session {session_id}: {e}")
            error_data = {"error": str(e), "session_id": session_id}
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate_results(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


# Configuration endpoint
@router.get("/config")
async def get_live_stream_config() -> Dict[str, Any]:
    """Get live stream processing configuration.
    
    Returns:
        Configuration settings
    """
    return {
        "default_segment_duration": 30.0,
        "default_overlap_duration": 5.0,
        "default_segment_interval": 25.0,
        "max_concurrent_sessions": 3,
        "supported_stream_types": ["youtube", "twitch", "direct-url"],
        "supported_formats": ["mp4", "webm", "m4a", "mp3"],
        "processing_timeout": 60.0
    } 