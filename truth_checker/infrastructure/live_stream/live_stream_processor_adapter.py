"""Infrastructure adapter for live stream processing."""

import os
import time
import asyncio
import logging
from typing import Dict, Optional, AsyncGenerator
from datetime import datetime

from ...domain.ports.live_stream_processor import LiveStreamProcessorPort
from ...domain.models.live_stream_segment import LiveStreamSegment, LiveStreamSession, SegmentStatus
from ...domain.services.stt_service import STTService
from ...domain.services.fact_checking_service import FactCheckingService
from ..stt.stream_downloader import download_stream_audio

logger = logging.getLogger(__name__)


class LiveStreamProcessorAdapter(LiveStreamProcessorPort):
    """Infrastructure adapter for live stream processing.
    
    This adapter implements the live stream processor port using
    existing STT and fact-checking services.
    """
    
    def __init__(
        self,
        stt_service: STTService,
        fact_checking_service: FactCheckingService,
        max_concurrent_sessions: int = 3
    ):
        """Initialize adapter with required services.
        
        Args:
            stt_service: STT service for transcription
            fact_checking_service: Fact checking service
            max_concurrent_sessions: Maximum concurrent sessions
        """
        self._stt_service = stt_service
        self._fact_checking_service = fact_checking_service
        self._max_concurrent_sessions = max_concurrent_sessions
        self._active_sessions: Dict[str, LiveStreamSession] = {}
        self._result_queues: Dict[str, asyncio.Queue] = {}
    
    async def record_segment(
        self,
        stream_url: str,
        stream_type: str,
        start_time: float,
        duration: float,
        segment_id: int
    ) -> str:
        """Record a segment from a live stream.
        
        Args:
            stream_url: URL of the live stream
            stream_type: Type of stream (youtube, twitch, etc.)
            start_time: Start time in seconds from stream beginning
            duration: Duration of segment in seconds
            segment_id: Unique identifier for the segment
            
        Returns:
            Path to the recorded audio file
        """
        logger.debug(f"🎙️ Recording segment {segment_id}: {start_time:.1f}s-{start_time + duration:.1f}s")
        
        try:
            # Use existing download_stream_audio function
            audio_file_path = await download_stream_audio(
                url=stream_url,
                stream_type=stream_type,
                start_time=start_time,
                duration=int(duration)
            )
            
            logger.debug(f"✅ Segment {segment_id} recorded: {audio_file_path}")
            return audio_file_path
            
        except Exception as e:
            logger.error(f"❌ Failed to record segment {segment_id}: {e}")
            raise e
    
    async def process_segment_async(
        self,
        segment: LiveStreamSegment,
        audio_file_path: str
    ) -> LiveStreamSegment:
        """Process a segment asynchronously (transcription + fact-checking).
        
        Args:
            segment: The segment to process
            audio_file_path: Path to the audio file
            
        Returns:
            Updated segment with processing results
        """
        start_time = time.time()
        
        try:
            logger.debug(f"🔄 Processing segment {segment.segment_id}")
            segment.mark_processing()
            
            # Step 1: Transcription
            logger.debug(f"📝 Transcribing segment {segment.segment_id}")
            transcription_result = await self._stt_service.transcribe_file(
                file_path=audio_file_path,
                format_type="audio",  # Assuming audio format
                language="en",
                fast_mode=True
            )
            
            if not transcription_result.text.strip():
                logger.warning(f"⚠️ No text transcribed for segment {segment.segment_id}")
                segment.mark_error("No text transcribed")
                return segment
            
            # Step 2: Fact checking
            logger.debug(f"🔍 Fact-checking segment {segment.segment_id}")
            fact_check_result = await self._fact_checking_service.fact_check(
                transcription_result.text
            )
            
            # Convert fact check result to dict format
            fact_check_dict = self._convert_fact_check_to_frontend_format(fact_check_result)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Mark segment as completed
            segment.mark_completed(
                transcription_text=transcription_result.text,
                transcription_confidence=transcription_result.confidence,
                fact_check_result=fact_check_dict,
                processing_time=processing_time
            )
            
            logger.debug(f"✅ Segment {segment.segment_id} processed in {processing_time:.2f}s")
            
            # Add to result queue if session has one
            session_id = self._find_session_for_segment(segment)
            if session_id and session_id in self._result_queues:
                await self._result_queues[session_id].put(segment)
            
            return segment
            
        except Exception as e:
            logger.error(f"❌ Failed to process segment {segment.segment_id}: {e}")
            segment.mark_error(str(e))
            
            # Add error to result queue
            session_id = self._find_session_for_segment(segment)
            if session_id and session_id in self._result_queues:
                await self._result_queues[session_id].put(segment)
            
            return segment
    
    async def cleanup_segment_file(self, audio_file_path: str) -> None:
        """Clean up temporary audio file.
        
        Args:
            audio_file_path: Path to the audio file to clean up
        """
        try:
            if os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
                logger.debug(f"🗑️ Cleaned up audio file: {audio_file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup file {audio_file_path}: {e}")
    
    async def start_continuous_processing(
        self,
        session: LiveStreamSession
    ) -> AsyncGenerator[LiveStreamSegment, None]:
        """Start continuous processing of a live stream.
        
        Args:
            session: Live stream session configuration
            
        Yields:
            Processed segments as they complete
        """
        logger.info(f"📡 Starting continuous processing for session: {session.session_id}")
        
        # Store session
        self._active_sessions[session.session_id] = session
        
        # Create result queue for this session
        result_queue = asyncio.Queue()
        self._result_queues[session.session_id] = result_queue
        
        try:
            while session.is_active:
                try:
                    # Wait for results with timeout
                    segment = await asyncio.wait_for(result_queue.get(), timeout=1.0)
                    yield segment
                except asyncio.TimeoutError:
                    # Continue waiting if no results yet
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Continuous processing failed for session {session.session_id}: {e}")
            raise e
        finally:
            # Cleanup
            self._result_queues.pop(session.session_id, None)
            logger.info(f"📡 Stopped continuous processing for session: {session.session_id}")
    
    async def stop_processing(self, session_id: str) -> None:
        """Stop processing for a specific session.
        
        Args:
            session_id: ID of the session to stop
        """
        logger.info(f"🛑 Stopping processing for session: {session_id}")
        
        # Mark session as inactive
        session = self._active_sessions.get(session_id)
        if session:
            session.stop_session()
        
        # Clean up result queue
        self._result_queues.pop(session_id, None)
        
        logger.info(f"✅ Processing stopped for session: {session_id}")
    
    async def get_session_status(self, session_id: str) -> Optional[LiveStreamSession]:
        """Get status of a processing session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Session status or None if not found
        """
        return self._active_sessions.get(session_id)
    
    async def list_active_sessions(self) -> Dict[str, LiveStreamSession]:
        """List all active processing sessions.
        
        Returns:
            Dictionary of session_id -> session
        """
        return self._active_sessions.copy()
    
    def _convert_fact_check_to_frontend_format(self, fact_check_result) -> Dict:
        """Convert FactCheckResult to frontend-expected format.
        
        Args:
            fact_check_result: FactCheckResult from domain service
            
        Returns:
            Dictionary in frontend format
        """
        try:
            # Use the to_dict method if available
            if hasattr(fact_check_result, 'to_dict'):
                base_dict = fact_check_result.to_dict()
            else:
                base_dict = {
                    'overall_assessment': getattr(fact_check_result, 'overall_assessment', ''),
                    'claims': getattr(fact_check_result, 'claims', []),
                    'sources': getattr(fact_check_result, 'sources', []),
                    'confidence_score': getattr(fact_check_result, 'confidence_score', 0.0)
                }
            
            # Convert to frontend format
            claims_list = []
            for claim in base_dict.get('claims', []):
                if isinstance(claim, dict):
                    claims_list.append({
                        'text': claim.get('text', ''),
                        'status': claim.get('status', 'uncertain'),
                        'confidence': claim.get('confidence', 'low'),
                        'explanation': claim.get('explanation', '')
                    })
            
            # Determine overall status
            overall_status = 'uncertain'
            if base_dict.get('confidence_score', 0) > 0.7:
                overall_status = 'true'
            elif base_dict.get('confidence_score', 0) < 0.3:
                overall_status = 'false'
            
            return {
                'status': overall_status,
                'claims': claims_list,
                'overall_confidence': base_dict.get('confidence_score', 0.0),
                'total_claims': len(claims_list),
                'processed_claims': len(claims_list),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to convert fact check result: {e}")
            return {
                'status': 'error',
                'claims': [],
                'overall_confidence': 0.0,
                'total_claims': 0,
                'processed_claims': 0,
                'error': str(e)
            }
    
    def _find_session_for_segment(self, segment: LiveStreamSegment) -> Optional[str]:
        """Find session ID for a segment.
        
        Args:
            segment: Segment to find session for
            
        Returns:
            Session ID or None if not found
        """
        for session_id, session in self._active_sessions.items():
            if session.stream_url == segment.stream_url:
                return session_id
        return None
    
    @property
    def is_available(self) -> bool:
        """Check if the processor is available."""
        return (
            self._stt_service.is_available and
            len(self._active_sessions) < self._max_concurrent_sessions
        )
    
    @property
    def max_concurrent_sessions(self) -> int:
        """Maximum number of concurrent sessions supported."""
        return self._max_concurrent_sessions 