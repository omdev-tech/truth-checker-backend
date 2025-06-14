"""Domain service for live stream processing."""

import logging
import asyncio
from typing import Dict, Optional, AsyncGenerator
from datetime import datetime

from ..ports.live_stream_processor import LiveStreamProcessorPort
from ..models.live_stream_segment import LiveStreamSegment, LiveStreamSession, SegmentStatus

logger = logging.getLogger(__name__)


class LiveStreamProcessingService:
    """Domain service for continuous live stream processing.
    
    This service coordinates live stream processing using the live stream
    processor port. It manages sessions, segments, and provides a clean
    interface for the application layer.
    """
    
    def __init__(self, live_stream_processor: LiveStreamProcessorPort):
        """Initialize service with live stream processor.
        
        Args:
            live_stream_processor: Live stream processor port implementation
        """
        self._processor = live_stream_processor
        self._active_sessions: Dict[str, LiveStreamSession] = {}
        self._processing_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_live_stream_processing(
        self,
        session_id: str,
        stream_url: str,
        stream_type: str,
        segment_duration: float = 30.0,
        overlap_duration: float = 5.0
    ) -> LiveStreamSession:
        """Start continuous processing of a live stream.
        
        Args:
            session_id: Unique identifier for the session
            stream_url: URL of the live stream
            stream_type: Type of stream (youtube, twitch, etc.)
            segment_duration: Duration of each segment in seconds
            overlap_duration: Overlap between segments in seconds
            
        Returns:
            Created live stream session
        """
        logger.info(f"🔴 Starting live stream processing: {session_id}")
        logger.info(f"📺 Stream: {stream_type} - {stream_url[:50]}...")
        logger.info(f"⚙️ Config: duration={segment_duration}s, overlap={overlap_duration}s")
        
        # Check if session already exists
        if session_id in self._active_sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        # Check processor availability
        if not self._processor.is_available:
            raise RuntimeError("Live stream processor is not available")
        
        # Create session
        session = LiveStreamSession(
            session_id=session_id,
            stream_url=stream_url,
            stream_type=stream_type,
            started_at=datetime.now(),
            segment_duration=segment_duration,
            overlap_duration=overlap_duration,
            segment_interval=segment_duration - overlap_duration
        )
        
        # Store session
        self._active_sessions[session_id] = session
        
        # Start processing task
        task = asyncio.create_task(
            self._continuous_processing_loop(session)
        )
        self._processing_tasks[session_id] = task
        
        logger.info(f"✅ Live stream processing started: {session_id}")
        return session
    
    async def stop_live_stream_processing(self, session_id: str) -> Optional[LiveStreamSession]:
        """Stop processing for a specific session.
        
        Args:
            session_id: ID of the session to stop
            
        Returns:
            Stopped session or None if not found
        """
        logger.info(f"🛑 Stopping live stream processing: {session_id}")
        
        session = self._active_sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None
        
        # Mark session as inactive
        session.stop_session()
        
        # Cancel processing task
        task = self._processing_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Clean up
        self._processing_tasks.pop(session_id, None)
        
        # Stop processor
        await self._processor.stop_processing(session_id)
        
        logger.info(f"✅ Live stream processing stopped: {session_id}")
        logger.info(f"📊 Final stats: {session.completed_segments}/{session.total_segments} completed")
        
        return session
    
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
    
    async def get_processing_results(
        self,
        session_id: str
    ) -> AsyncGenerator[LiveStreamSegment, None]:
        """Get processing results for a session as they complete.
        
        Args:
            session_id: ID of the session
            
        Yields:
            Completed segments
        """
        session = self._active_sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return
        
        logger.info(f"📡 Starting result stream for session: {session_id}")
        
        async for segment in self._processor.start_continuous_processing(session):
            # Update session statistics
            if segment.is_completed:
                session.mark_segment_completed()
                logger.info(f"✅ Segment {segment.segment_id} completed: {len(segment.transcription_text or '')} chars")
            elif segment.has_error:
                session.mark_segment_error()
                logger.error(f"❌ Segment {segment.segment_id} failed: {segment.error_message}")
            
            yield segment
    
    async def _continuous_processing_loop(self, session: LiveStreamSession) -> None:
        """Internal continuous processing loop for a session.
        
        Args:
            session: Live stream session to process
        """
        logger.info(f"🔄 Starting continuous processing loop: {session.session_id}")
        
        try:
            while session.is_active:
                # Create next segment
                segment_id = session.next_segment_id()
                start_time = session.calculate_segment_start_time(segment_id)
                
                segment = LiveStreamSegment(
                    segment_id=segment_id,
                    stream_url=session.stream_url,
                    start_time=start_time,
                    duration=session.segment_duration,
                    status=SegmentStatus.PENDING,
                    created_at=datetime.now(),
                    overlap_duration=session.overlap_duration
                )
                
                logger.info(f"📝 Creating segment {segment_id}: {start_time:.1f}s-{segment.end_time:.1f}s")
                
                # Record segment (this should be fast)
                try:
                    audio_file = await self._processor.record_segment(
                        stream_url=session.stream_url,
                        stream_type=session.stream_type,
                        start_time=start_time,
                        duration=session.segment_duration,
                        segment_id=segment_id
                    )
                    
                    segment.mark_recording(audio_file)
                    
                    # Process segment asynchronously (fire and forget)
                    asyncio.create_task(
                        self._process_segment_background(segment, audio_file)
                    )
                    
                except Exception as e:
                    logger.error(f"❌ Failed to record segment {segment_id}: {e}")
                    segment.mark_error(str(e))
                    session.mark_segment_error()
                
                # Wait for next segment interval
                await asyncio.sleep(session.segment_interval)
                
        except asyncio.CancelledError:
            logger.info(f"🛑 Processing loop cancelled: {session.session_id}")
            raise
        except Exception as e:
            logger.error(f"❌ Processing loop failed: {session.session_id} - {e}")
            session.stop_session(str(e))
            raise
    
    async def _process_segment_background(
        self,
        segment: LiveStreamSegment,
        audio_file_path: str
    ) -> None:
        """Process a segment in the background.
        
        Args:
            segment: Segment to process
            audio_file_path: Path to the audio file
        """
        try:
            logger.debug(f"🔄 Processing segment {segment.segment_id} in background")
            
            # Process segment (transcription + fact-checking)
            processed_segment = await self._processor.process_segment_async(
                segment, audio_file_path
            )
            
            logger.debug(f"✅ Background processing completed: segment {segment.segment_id}")
            
        except Exception as e:
            logger.error(f"❌ Background processing failed: segment {segment.segment_id} - {e}")
            segment.mark_error(str(e))
        finally:
            # Always cleanup audio file
            try:
                await self._processor.cleanup_segment_file(audio_file_path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup file {audio_file_path}: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the live stream processing service."""
        logger.info("🔄 Shutting down live stream processing service...")
        
        # Stop all active sessions
        session_ids = list(self._active_sessions.keys())
        for session_id in session_ids:
            await self.stop_live_stream_processing(session_id)
        
        # Cancel any remaining tasks
        for task in self._processing_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks.values(), return_exceptions=True)
        
        self._active_sessions.clear()
        self._processing_tasks.clear()
        
        logger.info("✅ Live stream processing service shutdown completed")
    
    @property
    def active_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._active_sessions)
    
    @property
    def is_available(self) -> bool:
        """Check if the service is available."""
        return self._processor.is_available
    
    @property
    def max_concurrent_sessions(self) -> int:
        """Maximum number of concurrent sessions supported."""
        return self._processor.max_concurrent_sessions 