"""Port for live stream processing operations."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Dict, Any
from ..models.live_stream_segment import LiveStreamSegment, LiveStreamSession


class LiveStreamProcessorPort(ABC):
    """Port for live stream processing operations.
    
    This port defines the interface for recording and processing
    live stream segments continuously.
    """
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def cleanup_segment_file(self, audio_file_path: str) -> None:
        """Clean up temporary audio file.
        
        Args:
            audio_file_path: Path to the audio file to clean up
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def stop_processing(self, session_id: str) -> None:
        """Stop processing for a specific session.
        
        Args:
            session_id: ID of the session to stop
        """
        pass
    
    @abstractmethod
    async def get_session_status(self, session_id: str) -> Optional[LiveStreamSession]:
        """Get status of a processing session.
        
        Args:
            session_id: ID of the session
            
        Returns:
            Session status or None if not found
        """
        pass
    
    @abstractmethod
    async def list_active_sessions(self) -> Dict[str, LiveStreamSession]:
        """List all active processing sessions.
        
        Returns:
            Dictionary of session_id -> session
        """
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the processor is available."""
        pass
    
    @property
    @abstractmethod
    def max_concurrent_sessions(self) -> int:
        """Maximum number of concurrent sessions supported."""
        pass 