"""Domain port for live stream detection."""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum


class LiveBroadcastStatus(Enum):
    """Live broadcast status enumeration."""
    LIVE = "live"
    UPCOMING = "upcoming"
    NONE = "none"


class DetectionMethod(Enum):
    """Detection method enumeration."""
    API_AUTHORITATIVE = "api_authoritative"
    URL_ANALYSIS = "url_analysis"
    MANUAL_OVERRIDE = "manual_override"
    FALLBACK = "fallback"


@dataclass
class LiveStreamInfo:
    """Live stream information."""
    is_live: bool
    broadcast_status: LiveBroadcastStatus
    detection_method: DetectionMethod
    video_id: Optional[str] = None
    title: Optional[str] = None
    concurrent_viewers: Optional[int] = None
    scheduled_start_time: Optional[str] = None
    actual_start_time: Optional[str] = None
    actual_end_time: Optional[str] = None
    duration: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class StreamDetectionRequest:
    """Stream detection request."""
    url: str
    stream_type: str
    manual_live_override: bool = False


class LiveStreamDetectorPort(ABC):
    """Port for live stream detection."""
    
    @abstractmethod
    async def detect_live_status(self, request: StreamDetectionRequest) -> LiveStreamInfo:
        """Detect if a stream is currently live.
        
        Args:
            request: Stream detection request
            
        Returns:
            Live stream information
        """
        pass
    
    @abstractmethod
    def extract_video_id(self, url: str, stream_type: str) -> Optional[str]:
        """Extract video/stream ID from URL.
        
        Args:
            url: Stream URL
            stream_type: Type of stream (youtube, twitch, etc.)
            
        Returns:
            Video/stream ID or None if not found
        """
        pass
    
    @abstractmethod
    async def get_stream_metadata(self, video_id: str, stream_type: str) -> Dict:
        """Get additional stream metadata.
        
        Args:
            video_id: Video/stream ID
            stream_type: Type of stream
            
        Returns:
            Stream metadata
        """
        pass 