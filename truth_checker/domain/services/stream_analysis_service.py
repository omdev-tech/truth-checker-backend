"""Domain service for stream analysis."""

import logging
from typing import Dict, Optional

from ..ports.live_stream_detector import (
    LiveStreamDetectorPort,
    StreamDetectionRequest,
    LiveStreamInfo,
    LiveBroadcastStatus,
    DetectionMethod
)

logger = logging.getLogger(__name__)


class StreamAnalysisService:
    """Domain service for analyzing streams and detecting live status."""
    
    def __init__(self, live_stream_detector: LiveStreamDetectorPort):
        """Initialize service with live stream detector.
        
        Args:
            live_stream_detector: Live stream detector port implementation
        """
        self._live_stream_detector = live_stream_detector
    
    async def analyze_stream(
        self, 
        url: str, 
        stream_type: str, 
        manual_live_override: bool = False
    ) -> LiveStreamInfo:
        """Analyze a stream to determine its live status and metadata.
        
        Args:
            url: Stream URL
            stream_type: Type of stream (youtube, twitch, etc.)
            manual_live_override: Manual override for live detection
            
        Returns:
            Complete live stream information
        """
        logger.info(f"🔍 Analyzing stream: {stream_type} - {url[:50]}...")
        
        request = StreamDetectionRequest(
            url=url,
            stream_type=stream_type,
            manual_live_override=manual_live_override
        )
        
        try:
            stream_info = await self._live_stream_detector.detect_live_status(request)
            
            logger.info(f"✅ Stream analysis completed:", {
                'is_live': stream_info.is_live,
                'broadcast_status': stream_info.broadcast_status.value,
                'detection_method': stream_info.detection_method.value,
                'video_id': stream_info.video_id,
                'title': stream_info.title[:50] + '...' if stream_info.title else None
            })
            
            return stream_info
            
        except Exception as e:
            logger.error(f"❌ Stream analysis failed: {e}")
            
            # Return fallback result
            return LiveStreamInfo(
                is_live=False,
                broadcast_status=LiveBroadcastStatus.NONE,
                detection_method=DetectionMethod.FALLBACK,
                error_message=str(e)
            )
    
    def extract_stream_id(self, url: str, stream_type: str) -> Optional[str]:
        """Extract stream ID from URL.
        
        Args:
            url: Stream URL
            stream_type: Type of stream
            
        Returns:
            Stream ID or None
        """
        return self._live_stream_detector.extract_video_id(url, stream_type)
    
    async def get_stream_metadata(self, video_id: str, stream_type: str) -> Dict:
        """Get additional metadata for a stream.
        
        Args:
            video_id: Stream ID
            stream_type: Type of stream
            
        Returns:
            Stream metadata
        """
        try:
            return await self._live_stream_detector.get_stream_metadata(video_id, stream_type)
        except Exception as e:
            logger.error(f"❌ Failed to get stream metadata: {e}")
            return {}
    
    def is_live_processing_recommended(self, stream_info: LiveStreamInfo) -> bool:
        """Determine if live processing mode is recommended.
        
        Args:
            stream_info: Stream information
            
        Returns:
            True if live processing is recommended
        """
        return (
            stream_info.is_live or
            stream_info.broadcast_status == LiveBroadcastStatus.LIVE or
            stream_info.detection_method == DetectionMethod.MANUAL_OVERRIDE
        )
    
    def get_recommended_chunk_duration(self, stream_info: LiveStreamInfo) -> int:
        """Get recommended chunk duration based on stream type.
        
        Args:
            stream_info: Stream information
            
        Returns:
            Recommended chunk duration in seconds
        """
        if self.is_live_processing_recommended(stream_info):
            return 30  # Shorter chunks for live streams
        else:
            return 45  # Standard chunks for recorded content 