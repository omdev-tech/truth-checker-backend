"""YouTube Live Detector Adapter implementing LiveStreamDetectorPort."""

import logging
import os
import re
from typing import Dict, Optional

import httpx
from pydantic import BaseModel

from ...domain.ports.live_stream_detector import (
    LiveStreamDetectorPort,
    StreamDetectionRequest,
    LiveStreamInfo,
    LiveBroadcastStatus,
    DetectionMethod
)

logger = logging.getLogger(__name__)


class YouTubeVideoInfo(BaseModel):
    """YouTube video information from Data API."""
    video_id: str
    title: str
    is_live: bool
    live_broadcast_content: str  # "live", "upcoming", "none"
    scheduled_start_time: Optional[str] = None
    actual_start_time: Optional[str] = None
    actual_end_time: Optional[str] = None
    concurrent_viewers: Optional[int] = None
    duration: Optional[str] = None


class YouTubeLiveDetectorAdapter(LiveStreamDetectorPort):
    """YouTube Live Detector Adapter using YouTube Data API v3."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with YouTube Data API key.
        
        Args:
            api_key: YouTube Data API key
        """
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            logger.warning("⚠️ YouTube API key not configured - live detection will use fallback method")
        
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.timeout = 10.0
    
    def extract_video_id(self, url: str, stream_type: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats."""
        if stream_type != 'youtube':
            return None
            
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/live\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/watch\?.*&v=([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    async def get_video_info_from_api(self, video_id: str) -> Optional[YouTubeVideoInfo]:
        """Get video information from YouTube Data API."""
        if not self.api_key:
            logger.warning("⚠️ YouTube API key not available - cannot get authoritative live status")
            return None
        
        try:
            url = f"{self.base_url}/videos"
            params = {
                'part': 'snippet,liveStreamingDetails,contentDetails',
                'id': video_id,
                'key': self.api_key
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if not data.get('items'):
                    logger.warning(f"⚠️ Video not found: {video_id}")
                    return None
                
                item = data['items'][0]
                snippet = item.get('snippet', {})
                live_details = item.get('liveStreamingDetails', {})
                content_details = item.get('contentDetails', {})
                
                # Determine live status from liveBroadcastContent
                live_broadcast_content = snippet.get('liveBroadcastContent', 'none')
                is_live = live_broadcast_content == 'live'
                
                video_info = YouTubeVideoInfo(
                    video_id=video_id,
                    title=snippet.get('title', 'Unknown'),
                    is_live=is_live,
                    live_broadcast_content=live_broadcast_content,
                    scheduled_start_time=live_details.get('scheduledStartTime'),
                    actual_start_time=live_details.get('actualStartTime'),
                    actual_end_time=live_details.get('actualEndTime'),
                    concurrent_viewers=live_details.get('concurrentViewers'),
                    duration=content_details.get('duration')
                )
                
                logger.info(f"✅ YouTube API live detection for {video_id}", {
                    'title': snippet.get('title', 'Unknown')[:50] + '...',
                    'liveBroadcastContent': live_broadcast_content,
                    'isLive': is_live,
                    'isUpcoming': live_broadcast_content == 'upcoming',
                    'hasLiveStreamingDetails': bool(live_details),
                    'actualStartTime': live_details.get('actualStartTime'),
                    'concurrentViewers': live_details.get('concurrentViewers')
                })
                
                return video_info
                
        except httpx.HTTPError as e:
            logger.error(f"❌ YouTube API HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ YouTube API unexpected error: {e}")
            return None
    
    def _analyze_url_for_live_indicators(self, url: str) -> bool:
        """Analyze URL for live indicators as fallback method."""
        url_lower = url.lower()
        
        live_indicators = [
            '/live/',
            'live=1',
            'live=true',
            '&live',
            '?live',
            '/live?',
            '/live#'
        ]
        
        return any(indicator in url_lower for indicator in live_indicators)
    
    async def detect_live_status(self, request: StreamDetectionRequest) -> LiveStreamInfo:
        """Detect live status for YouTube streams."""
        if request.stream_type != 'youtube':
            return LiveStreamInfo(
                is_live=False,
                broadcast_status=LiveBroadcastStatus.NONE,
                detection_method=DetectionMethod.FALLBACK,
                error_message=f"Unsupported stream type: {request.stream_type}"
            )
        
        # Manual override takes precedence
        if request.manual_live_override:
            logger.info("🔧 Manual live override enabled")
            return LiveStreamInfo(
                is_live=True,
                broadcast_status=LiveBroadcastStatus.LIVE,
                detection_method=DetectionMethod.MANUAL_OVERRIDE,
                video_id=self.extract_video_id(request.url, request.stream_type)
            )
        
        video_id = self.extract_video_id(request.url, request.stream_type)
        if not video_id:
            return LiveStreamInfo(
                is_live=False,
                broadcast_status=LiveBroadcastStatus.NONE,
                detection_method=DetectionMethod.FALLBACK,
                error_message='Could not extract video ID from URL'
            )
        
        # Try YouTube Data API first (most reliable)
        if self.api_key:
            video_info = await self.get_video_info_from_api(video_id)
            if video_info:
                # Convert to domain model
                broadcast_status = LiveBroadcastStatus.NONE
                if video_info.live_broadcast_content == 'live':
                    broadcast_status = LiveBroadcastStatus.LIVE
                elif video_info.live_broadcast_content == 'upcoming':
                    broadcast_status = LiveBroadcastStatus.UPCOMING
                
                return LiveStreamInfo(
                    is_live=video_info.is_live,
                    broadcast_status=broadcast_status,
                    detection_method=DetectionMethod.API_AUTHORITATIVE,
                    video_id=video_id,
                    title=video_info.title,
                    concurrent_viewers=video_info.concurrent_viewers,
                    scheduled_start_time=video_info.scheduled_start_time,
                    actual_start_time=video_info.actual_start_time,
                    actual_end_time=video_info.actual_end_time,
                    duration=video_info.duration
                )
        
        # Fallback to URL analysis
        logger.warning(f"⚠️ Using URL-based fallback for live detection: {video_id}")
        has_live_indicators = self._analyze_url_for_live_indicators(request.url)
        
        return LiveStreamInfo(
            is_live=has_live_indicators,
            broadcast_status=LiveBroadcastStatus.LIVE if has_live_indicators else LiveBroadcastStatus.NONE,
            detection_method=DetectionMethod.URL_ANALYSIS,
            video_id=video_id,
            error_message='YouTube API not available' if not self.api_key else 'API call failed'
        )
    
    async def get_stream_metadata(self, video_id: str, stream_type: str) -> Dict:
        """Get additional stream metadata."""
        if stream_type != 'youtube' or not self.api_key:
            return {}
        
        video_info = await self.get_video_info_from_api(video_id)
        if not video_info:
            return {}
        
        return {
            'title': video_info.title,
            'duration': video_info.duration,
            'concurrent_viewers': video_info.concurrent_viewers,
            'scheduled_start_time': video_info.scheduled_start_time,
            'actual_start_time': video_info.actual_start_time,
            'actual_end_time': video_info.actual_end_time,
            'thumbnail_url': f'https://img.youtube.com/vi/{video_id}/maxresdefault.jpg'
        } 