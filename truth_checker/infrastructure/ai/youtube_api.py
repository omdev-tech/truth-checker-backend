"""YouTube Data API integration for live stream detection."""

import asyncio
import logging
import os
from typing import Dict, Optional

import httpx
from pydantic import BaseModel

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


class YouTubeDataAPI:
    """YouTube Data API v3 client for live stream detection."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            logger.warning("⚠️ YouTube API key not configured - live detection will use fallback method")
        
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.timeout = 10.0
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats."""
        import re
        
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
    
    async def get_video_info(self, video_id: str) -> Optional[YouTubeVideoInfo]:
        """Get video information including live status from YouTube Data API."""
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
                
                logger.info(f"✅ YouTube API live detection for {video_id}:", {
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
    
    async def check_live_status(self, url: str) -> Dict[str, any]:
        """Check if a YouTube URL is currently live using Data API.
        
        Returns:
            Dictionary with live status information including:
            - is_live: bool
            - live_broadcast_content: str
            - method: str (detection method used)
            - video_id: str or None
            - error: str or None
        """
        video_id = self.extract_video_id(url)
        
        if not video_id:
            return {
                'is_live': False,
                'live_broadcast_content': 'none',
                'method': 'url_parsing_failed',
                'video_id': None,
                'error': 'Could not extract video ID from URL'
            }
        
        # Try YouTube Data API first (most reliable)
        if self.api_key:
            video_info = await self.get_video_info(video_id)
            if video_info:
                return {
                    'is_live': video_info.is_live,
                    'live_broadcast_content': video_info.live_broadcast_content,
                    'method': 'youtube_data_api',
                    'video_id': video_id,
                    'title': video_info.title,
                    'concurrent_viewers': video_info.concurrent_viewers,
                    'error': None
                }
        
        # Fallback to URL-based detection
        logger.warning(f"⚠️ Using URL-based fallback for live detection: {video_id}")
        url_lower = url.lower()
        
        # URL-based live indicators
        has_live_indicators = (
            '/live/' in url_lower or
            'live=' in url_lower or
            url_lower.endswith('/live') or
            '?live' in url_lower or
            '&live' in url_lower
        )
        
        return {
            'is_live': has_live_indicators,
            'live_broadcast_content': 'live' if has_live_indicators else 'none',
            'method': 'url_fallback',
            'video_id': video_id,
            'error': 'YouTube API not available' if not self.api_key else 'API call failed'
        }


# Global instance
youtube_api = YouTubeDataAPI() 