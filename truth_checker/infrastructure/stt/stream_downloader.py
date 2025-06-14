"""Stream audio downloader utility."""

import os
import asyncio
import tempfile
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_youtube_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    import re
    
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
        r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def extract_twitch_channel(url: str) -> Optional[str]:
    """Extract Twitch channel from URL."""
    import re
    
    match = re.search(r'twitch\.tv/([a-zA-Z0-9_]+)', url)
    return match.group(1) if match else None


async def get_video_duration(url: str, stream_type: str) -> float:
    """Get video duration using yt-dlp."""
    try:
        if stream_type == 'youtube':
            video_id = extract_youtube_video_id(url)
            if not video_id:
                return 3600.0  # Default fallback
            
            cmd = ['yt-dlp', '--quiet', '--no-warnings', '--get-duration', 
                   f'https://www.youtube.com/watch?v={video_id}']
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                duration_str = stdout.decode().strip()
                # Parse duration (format: HH:MM:SS or MM:SS or SS)
                parts = duration_str.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                elif len(parts) == 2:  # MM:SS
                    return int(parts[0]) * 60 + int(parts[1])
                else:  # SS
                    return int(parts[0])
            
    except Exception as e:
        logger.warning(f"⚠️ Could not determine video duration: {e}")
    
    return 3600.0  # Default to 1 hour


async def download_stream_audio(
    url: str, 
    stream_type: str, 
    start_time: float = 0.0, 
    duration: float = 300.0
) -> str:
    """Download audio from stream URL using yt-dlp.

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
            
            # Use yt-dlp to extract audio stream URL
            cmd = ['yt-dlp', '--quiet', '--no-warnings',
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