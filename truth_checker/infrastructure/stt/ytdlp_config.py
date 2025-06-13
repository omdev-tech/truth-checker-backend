"""yt-dlp configuration management for YouTube authentication."""

import logging
import os
from typing import List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class YtDlpConfig(BaseModel):
    """Configuration for yt-dlp operations."""
    
    enable_cookies: bool = True
    cookies_from_browser: Optional[str] = None  # 'firefox', 'chrome', 'safari', etc.
    cookies_file: Optional[str] = None  # Path to cookies.txt file
    timeout: float = 30.0
    retries: int = 3
    
    @classmethod
    def from_env(cls) -> "YtDlpConfig":
        """Create configuration from environment variables."""
        enable_cookies = os.getenv('YT_DLP_ENABLE_COOKIES', 'true').lower() == 'true'
        cookies_from_browser = os.getenv('YT_DLP_COOKIES_FROM_BROWSER')
        cookies_file = os.getenv('YT_DLP_COOKIES_FILE')
        
        # Log configuration status
        if enable_cookies:
            if cookies_from_browser:
                logger.info(f"🍪 yt-dlp configured to extract cookies from browser: {cookies_from_browser}")
            elif cookies_file:
                logger.info(f"🍪 yt-dlp configured to use cookies file: {cookies_file}")
            else:
                logger.warning("⚠️ yt-dlp cookies enabled but no source specified - consider setting YT_DLP_COOKIES_FILE or YT_DLP_COOKIES_FROM_BROWSER")
        else:
            logger.info("🚫 yt-dlp cookies disabled")
        
        return cls(
            enable_cookies=enable_cookies,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file
        )
    
    def get_cookie_args(self) -> List[str]:
        """Get yt-dlp cookie arguments based on configuration.
        
        NOTE: This method is now deprecated in favor of using YouTubeAuthService.
        Use YouTubeAuthService.get_yt_dlp_args() for better authentication handling.
        """
        args = []
        
        if not self.enable_cookies:
            return args
        
        if self.cookies_from_browser:
            args.extend(['--cookies-from-browser', self.cookies_from_browser])
            logger.debug(f"🍪 Adding browser cookie args: --cookies-from-browser {self.cookies_from_browser}")
        elif self.cookies_file and os.path.exists(self.cookies_file):
            args.extend(['--cookies', self.cookies_file])
            logger.debug(f"🍪 Adding cookie file args: --cookies {self.cookies_file}")
        elif self.cookies_file:
            logger.warning(f"⚠️ Cookie file specified but not found: {self.cookies_file}")
        
        return args
    
    def get_base_args(self) -> List[str]:
        """Get base yt-dlp arguments with authentication.
        
        NOTE: This method is now deprecated in favor of using YouTubeAuthService.
        Use YouTubeAuthService.ensure_valid_authentication() for better authentication handling.
        """
        args = [
            'yt-dlp',
            '--quiet',
            '--no-warnings'
        ]
        
        # Add cookie arguments
        cookie_args = self.get_cookie_args()
        args.extend(cookie_args)
        
        return args


# Global instance - initialized from environment
ytdlp_config = YtDlpConfig.from_env() 


# Compatibility function for existing code
def get_ytdlp_auth_args() -> List[str]:
    """Get yt-dlp authentication arguments for backward compatibility.
    
    Returns:
        List of yt-dlp authentication arguments
    """
    logger.warning("⚠️ get_ytdlp_auth_args() is deprecated - use YouTubeAuthService instead")
    return ytdlp_config.get_cookie_args() 