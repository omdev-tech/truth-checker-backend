"""YouTube authentication adapter implementing YouTubeAuthProvider."""

import asyncio
import logging
import os
import platform
from typing import Dict, List, Optional

from ...domain.ports.youtube_auth_provider import (
    YouTubeAuthProvider,
    AuthMethod,
    AuthResult,
    AuthCredentials
)

logger = logging.getLogger(__name__)


class YouTubeAuthAdapter(YouTubeAuthProvider):
    """YouTube authentication adapter with multiple strategy support.
    
    This adapter implements various YouTube authentication methods
    for yt-dlp operations, prioritizing security and reliability.
    """
    
    def __init__(self):
        """Initialize the YouTube authentication adapter."""
        self._browser_priority = ['firefox', 'chrome', 'safari', 'edge', 'opera']
        self._test_video_id = "jNQXAC9IVRw"  # A public YouTube video for testing
    
    async def authenticate(self, preferred_method: Optional[AuthMethod] = None) -> AuthResult:
        """Attempt to authenticate with YouTube using available methods."""
        logger.info("🔐 Starting YouTube authentication...")
        
        # Determine authentication order
        methods_to_try = self._get_authentication_order(preferred_method)
        
        for method in methods_to_try:
            logger.info(f"🔧 Trying authentication method: {method}")
            
            try:
                if method == AuthMethod.BROWSER_COOKIES:
                    result = await self._try_browser_cookies()
                elif method == AuthMethod.COOKIES_FILE:
                    result = await self._try_cookies_file()
                elif method == AuthMethod.API_KEY:
                    result = await self._try_api_key()
                elif method == AuthMethod.NO_AUTH:
                    result = await self._try_no_auth()
                else:
                    continue
                
                if result.success:
                    # Test the authentication
                    if await self.test_youtube_access(result):
                        logger.info(f"✅ Authentication successful with {method}")
                        return result
                    else:
                        logger.warning(f"⚠️ Authentication {method} failed YouTube access test")
                
            except Exception as e:
                logger.warning(f"⚠️ Authentication method {method} failed: {e}")
                continue
        
        # All methods failed
        return AuthResult(
            success=False,
            method_used=AuthMethod.NO_AUTH,
            error_message="All authentication methods failed"
        )
    
    async def _try_browser_cookies(self) -> AuthResult:
        """Try to authenticate using browser cookies."""
        # Check environment configuration first
        browser_name = os.getenv('YT_DLP_COOKIES_FROM_BROWSER')
        
        if browser_name:
            logger.info(f"🍪 Using configured browser: {browser_name}")
            return await self._test_browser_cookies(browser_name)
        
        # Try browsers in priority order
        for browser in self._browser_priority:
            if self._is_browser_available(browser):
                logger.info(f"🍪 Trying browser: {browser}")
                result = await self._test_browser_cookies(browser)
                if result.success:
                    return result
        
        return AuthResult(
            success=False,
            method_used=AuthMethod.BROWSER_COOKIES,
            error_message="No browsers with valid cookies found"
        )
    
    async def _test_browser_cookies(self, browser_name: str) -> AuthResult:
        """Test browser cookie authentication."""
        try:
            yt_dlp_args = ['--cookies-from-browser', browser_name]
            
            # Test with a simple yt-dlp command
            cmd = ['yt-dlp', '--quiet', '--no-warnings'] + yt_dlp_args + [
                '--print', 'title',
                f'https://www.youtube.com/watch?v={self._test_video_id}'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and stdout.decode().strip():
                return AuthResult(
                    success=True,
                    method_used=AuthMethod.BROWSER_COOKIES,
                    yt_dlp_args=yt_dlp_args,
                    metadata={'browser': browser_name}
                )
            else:
                error_msg = stderr.decode().strip()
                return AuthResult(
                    success=False,
                    method_used=AuthMethod.BROWSER_COOKIES,
                    error_message=f"Browser {browser_name} failed: {error_msg}"
                )
                
        except Exception as e:
            return AuthResult(
                success=False,
                method_used=AuthMethod.BROWSER_COOKIES,
                error_message=f"Browser {browser_name} error: {str(e)}"
            )
    
    async def _try_cookies_file(self) -> AuthResult:
        """Try to authenticate using cookies file."""
        cookies_file = os.getenv('YT_DLP_COOKIES_FILE')
        
        if not cookies_file:
            return AuthResult(
                success=False,
                method_used=AuthMethod.COOKIES_FILE,
                error_message="No cookies file specified in YT_DLP_COOKIES_FILE"
            )
        
        if not os.path.exists(cookies_file):
            return AuthResult(
                success=False,
                method_used=AuthMethod.COOKIES_FILE,
                error_message=f"Cookies file not found: {cookies_file}"
            )
        
        try:
            yt_dlp_args = ['--cookies', cookies_file]
            
            # Test with a simple yt-dlp command
            cmd = ['yt-dlp', '--quiet', '--no-warnings'] + yt_dlp_args + [
                '--print', 'title',
                f'https://www.youtube.com/watch?v={self._test_video_id}'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and stdout.decode().strip():
                return AuthResult(
                    success=True,
                    method_used=AuthMethod.COOKIES_FILE,
                    yt_dlp_args=yt_dlp_args,
                    metadata={'cookies_file': cookies_file}
                )
            else:
                error_msg = stderr.decode().strip()
                return AuthResult(
                    success=False,
                    method_used=AuthMethod.COOKIES_FILE,
                    error_message=f"Cookies file failed: {error_msg}"
                )
                
        except Exception as e:
            return AuthResult(
                success=False,
                method_used=AuthMethod.COOKIES_FILE,
                error_message=f"Cookies file error: {str(e)}"
            )
    
    async def _try_api_key(self) -> AuthResult:
        """Try to authenticate using API key (note: limited functionality)."""
        api_key = os.getenv('YOUTUBE_API_KEY')
        
        if not api_key:
            return AuthResult(
                success=False,
                method_used=AuthMethod.API_KEY,
                error_message="No API key found in YOUTUBE_API_KEY"
            )
        
        # Note: API key doesn't directly help with yt-dlp authentication
        # but we can indicate it's available for other operations
        return AuthResult(
            success=True,
            method_used=AuthMethod.API_KEY,
            yt_dlp_args=[],  # API key doesn't add yt-dlp args
            metadata={'api_key_available': True},
            error_message="API key available but provides limited yt-dlp access"
        )
    
    async def _try_no_auth(self) -> AuthResult:
        """Try without authentication (will likely fail for some content)."""
        try:
            # Test with a simple yt-dlp command without auth
            cmd = ['yt-dlp', '--quiet', '--no-warnings',
                   '--print', 'title',
                   f'https://www.youtube.com/watch?v={self._test_video_id}']
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and stdout.decode().strip():
                return AuthResult(
                    success=True,
                    method_used=AuthMethod.NO_AUTH,
                    yt_dlp_args=[],
                    error_message="No authentication - limited access expected"
                )
            else:
                error_msg = stderr.decode().strip()
                return AuthResult(
                    success=False,
                    method_used=AuthMethod.NO_AUTH,
                    error_message=f"No auth failed: {error_msg}"
                )
                
        except Exception as e:
            return AuthResult(
                success=False,
                method_used=AuthMethod.NO_AUTH,
                error_message=f"No auth error: {str(e)}"
            )
    
    def _get_authentication_order(self, preferred_method: Optional[AuthMethod]) -> List[AuthMethod]:
        """Get the order of authentication methods to try."""
        default_order = [
            AuthMethod.BROWSER_COOKIES,
            AuthMethod.COOKIES_FILE,
            AuthMethod.API_KEY,
            AuthMethod.NO_AUTH
        ]
        
        if preferred_method and preferred_method in default_order:
            # Move preferred method to front
            order = [preferred_method]
            order.extend([m for m in default_order if m != preferred_method])
            return order
        
        return default_order
    
    def _is_browser_available(self, browser_name: str) -> bool:
        """Check if a browser is available on the system."""
        system = platform.system().lower()
        
        # Basic browser availability detection
        browser_paths = {
            'firefox': {
                'linux': ['/usr/bin/firefox', '/usr/local/bin/firefox'],
                'darwin': ['/Applications/Firefox.app'],
                'windows': ['C:\\Program Files\\Mozilla Firefox\\firefox.exe']
            },
            'chrome': {
                'linux': ['/usr/bin/google-chrome', '/usr/bin/chromium'],
                'darwin': ['/Applications/Google Chrome.app'],
                'windows': ['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe']
            },
            'safari': {
                'darwin': ['/Applications/Safari.app']
            }
        }
        
        if browser_name not in browser_paths:
            return True  # Assume available if not in our list
        
        if system not in browser_paths[browser_name]:
            return False
        
        paths = browser_paths[browser_name][system]
        return any(os.path.exists(path) for path in paths)
    
    async def validate_credentials(self, credentials: AuthCredentials) -> bool:
        """Validate if the provided credentials are still valid."""
        if credentials.method == AuthMethod.BROWSER_COOKIES and credentials.browser_name:
            result = await self._test_browser_cookies(credentials.browser_name)
            return result.success
        elif credentials.method == AuthMethod.COOKIES_FILE and credentials.cookies_file_path:
            # Check if file still exists and test it
            if not os.path.exists(credentials.cookies_file_path):
                return False
            # Create temporary environment and test
            old_env = os.environ.get('YT_DLP_COOKIES_FILE')
            os.environ['YT_DLP_COOKIES_FILE'] = credentials.cookies_file_path
            result = await self._try_cookies_file()
            if old_env:
                os.environ['YT_DLP_COOKIES_FILE'] = old_env
            else:
                os.environ.pop('YT_DLP_COOKIES_FILE', None)
            return result.success
        
        return False
    
    async def refresh_credentials(self, credentials: AuthCredentials) -> AuthResult:
        """Refresh expired credentials if possible."""
        # For browser cookies, we can try to re-extract them
        if credentials.method == AuthMethod.BROWSER_COOKIES:
            return await self._try_browser_cookies()
        
        # For cookies file, user needs to manually refresh
        if credentials.method == AuthMethod.COOKIES_FILE:
            return AuthResult(
                success=False,
                method_used=credentials.method,
                error_message="Cookie file requires manual refresh"
            )
        
        # Try full re-authentication
        return await self.authenticate(credentials.method)
    
    def get_available_methods(self) -> List[AuthMethod]:
        """Get list of authentication methods available on this system."""
        methods = []
        
        # Browser cookies - check if any browsers are available
        if any(self._is_browser_available(browser) for browser in self._browser_priority):
            methods.append(AuthMethod.BROWSER_COOKIES)
        
        # Cookies file - always available if file exists
        if os.getenv('YT_DLP_COOKIES_FILE'):
            methods.append(AuthMethod.COOKIES_FILE)
        
        # API key - check if available
        if os.getenv('YOUTUBE_API_KEY'):
            methods.append(AuthMethod.API_KEY)
        
        # No auth - always available
        methods.append(AuthMethod.NO_AUTH)
        
        return methods
    
    async def test_youtube_access(self, auth_result: AuthResult) -> bool:
        """Test if authentication allows access to YouTube content."""
        try:
            cmd = ['yt-dlp', '--quiet', '--no-warnings'] + auth_result.yt_dlp_args + [
                '--print', 'title',
                f'https://www.youtube.com/watch?v={self._test_video_id}'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            success = process.returncode == 0 and bool(stdout.decode().strip())
            
            if not success:
                logger.warning(f"⚠️ YouTube access test failed: {stderr.decode().strip()}")
            
            return success
            
        except Exception as e:
            logger.error(f"💥 YouTube access test error: {e}")
            return False
    
    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "youtube_auth_adapter" 