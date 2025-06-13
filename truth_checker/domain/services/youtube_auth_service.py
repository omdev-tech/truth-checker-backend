"""Domain service for YouTube authentication management."""

import logging
from typing import List, Optional

from ..ports.youtube_auth_provider import (
    YouTubeAuthProvider,
    AuthMethod,
    AuthResult,
    AuthCredentials
)

logger = logging.getLogger(__name__)


class YouTubeAuthService:
    """Domain service for YouTube authentication.
    
    This service coordinates YouTube authentication using different providers
    and strategies, following the hexagonal architecture pattern.
    """
    
    def __init__(self, auth_provider: YouTubeAuthProvider):
        """Initialize service with authentication provider.
        
        Args:
            auth_provider: YouTube authentication provider implementation
        """
        self._auth_provider = auth_provider
        self._current_auth: Optional[AuthResult] = None
    
    async def authenticate(self, preferred_method: Optional[AuthMethod] = None) -> AuthResult:
        """Authenticate with YouTube using the best available method.
        
        Args:
            preferred_method: Preferred authentication method to try first
            
        Returns:
            Authentication result with yt-dlp arguments
        """
        logger.info(f"🔐 Starting YouTube authentication (preferred: {preferred_method})")
        
        try:
            # Attempt authentication
            auth_result = await self._auth_provider.authenticate(preferred_method)
            
            if auth_result.success:
                logger.info(f"✅ YouTube authentication successful: {auth_result.method_used}")
                logger.debug(f"🔧 yt-dlp args: {auth_result.yt_dlp_args}")
                self._current_auth = auth_result
            else:
                logger.warning(f"❌ YouTube authentication failed: {auth_result.error_message}")
            
            return auth_result
            
        except Exception as e:
            logger.error(f"💥 YouTube authentication error: {e}")
            return AuthResult(
                success=False,
                method_used=AuthMethod.NO_AUTH,
                error_message=str(e)
            )
    
    async def ensure_valid_authentication(self) -> AuthResult:
        """Ensure we have valid authentication, refreshing if needed.
        
        Returns:
            Valid authentication result
        """
        # If no current auth, try to authenticate
        if not self._current_auth:
            logger.info("🔄 No current authentication, attempting to authenticate...")
            return await self.authenticate()
        
        # Test if current auth is still valid
        if await self._auth_provider.test_youtube_access(self._current_auth):
            logger.debug("✅ Current authentication is still valid")
            return self._current_auth
        
        logger.info("🔄 Current authentication invalid, attempting refresh...")
        
        # Try to refresh credentials
        credentials = AuthCredentials(
            method=self._current_auth.method_used,
            metadata=self._current_auth.metadata
        )
        
        refresh_result = await self._auth_provider.refresh_credentials(credentials)
        
        if refresh_result.success:
            logger.info(f"✅ Authentication refreshed: {refresh_result.method_used}")
            self._current_auth = refresh_result
            return refresh_result
        
        # Refresh failed, try new authentication
        logger.warning("⚠️ Authentication refresh failed, trying new authentication...")
        return await self.authenticate()
    
    def get_yt_dlp_args(self) -> List[str]:
        """Get yt-dlp arguments for current authentication.
        
        Returns:
            List of yt-dlp arguments, empty if not authenticated
        """
        if self._current_auth and self._current_auth.success:
            return self._current_auth.yt_dlp_args
        return []
    
    def get_available_auth_methods(self) -> List[AuthMethod]:
        """Get list of available authentication methods.
        
        Returns:
            List of available authentication methods
        """
        return self._auth_provider.get_available_methods()
    
    async def test_authentication(self) -> bool:
        """Test if current authentication works with YouTube.
        
        Returns:
            True if authentication works, False otherwise
        """
        if not self._current_auth:
            return False
        
        return await self._auth_provider.test_youtube_access(self._current_auth)
    
    @property
    def is_authenticated(self) -> bool:
        """Check if currently authenticated.
        
        Returns:
            True if authenticated, False otherwise
        """
        return self._current_auth is not None and self._current_auth.success
    
    @property
    def current_method(self) -> Optional[AuthMethod]:
        """Get current authentication method.
        
        Returns:
            Current authentication method or None
        """
        if self._current_auth:
            return self._current_auth.method_used
        return None
    
    @property
    def provider_name(self) -> str:
        """Get the authentication provider name."""
        return self._auth_provider.provider_name 