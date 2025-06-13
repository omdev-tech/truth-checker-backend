"""Port interface for YouTube authentication providers."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel


class AuthMethod(str, Enum):
    """Available YouTube authentication methods."""
    
    BROWSER_COOKIES = "browser_cookies"
    COOKIES_FILE = "cookies_file"
    OAUTH_TOKEN = "oauth_token"
    API_KEY = "api_key"
    NO_AUTH = "no_auth"


class AuthCredentials(BaseModel):
    """YouTube authentication credentials."""
    
    method: AuthMethod
    browser_name: Optional[str] = None  # For browser cookie extraction
    cookies_file_path: Optional[str] = None  # For cookie file
    oauth_token: Optional[str] = None  # For OAuth authentication
    api_key: Optional[str] = None  # For API key authentication
    metadata: Dict = {}  # Additional metadata


class AuthResult(BaseModel):
    """Result of authentication attempt."""
    
    success: bool
    method_used: AuthMethod
    yt_dlp_args: List[str] = []  # Arguments to pass to yt-dlp
    error_message: Optional[str] = None
    expiry_time: Optional[str] = None  # When credentials expire
    metadata: Dict = {}


class YouTubeAuthProvider(ABC):
    """Abstract interface for YouTube authentication providers.
    
    This port defines how different authentication strategies will be handled
    for YouTube access, particularly for yt-dlp operations.
    """

    @abstractmethod
    async def authenticate(self, preferred_method: Optional[AuthMethod] = None) -> AuthResult:
        """Attempt to authenticate with YouTube using available methods.

        Args:
            preferred_method: Preferred authentication method to try first

        Returns:
            Authentication result with yt-dlp arguments
        """
        pass

    @abstractmethod
    async def validate_credentials(self, credentials: AuthCredentials) -> bool:
        """Validate if the provided credentials are still valid.

        Args:
            credentials: Credentials to validate

        Returns:
            True if credentials are valid, False otherwise
        """
        pass

    @abstractmethod
    async def refresh_credentials(self, credentials: AuthCredentials) -> AuthResult:
        """Refresh expired credentials if possible.

        Args:
            credentials: Credentials to refresh

        Returns:
            New authentication result
        """
        pass

    @abstractmethod
    def get_available_methods(self) -> List[AuthMethod]:
        """Get list of authentication methods available on this system.

        Returns:
            List of available authentication methods
        """
        pass

    @abstractmethod
    async def test_youtube_access(self, auth_result: AuthResult) -> bool:
        """Test if authentication allows access to YouTube content.

        Args:
            auth_result: Authentication result to test

        Returns:
            True if YouTube access works, False otherwise
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass 