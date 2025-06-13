"""Dependency injection configuration for hexagonal architecture."""

import logging
import os
from functools import lru_cache
from typing import Dict, Any

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Try to load .env from current directory or parent directories
    load_dotenv(verbose=True)
    logger = logging.getLogger(__name__)
    logger.info("📁 Environment variables loaded from .env file via python-dotenv")
except ImportError:
    # python-dotenv not installed, continue with system environment
    pass
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Could not load .env file: {e}")

from ..domain.services.stream_analysis_service import StreamAnalysisService
from ..domain.services.stt_service import STTService
from ..domain.services.fact_checking_service import FactCheckingService
from ..domain.services.youtube_auth_service import YouTubeAuthService
from ..infrastructure.ai.youtube_live_detector_adapter import YouTubeLiveDetectorAdapter
from ..infrastructure.ai.factory import AIProviderFactory
from ..infrastructure.mcp.factory import MCPProviderFactory
from ..infrastructure.stt.elevenlabs_adapter import ElevenLabsAdapter, ElevenLabsConfig
from ..infrastructure.stt.youtube_auth_adapter import YouTubeAuthAdapter
from ..infrastructure.stt.ytdlp_config import ytdlp_config

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Service container for dependency injection."""
    
    def __init__(self):
        """Initialize service container."""
        self._services: Dict[str, Any] = {}
        self._setup_services()
    
    async def _setup_ai_and_mcp_providers(self):
        """Setup AI and MCP providers for fact checking."""
        try:
            # Create AI provider
            logger.info("🤖 Setting up AI provider...")
            ai_factory = AIProviderFactory()
            ai_provider = ai_factory.get_provider("chatgpt")
            if ai_provider is None:
                logger.info("🔨 Creating new AI provider...")
                ai_provider = await ai_factory.create_provider("chatgpt")
            logger.info("✅ AI provider ready")
            
            # Create MCP provider
            logger.info("📚 Setting up MCP provider...")
            mcp_factory = MCPProviderFactory()
            mcp_provider = mcp_factory.get_provider("wikipedia")
            if mcp_provider is None:
                logger.info("🔨 Creating new MCP provider...")
                mcp_provider = await mcp_factory.create_provider("wikipedia")
            logger.info("✅ MCP provider ready")
            
            return ai_provider, mcp_provider
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to setup AI/MCP providers: {e}")
            logger.info("🎭 FactCheckingService will use mock implementation")
            return None, None
    
    def _setup_services(self):
        """Setup all services and their dependencies."""
        logger.info("🔧 Setting up service container...")
        
        # Initialize yt-dlp configuration and log status
        logger.info("🔧 Initializing yt-dlp configuration...")
        if ytdlp_config.enable_cookies:
            if ytdlp_config.cookies_from_browser:
                logger.info(f"✅ yt-dlp cookies configured: browser={ytdlp_config.cookies_from_browser}")
            elif ytdlp_config.cookies_file:
                logger.info(f"✅ yt-dlp cookies configured: file={ytdlp_config.cookies_file}")
            else:
                logger.warning("⚠️ yt-dlp cookies enabled but no source configured - may face YouTube bot detection")
        else:
            logger.warning("⚠️ yt-dlp cookies disabled - may face YouTube bot detection")
        
        # Infrastructure adapters
        youtube_detector = YouTubeLiveDetectorAdapter()
        youtube_auth_adapter = YouTubeAuthAdapter()
        
        # Configure ElevenLabs adapter with API key from environment
        elevenlabs_api_key = os.getenv('ELEVENLABS_API_KEY', '')
        if not elevenlabs_api_key:
            logger.warning("⚠️ ELEVENLABS_API_KEY not found in environment variables")
        else:
            logger.info(f"✅ ElevenLabs API key loaded: {len(elevenlabs_api_key)} chars")
        
        elevenlabs_config = ElevenLabsConfig(api_key=elevenlabs_api_key)
        stt_adapter = ElevenLabsAdapter(config=elevenlabs_config)
        
        # Domain services
        stream_analysis_service = StreamAnalysisService(youtube_detector)
        stt_service = STTService(stt_adapter)
        youtube_auth_service = YouTubeAuthService(youtube_auth_adapter)
        
        # Note: FactCheckingService will be created lazily with providers
        fact_checking_service = None
        
        # Register services
        self._services = {
            'stream_analysis_service': stream_analysis_service,
            'stt_service': stt_service,
            'fact_checking_service': fact_checking_service,  # Will be created on-demand
            'youtube_auth_service': youtube_auth_service,
            'youtube_detector': youtube_detector,
        }
        
        logger.info("✅ Service container setup completed")
    
    async def _ensure_fact_checking_service(self):
        """Ensure fact checking service is created with providers."""
        if self._services['fact_checking_service'] is None:
            logger.info("🔧 Creating FactCheckingService with providers...")
            ai_provider, mcp_provider = await self._setup_ai_and_mcp_providers()
            self._services['fact_checking_service'] = FactCheckingService(ai_provider, mcp_provider)
            logger.info("✅ FactCheckingService created with providers")
        
        return self._services['fact_checking_service']
    
    def get(self, service_name: str) -> Any:
        """Get a service by name.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service not found
        """
        if service_name not in self._services:
            raise KeyError(f"Service '{service_name}' not found")
        return self._services[service_name]
    
    def get_stream_analysis_service(self) -> StreamAnalysisService:
        """Get stream analysis service."""
        return self.get('stream_analysis_service')
    
    def get_stt_service(self) -> STTService:
        """Get STT service."""
        return self.get('stt_service')
    
    def get_youtube_auth_service(self) -> YouTubeAuthService:
        """Get YouTube authentication service."""
        return self.get('youtube_auth_service')
    
    async def get_fact_checking_service(self) -> FactCheckingService:
        """Get fact checking service with providers."""
        return await self._ensure_fact_checking_service()


# Global service container instance
@lru_cache()
def get_service_container() -> ServiceContainer:
    """Get global service container instance.
    
    Returns:
        Service container instance
    """
    return ServiceContainer()


# Convenience functions for FastAPI dependency injection
def get_stream_analysis_service() -> StreamAnalysisService:
    """FastAPI dependency for stream analysis service."""
    return get_service_container().get_stream_analysis_service()


def get_stt_service() -> STTService:
    """FastAPI dependency for STT service."""
    return get_service_container().get_stt_service()


def get_youtube_auth_service() -> YouTubeAuthService:
    """FastAPI dependency for YouTube authentication service."""
    return get_service_container().get_youtube_auth_service()


async def get_fact_checking_service() -> FactCheckingService:
    """FastAPI dependency for fact checking service."""
    container = get_service_container()
    return await container.get_fact_checking_service() 