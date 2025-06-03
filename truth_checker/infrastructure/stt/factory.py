"""Factory for managing STT providers."""

import os
from typing import Dict, Optional, Type

from ...domain.ports.stt_provider import STTProvider
from .elevenlabs_adapter import ElevenLabsAdapter, ElevenLabsConfig


class STTProviderFactory:
    """Factory for managing STT providers."""

    def __init__(self):
        """Initialize the factory."""
        self._providers: Dict[str, Type[STTProvider]] = {}
        self._instances: Dict[str, STTProvider] = {}
        self._configs: Dict[str, dict] = {}

        # Register built-in providers
        self.register_provider("elevenlabs", ElevenLabsAdapter)

    def register_provider(
        self,
        provider_name: str,
        provider_class: Type[STTProvider],
        config: Optional[dict] = None,
    ) -> None:
        """Register a new STT provider.

        Args:
            provider_name: Name of the provider
            provider_class: Provider class
            config: Optional provider configuration
        """
        if provider_name in self._providers:
            raise ValueError(f"Provider {provider_name} already registered")

        self._providers[provider_name] = provider_class
        if config:
            self._configs[provider_name] = config

    async def create_provider(
        self,
        provider_name: str,
        config: Optional[dict] = None,
    ) -> STTProvider:
        """Create and initialize a provider instance.

        Args:
            provider_name: Name of the provider to create
            config: Optional provider configuration

        Returns:
            Initialized provider instance
        """
        if provider_name not in self._providers:
            raise ValueError(f"Unknown provider: {provider_name}")

        if provider_name in self._instances:
            return self._instances[provider_name]

        provider_class = self._providers[provider_name]
        provider_config = config or self._configs.get(provider_name, {})

        if provider_name == "elevenlabs":
            # Automatically load API key from environment if not provided
            if "api_key" not in provider_config:
                provider_config["api_key"] = os.getenv("ELEVENLABS_API_KEY", "")
            
            instance = provider_class(
                config=ElevenLabsConfig(**provider_config),
                provider_name=provider_name,
            )
        else:
            instance = provider_class(**provider_config)

        await instance.initialize()
        self._instances[provider_name] = instance
        return instance

    async def get_provider(self, provider_name: str) -> STTProvider:
        """Get an existing provider instance.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider instance
        """
        if provider_name not in self._instances:
            raise ValueError(f"Provider {provider_name} not initialized")
        return self._instances[provider_name]

    def list_providers(self) -> list[str]:
        """Get list of registered provider names."""
        return list(self._providers.keys())

    def list_active_providers(self) -> list[str]:
        """Get list of active provider names."""
        return list(self._instances.keys())

    async def shutdown_provider(self, provider_name: str) -> None:
        """Shutdown a specific provider.

        Args:
            provider_name: Name of the provider to shutdown
        """
        if provider_name in self._instances:
            await self._instances[provider_name].shutdown()
            del self._instances[provider_name]

    async def shutdown_all(self) -> None:
        """Shutdown all active providers."""
        for provider_name in list(self._instances.keys()):
            await self.shutdown_provider(provider_name)

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available.

        Args:
            provider_name: Name of the provider to check

        Returns:
            True if provider is available
        """
        return (
            provider_name in self._instances
            and self._instances[provider_name].is_available
        ) 