"""Factory for creating and managing AI providers."""

import os
from typing import Dict, Optional, Type

from ...domain.ports.ai_provider import AIProvider
from .chatgpt_adapter import ChatGPTAdapter, ChatGPTConfig


class AIProviderFactory:
    """Factory for creating and managing AI providers."""

    def __init__(self):
        """Initialize the factory."""
        self._providers: Dict[str, Type[AIProvider]] = {}
        self._instances: Dict[str, AIProvider] = {}
        
        # Register default providers
        self.register_provider("chatgpt", ChatGPTAdapter)

    def register_provider(self, name: str, provider_class: Type[AIProvider]) -> None:
        """Register a new AI provider.
        
        Args:
            name: Provider name
            provider_class: Provider class
        """
        self._providers[name] = provider_class

    async def create_provider(
        self,
        name: str,
        **kwargs
    ) -> AIProvider:
        """Create and initialize a provider instance.
        
        Args:
            name: Provider name
            **kwargs: Provider-specific configuration
            
        Returns:
            Initialized provider instance
            
        Raises:
            ValueError: If provider not found
        """
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not found")
        
        if name not in self._instances:
            # Handle ChatGPT configuration
            if name == "chatgpt":
                config = ChatGPTConfig(
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                    **kwargs
                )
                provider = self._providers[name](config=config)
            else:
                provider = self._providers[name](**kwargs)
                
            await provider.initialize()
            self._instances[name] = provider
            
        return self._instances[name]

    def get_provider(self, name: str) -> Optional[AIProvider]:
        """Get an existing provider instance.
        
        Args:
            name: Provider name
            
        Returns:
            Provider instance if exists, None otherwise
        """
        return self._instances.get(name)

    @property
    def available_providers(self) -> Dict[str, bool]:
        """Get dictionary of registered providers and their availability."""
        return {
            name: name in self._instances
            for name in self._providers
        }

    async def shutdown(self) -> None:
        """Shutdown all provider instances."""
        for provider in self._instances.values():
            await provider.shutdown()
        self._instances.clear()


# Global factory instance
_factory: Optional[AIProviderFactory] = None


def get_factory() -> AIProviderFactory:
    """Get the global factory instance."""
    global _factory
    if _factory is None:
        _factory = AIProviderFactory()
    return _factory 