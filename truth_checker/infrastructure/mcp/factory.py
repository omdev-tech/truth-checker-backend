"""Factory for creating and managing MCP providers."""

from typing import Dict, Optional, Type

from ...domain.ports.mcp_provider import MCPProvider
from .wikipedia_adapter import WikipediaMCPAdapter


class MCPProviderFactory:
    """Factory for creating and managing MCP providers.
    
    This factory maintains a registry of available MCP providers
    and handles their lifecycle (initialization, shutdown).
    """

    def __init__(self):
        """Initialize the factory."""
        self._provider_registry: Dict[str, Type[MCPProvider]] = {}
        self._active_providers: Dict[str, MCPProvider] = {}
        
        # Register default providers
        self.register_provider("wikipedia", WikipediaMCPAdapter)

    def register_provider(
        self, name: str, provider_class: Type[MCPProvider]
    ) -> None:
        """Register a new MCP provider class.
        
        Args:
            name: Unique identifier for the provider
            provider_class: The provider class to register
        """
        if name in self._provider_registry:
            raise ValueError(f"Provider {name} already registered")
        self._provider_registry[name] = provider_class

    async def create_provider(
        self,
        name: str,
        **config: Dict
    ) -> MCPProvider:
        """Create and initialize a new MCP provider instance.
        
        Args:
            name: Name of the provider to create
            **config: Provider-specific configuration
            
        Returns:
            Initialized provider instance
            
        Raises:
            ValueError: If provider not found
            RuntimeError: If initialization fails
        """
        if name not in self._provider_registry:
            raise ValueError(f"Provider {name} not registered")

        provider_class = self._provider_registry[name]
        provider = provider_class(**config)
        
        try:
            await provider.initialize()
            self._active_providers[name] = provider
            return provider
        except Exception as e:
            raise RuntimeError(f"Failed to initialize provider {name}: {e}")

    def get_provider(self, name: str) -> Optional[MCPProvider]:
        """Get an active provider instance by name.
        
        Args:
            name: Name of the provider
            
        Returns:
            Provider instance if active, None otherwise
        """
        return self._active_providers.get(name)

    async def shutdown_provider(self, name: str) -> None:
        """Shutdown a specific provider.
        
        Args:
            name: Name of the provider to shutdown
        """
        provider = self._active_providers.get(name)
        if provider:
            await provider.shutdown()
            del self._active_providers[name]

    async def shutdown_all(self) -> None:
        """Shutdown all active providers."""
        for name in list(self._active_providers.keys()):
            await self.shutdown_provider(name)

    @property
    def available_providers(self) -> Dict[str, bool]:
        """Get dictionary of registered providers and their availability."""
        return {
            name: bool(self.get_provider(name))
            for name in self._provider_registry
        }

    def list_providers(self) -> Dict[str, bool]:
        """Get dictionary of registered providers and their availability.
        
        Returns:
            Dictionary mapping provider names to their availability status
        """
        return {
            name: bool(self.get_provider(name))
            for name in self._provider_registry
        }


# Create global factory instance
mcp_factory = MCPProviderFactory()

# Note: Default providers are now registered in __init__ 