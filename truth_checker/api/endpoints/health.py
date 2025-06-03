"""Health check endpoints."""

from typing import Dict

from fastapi import APIRouter

from ...infrastructure.ai.factory import AIProviderFactory
from ...infrastructure.mcp.factory import MCPProviderFactory

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> Dict[str, Dict[str, bool]]:
    """Check the health of all service components.
    
    Returns:
        Health status of AI and MCP providers
    """
    # Check AI providers using the available_providers property
    ai_factory = AIProviderFactory()
    ai_status = {}
    
    for provider_name, is_active in ai_factory.available_providers.items():
        try:
            ai_status[provider_name.title()] = is_active
        except Exception:
            ai_status[provider_name.title()] = False
    
    # Check MCP providers using the available_providers property
    mcp_factory = MCPProviderFactory()
    mcp_status = {}
    
    for provider_name, is_active in mcp_factory.available_providers.items():
        try:
            mcp_status[provider_name.title()] = is_active
        except Exception:
            mcp_status[provider_name.title()] = False
    
    return {
        "ai_providers": ai_status,
        "mcp_providers": mcp_status,
    } 