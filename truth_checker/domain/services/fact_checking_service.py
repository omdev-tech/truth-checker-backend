"""Service for coordinating fact checking between AI and MCP providers."""

from typing import Protocol

from ..models.fact_check_result import FactCheckResult


class MCPClient(Protocol):
    """Protocol for MCP clients."""
    
    async def search(self, query: str) -> list[str]:
        """Search for articles."""
        ...
        
    async def summary(self, article: str) -> str:
        """Get article summary."""
        ...


class FactCheckingService:
    """Service for coordinating fact checking."""
    
    def __init__(self, ai_provider: any, mcp_client: MCPClient):
        """Initialize the service."""
        self.ai = ai_provider
        self.mcp = mcp_client
        
    async def check_fact(self, statement: str) -> FactCheckResult:
        """Check a factual statement."""
        # Get key claims from AI
        claims = await self.ai.analyze_statement(statement)
        
        # Search for evidence
        articles = await self.mcp.search(claims)
        evidence = []
        for article in articles:
            summary = await self.mcp.summary(article)
            evidence.append(summary)
            
        # Validate with AI
        is_valid, confidence = await self.ai.validate_with_evidence(statement, evidence)
        
        # Generate explanation
        explanation = await self.ai.generate_explanation(statement, is_valid, evidence)
        
        return FactCheckResult(
            is_valid=is_valid,
            confidence=confidence,
            explanation=explanation,
            evidence=evidence
        ) 