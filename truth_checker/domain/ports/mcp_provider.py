"""MCP provider interface for fact-checking sources."""

from typing import Dict, List, Optional, Protocol, Any
from pydantic import BaseModel, Field

from ..models.claim import Claim
from ..models.verification import VerificationResult


class MCPSearchResult(BaseModel):
    """Search result from MCP provider."""
    
    title: str = Field(..., description="Article title")
    summary: str = Field(..., description="Article summary")
    url: str = Field(..., description="Article URL")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MCPValidationResult(BaseModel):
    """Result of fact validation."""
    
    is_valid: bool = Field(..., description="Whether the fact is valid")
    confidence: float = Field(..., description="Confidence score (0-1)")
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Supporting evidence texts"
    )
    contradicting_evidence: List[str] = Field(
        default_factory=list,
        description="Contradicting evidence texts"
    )
    source_urls: List[str] = Field(
        default_factory=list,
        description="Source URLs"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class MCPProvider(Protocol):
    """Protocol for MCP-enabled fact-checking sources."""
    
    async def initialize(self) -> None:
        """Initialize the provider and verify connection."""
        ...
    
    async def search(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.5,
    ) -> List[MCPSearchResult]:
        """Search for articles matching the query."""
        ...
    
    async def validate_fact(
        self,
        statement: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> MCPValidationResult:
        """Validate a factual statement."""
        ...
    
    async def get_article_summary(self, title: str) -> str:
        """Get a summary of an article."""
        ...
    
    async def get_article_content(self, title: str) -> Dict[str, Any]:
        """Get the full content of an article."""
        ...
    
    async def set_language(self, language_code: str) -> None:
        """Set the language for subsequent operations."""
        ...
    
    async def shutdown(self) -> None:
        """Shutdown the provider and clean up resources."""
        ...
    
    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        ...
    
    @property
    def is_available(self) -> bool:
        """Check if the provider is available and ready."""
        ...
    
    @property
    def capabilities(self) -> Dict[str, bool]:
        """Get the provider's capabilities."""
        ...
    
    async def verify_fact(
        self,
        claim: Claim,
        context: Optional[Dict[str, str]] = None
    ) -> VerificationResult:
        """Verify a single fact against the provider's sources."""
        ...
    
    async def get_sources(
        self,
        query: str,
        max_sources: int = 5
    ) -> List[Dict[str, str]]:
        """Get authoritative sources for a given query."""
        ...
    
    async def get_context(
        self,
        topic: str,
        max_length: int = 1000
    ) -> str:
        """Get contextual information about a topic."""
        ... 