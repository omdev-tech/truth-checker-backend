"""Protocol for AI providers."""

from typing import Protocol, List
from pydantic import BaseModel


class AIAnalysisResult(BaseModel):
    """Result of AI analysis of text."""
    facts: List[str]
    confidence: float


class AIVerificationResult(BaseModel):
    """Result of AI verification of facts."""
    is_valid: bool
    confidence: float
    explanation: str
    evidence: List[str]


class AIProvider(Protocol):
    """Protocol defining the interface for AI providers."""
    
    async def initialize(self) -> None:
        """Initialize the AI provider."""
        ...
        
    async def shutdown(self) -> None:
        """Clean up resources."""
        ...
        
    async def analyze_statement(self, statement: str) -> AIAnalysisResult:
        """Analyze a statement and extract key claims."""
        ...
        
    async def validate_with_evidence(self, statement: str, evidence: List[str]) -> AIVerificationResult:
        """Validate a statement against evidence."""
        ...
        
    async def generate_explanation(self, statement: str, is_valid: bool, evidence: List[str]) -> str:
        """Generate a human-readable explanation."""
        ... 