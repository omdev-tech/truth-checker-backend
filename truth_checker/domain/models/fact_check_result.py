"""Domain model for fact checking results."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FactCheckResult:
    """Result of a fact check operation."""
    
    is_valid: bool
    confidence: float
    explanation: str
    evidence: List[str]
    
    def __post_init__(self):
        """Validate the result."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
            
        if not self.explanation:
            raise ValueError("Explanation cannot be empty")
            
        if not self.evidence:
            raise ValueError("Evidence list cannot be empty") 