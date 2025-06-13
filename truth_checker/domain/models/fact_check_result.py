"""Domain model for fact checking results."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class FactCheckResult:
    """Result of a fact check operation."""
    
    overall_assessment: str
    claims: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    confidence_score: float
    
    # Legacy fields for backward compatibility
    is_valid: Optional[bool] = None
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    evidence: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate the result and set legacy fields."""
        if not 0 <= self.confidence_score <= 1:
            raise ValueError("Confidence score must be between 0 and 1")
            
        if not self.overall_assessment:
            raise ValueError("Overall assessment cannot be empty")
            
        # Set legacy fields for backward compatibility
        if self.confidence is None:
            self.confidence = self.confidence_score
            
        if self.explanation is None:
            self.explanation = self.overall_assessment
            
        if self.is_valid is None:
            # Determine validity based on confidence and assessment
            self.is_valid = self.confidence_score > 0.6 and 'accurate' in self.overall_assessment.lower()
            
        if self.evidence is None:
            # Convert sources to simple evidence list
            self.evidence = []
            for source in self.sources:
                if isinstance(source, dict):
                    self.evidence.append(source.get('title', source.get('url', 'Unknown source')))
                else:
                    # Handle string sources
                    self.evidence.append(str(source))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert FactCheckResult to dictionary format for API responses."""
        return {
            'overall_assessment': self.overall_assessment,
            'claims': self.claims,
            'sources': self.sources,
            'confidence_score': self.confidence_score,
            'is_valid': self.is_valid,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'evidence': self.evidence
        } 