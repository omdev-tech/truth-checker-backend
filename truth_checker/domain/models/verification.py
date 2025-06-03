"""Domain models for verification results and related entities."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, constr


class VerificationStatus(str, Enum):
    """Possible verification outcomes."""

    TRUE = "true"  # Claim is verified as true
    FALSE = "false"  # Claim is verified as false
    PARTIALLY_TRUE = "partially_true"  # Some aspects are true, others false
    MISLEADING = "misleading"  # Technically true but lacks context
    UNVERIFIABLE = "unverifiable"  # Cannot be verified with available sources
    DISPUTED = "disputed"  # Conflicting authoritative sources


class ConfidenceLevel(str, Enum):
    """Confidence levels in the verification result."""

    HIGH = "high"  # 90-100% confidence
    MEDIUM = "medium"  # 70-89% confidence
    LOW = "low"  # 50-69% confidence
    INSUFFICIENT = "insufficient"  # <50% confidence


class Source(BaseModel):
    """Represents a source used in verification."""

    url: str = Field(..., description="URL or identifier of the source")
    title: str = Field(..., description="Title or name of the source")
    authority_level: str = Field(..., description="Authority level of the source")
    snippet: Optional[str] = Field(None, description="Relevant excerpt from source")
    access_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the source was accessed",
    )


class VerificationResult(BaseModel):
    """Represents the result of a fact-checking process."""
    
    claim_text: str = Field(..., description="The claim that was verified")
    status: VerificationStatus = Field(..., description="Verification status")
    confidence: ConfidenceLevel = Field(..., description="Confidence in the result")
    explanation: str = Field(..., description="Brief explanation of the verification result")
    sources: List[str] = Field(default_factory=list, description="Sources used for verification")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When verification was completed")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional verification metadata")
    
    class Config:
        """Pydantic model configuration."""
        frozen = True  # Immutable model
        json_schema_extra = {
            "example": {
                "claim_text": "The Earth is approximately 4.54 billion years old.",
                "status": "true",
                "confidence": "high",
                "explanation": "This age is confirmed by multiple radiometric dating studies and is widely accepted in the scientific community.",
                "sources": ["Nature Journal", "Geological Society"],
                "metadata": {"verification_method": "scientific_consensus", "fact_type": "scientific"}
            }
        } 