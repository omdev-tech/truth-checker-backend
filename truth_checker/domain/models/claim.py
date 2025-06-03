"""Domain model for factual claims."""

from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field


class Claim(BaseModel):
    """Represents a factual statement to be verified."""
    
    text: str = Field(..., description="The actual claim text to be verified")
    source: Optional[str] = Field(None, description="Source of the claim if available")
    context: Optional[str] = Field(None, description="Additional context for the claim")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the claim was made")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata about the claim")
    
    class Config:
        """Pydantic model configuration."""
        frozen = True  # Immutable model
        json_schema_extra = {
            "example": {
                "text": "The Earth is approximately 4.54 billion years old.",
                "source": "Scientific presentation",
                "context": "Discussion about planetary formation",
                "metadata": {"category": "science", "confidence": "high"}
            }
        } 