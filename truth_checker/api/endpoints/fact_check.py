"""Fact-checking API endpoints."""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket
from pydantic import BaseModel, Field

from ...domain.models.claim import Claim
from ...domain.models.verification import VerificationResult, ConfidenceLevel
from ...infrastructure.ai.factory import AIProviderFactory
from ...infrastructure.mcp.factory import MCPProviderFactory, mcp_factory

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fact-check", tags=["fact-check"])


def float_to_confidence_level(confidence: float) -> ConfidenceLevel:
    """Convert float confidence (0-1) to ConfidenceLevel enum."""
    if confidence >= 0.9:
        return ConfidenceLevel.HIGH
    elif confidence >= 0.7:
        return ConfidenceLevel.MEDIUM
    elif confidence >= 0.5:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.INSUFFICIENT


class TextCheckRequest(BaseModel):
    """Request model for text fact-checking."""
    
    text: str = Field(..., description="Text to fact-check")
    context: Optional[Dict[str, str]] = Field(None, description="Additional context")
    language: str = Field(default="en", description="Language code")


class TextCheckResponse(BaseModel):
    """Response model for text fact-checking."""
    
    claims: List[Claim] = Field(..., description="Extracted claims")
    results: List[VerificationResult] = Field(..., description="Verification results")


@router.post("/text", response_model=TextCheckResponse)
async def check_text(request: TextCheckRequest) -> TextCheckResponse:
    """Check facts in text input.
    
    Args:
        request: Text check request
        
    Returns:
        Verification results for extracted claims
    """
    logger.info(f"Starting fact-check for text: {request.text[:100]}...")
    
    try:
        # Get or create AI provider for claim extraction
        logger.info("Getting AI provider...")
        ai_factory = AIProviderFactory()
        ai_provider = ai_factory.get_provider("chatgpt")
        if ai_provider is None:
            logger.info("AI provider not found, creating new one...")
            ai_provider = await ai_factory.create_provider("chatgpt")
        logger.info("AI provider ready")
        
        # Get or create MCP provider for fact verification
        logger.info("Getting MCP provider...")
        mcp_provider = mcp_factory.get_provider("wikipedia")
        if mcp_provider is None:
            logger.info("MCP provider not found, creating new one...")
            mcp_provider = await mcp_factory.create_provider("wikipedia")
        logger.info("MCP provider ready")
        
        # Set language if specified
        if request.language != "en":
            logger.info(f"Setting language to {request.language}")
            await mcp_provider.set_language(request.language)
        
        # Extract claims from text
        logger.info("Extracting claims from text...")
        claims = await ai_provider.analyze_text(request.text, request.context)
        logger.info(f"Found {len(claims)} claims")
        
        # Verify each claim
        results = []
        for i, claim in enumerate(claims):
            logger.info(f"Verifying claim {i+1}/{len(claims)}: {claim.text}")
            
            # Get AI verification
            logger.info("Getting AI verification...")
            ai_result = await ai_provider.verify_claim(claim, request.context)
            logger.info(f"AI verification complete: {ai_result.status}")
            
            # Get MCP validation
            logger.info("Getting MCP validation...")
            mcp_result = await mcp_provider.validate_fact(claim.text)
            logger.info(f"MCP validation complete: {mcp_result.is_valid}")
            
            # Combine results (prefer MCP when available)
            if mcp_result.is_valid:
                # Create new result with combined data
                combined_sources = ai_result.sources + mcp_result.source_urls
                final_result = VerificationResult(
                    claim_text=ai_result.claim_text,
                    status=ai_result.status,
                    confidence=float_to_confidence_level(mcp_result.confidence),
                    explanation=ai_result.explanation,
                    sources=combined_sources,
                    timestamp=ai_result.timestamp,
                    metadata=ai_result.metadata
                )
                results.append(final_result)
            else:
                results.append(ai_result)
        
        logger.info("Fact-checking complete")
        return TextCheckResponse(claims=claims, results=results)
    
    except Exception as e:
        logger.error(f"Error in fact-checking: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@router.post("/file")
async def check_file(
    file: UploadFile = File(...),
    language: str = "en",
) -> TextCheckResponse:
    """Check facts in uploaded file content.
    
    Args:
        file: Uploaded file
        language: Language code
        
    Returns:
        Verification results for extracted claims
    """
    try:
        # Read file content
        content = await file.read()
        text = content.decode()
        
        # Process using text endpoint
        request = TextCheckRequest(text=text, language=language)
        return await check_text(request)
    
    except Exception as e:
        logger.error(f"Error in file fact-checking: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time fact-checking via WebSocket.
    
    Accepts streaming text input and returns verification results.
    """
    await websocket.accept()
    
    try:
        # Initialize providers
        ai_factory = AIProviderFactory()
        ai_provider = ai_factory.get_provider("chatgpt")
        if ai_provider is None:
            ai_provider = await ai_factory.create_provider("chatgpt")
        
        mcp_provider = mcp_factory.get_provider("wikipedia")
        if mcp_provider is None:
            mcp_provider = await mcp_factory.create_provider("wikipedia")
        
        while True:
            # Receive text chunk
            data = await websocket.receive_json()
            text = data.get("text", "")
            language = data.get("language", "en")
            context = data.get("context")
            
            if not text:
                continue
                
            # Set language if needed
            if language != "en":
                await mcp_provider.set_language(language)
            
            # Process text chunk
            claims = await ai_provider.analyze_text(text, context)
            results = []
            
            for claim in claims:
                # Get AI verification
                ai_result = await ai_provider.verify_claim(claim, context)
                
                # Get MCP validation
                mcp_result = await mcp_provider.validate_fact(claim.text)
                
                # Combine results
                if mcp_result.is_valid:
                    # Create new result with combined data
                    combined_sources = ai_result.sources + mcp_result.source_urls
                    final_result = VerificationResult(
                        claim_text=ai_result.claim_text,
                        status=ai_result.status,
                        confidence=float_to_confidence_level(mcp_result.confidence),
                        explanation=ai_result.explanation,
                        sources=combined_sources,
                        timestamp=ai_result.timestamp,
                        metadata=ai_result.metadata
                    )
                    results.append(final_result)
                else:
                    results.append(ai_result)
            
            # Send results back
            await websocket.send_json({
                "claims": [claim.model_dump() for claim in claims],
                "results": [result.model_dump() for result in results],
            })
    
    except Exception as e:
        logger.error(f"WebSocket error: {type(e).__name__}: {str(e)}", exc_info=True)
        await websocket.close(code=1001, reason=str(e)) 