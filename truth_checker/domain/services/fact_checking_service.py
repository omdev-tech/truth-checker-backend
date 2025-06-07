"""Service for coordinating fact checking between AI and MCP providers."""

import logging
from typing import Protocol, Optional

from ..models.fact_check_result import FactCheckResult

logger = logging.getLogger(__name__)


def confidence_level_to_float(confidence_level) -> float:
    """Convert ConfidenceLevel enum to float value.
    
    Args:
        confidence_level: ConfidenceLevel enum or float
        
    Returns:
        Float confidence value between 0.0 and 1.0
    """
    if isinstance(confidence_level, float):
        return confidence_level
    
    # Handle ConfidenceLevel enum
    if hasattr(confidence_level, 'value'):
        # If it's an enum with string values, map them to floats
        level_str = str(confidence_level.value).lower() if hasattr(confidence_level, 'value') else str(confidence_level).lower()
        if 'high' in level_str:
            return 0.9
        elif 'medium' in level_str:
            return 0.7
        elif 'low' in level_str:
            return 0.5
        else:
            return 0.3  # INSUFFICIENT
    
    # Handle string values directly
    level_str = str(confidence_level).lower()
    if 'high' in level_str:
        return 0.9
    elif 'medium' in level_str:
        return 0.7
    elif 'low' in level_str:
        return 0.5
    else:
        return 0.3


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
    
    def __init__(self, ai_provider: Optional[any] = None, mcp_client: Optional[MCPClient] = None):
        """Initialize the service.
        
        Args:
            ai_provider: AI provider for fact checking (optional for now)
            mcp_client: MCP client for evidence gathering (optional for now)
        """
        self.ai = ai_provider
        self.mcp = mcp_client
        logger.info("🔧 FactCheckingService initialized")
        
    async def fact_check(self, statement: str) -> FactCheckResult:
        """Fact check a statement.
        
        Args:
            statement: Statement to fact check
            
        Returns:
            Fact check result
        """
        logger.info(f"🔍 Starting fact check for statement: {statement[:100]}...")
        
        try:
            # For now, provide a mock implementation until providers are set up
            if not self.ai or not self.mcp:
                logger.warning("⚠️ AI provider or MCP client not configured - using mock fact check")
                return self._mock_fact_check(statement)
            
            # Real implementation using the same pattern as the working fact-check endpoint
            return await self._check_fact_with_providers(statement)
            
        except Exception as e:
            logger.error(f"❌ Fact check failed: {e}")
            # Return error result
            return FactCheckResult(
                overall_assessment="Error during fact checking",
                claims=[],
                sources=[],
                confidence_score=0.0
            )
    
    async def _check_fact_with_providers(self, statement: str) -> FactCheckResult:
        """Check a factual statement using AI and MCP providers (new implementation).
        
        Args:
            statement: Statement to fact check
            
        Returns:
            Fact check result
        """
        # Extract claims from text using AI provider
        claims = await self.ai.analyze_text(statement, None)
        logger.info(f"📝 Extracted {len(claims)} claims from statement")
        
        # Verify each claim
        verification_results = []
        mcp_validations = []
        
        for i, claim in enumerate(claims):
            logger.info(f"🔍 Verifying claim {i+1}/{len(claims)}: {claim.text}")
            
            try:
                # Get AI verification
                ai_result = await self.ai.verify_claim(claim, None)
                logger.info(f"🤖 AI result type: {type(ai_result)}, value: {ai_result}")
                verification_results.append(ai_result)
                
                # Get MCP validation
                mcp_result = await self.mcp.validate_fact(claim.text)
                logger.info(f"📚 MCP result type: {type(mcp_result)}, value: {mcp_result}")
                mcp_validations.append(mcp_result)
                
            except Exception as e:
                logger.warning(f"⚠️ Error verifying claim '{claim.text}': {e}")
                # Create a fallback result for this claim
                verification_results.append(None)
                mcp_validations.append(None)
        
        # Combine results and generate overall assessment
        overall_confidence = 0.0
        valid_results = []
        all_sources = []
        
        for i, (ai_result, mcp_result) in enumerate(zip(verification_results, mcp_validations)):
            logger.info(f"🔄 Processing result {i}: ai_result={type(ai_result)}, mcp_result={type(mcp_result)}")
            
            if ai_result and mcp_result:
                try:
                    # Convert confidence levels to float safely
                    if hasattr(mcp_result, 'confidence'):
                        mcp_confidence = mcp_result.confidence if isinstance(mcp_result.confidence, float) else 0.5
                    else:
                        logger.warning(f"⚠️ mcp_result has no confidence attribute: {mcp_result}")
                        mcp_confidence = 0.5
                    
                    if hasattr(ai_result, 'confidence'):
                        ai_confidence = confidence_level_to_float(ai_result.confidence)
                    else:
                        logger.warning(f"⚠️ ai_result has no confidence attribute: {ai_result}")
                        ai_confidence = 0.5
                    
                    # Combine confidence scores (prefer MCP when available)
                    if hasattr(mcp_result, 'is_valid'):
                        combined_confidence = mcp_confidence if mcp_result.is_valid else ai_confidence
                    else:
                        logger.warning(f"⚠️ mcp_result has no is_valid attribute: {mcp_result}")
                        combined_confidence = ai_confidence
                    
                    overall_confidence += combined_confidence
                    
                    # Collect sources safely
                    try:
                        if hasattr(ai_result, 'sources') and ai_result.sources:
                            logger.info(f"📚 AI sources type: {type(ai_result.sources)}, value: {ai_result.sources}")
                            if isinstance(ai_result.sources, list):
                                all_sources.extend(ai_result.sources)
                            else:
                                all_sources.append(str(ai_result.sources))
                    except Exception as source_error:
                        logger.warning(f"⚠️ Error processing AI sources: {source_error}")
                    
                    try:
                        if hasattr(mcp_result, 'source_urls') and mcp_result.source_urls:
                            logger.info(f"📚 MCP sources type: {type(mcp_result.source_urls)}, value: {mcp_result.source_urls}")
                            source_objects = [{'title': 'Wikipedia Source', 'url': url, 'relevance': 0.8} for url in mcp_result.source_urls]
                            all_sources.extend(source_objects)
                    except Exception as source_error:
                        logger.warning(f"⚠️ Error processing MCP sources: {source_error}")
                    
                    # Get status safely
                    ai_status = getattr(ai_result, 'status', 'unknown')
                    mcp_valid = getattr(mcp_result, 'is_valid', False)
                    
                    valid_results.append({
                        'claim': claims[i].text,
                        'status': 'verified' if mcp_valid else 'disputed',
                        'confidence': combined_confidence,
                        'ai_status': ai_status,
                        'mcp_valid': mcp_valid
                    })
                    
                except Exception as result_error:
                    logger.error(f"❌ Error processing result {i}: {result_error}", exc_info=True)
                    valid_results.append({
                        'claim': claims[i].text,
                        'status': 'error',
                        'confidence': 0.3,
                        'ai_status': 'error',
                        'mcp_valid': False
                    })
        
        # Calculate average confidence
        if valid_results:
            overall_confidence = overall_confidence / len(valid_results)
        
        # Generate overall assessment
        verified_count = sum(1 for r in valid_results if r['status'] == 'verified')
        total_claims = len(valid_results) if valid_results else len(claims)
        
        if total_claims == 0:
            assessment = "No verifiable claims found in the statement"
        elif verified_count == total_claims:
            assessment = f"All {total_claims} claims verified as accurate"
        elif verified_count == 0:
            assessment = f"All {total_claims} claims could not be verified or are disputed"
        else:
            assessment = f"{verified_count} out of {total_claims} claims verified as accurate"
        
        logger.info(f"✅ Fact check complete: {assessment}")
        logger.info(f"📊 Results: {len(valid_results)} claims, {len(all_sources)} sources, confidence: {overall_confidence:.2f}")
        
        return FactCheckResult(
            overall_assessment=assessment,
            claims=valid_results if valid_results else [
                {
                    'claim': statement[:100] + ('...' if len(statement) > 100 else ''),
                    'status': 'unverified',
                    'confidence': 0.5
                }
            ],
            sources=all_sources[:10],  # Limit to 10 sources
            confidence_score=overall_confidence
        )
    
    def _mock_fact_check(self, statement: str) -> FactCheckResult:
        """Mock fact check implementation for development.
        
        Args:
            statement: Statement to fact check
            
        Returns:
            Mock fact check result
        """
        logger.info("🎭 Using mock fact check implementation")
        
        # Simple mock based on statement length and content
        confidence = min(0.8, len(statement) / 100.0)
        
        if any(word in statement.lower() for word in ['true', 'correct', 'accurate', 'fact']):
            assessment = "Likely accurate based on content analysis"
        elif any(word in statement.lower() for word in ['false', 'wrong', 'incorrect', 'fake']):
            assessment = "Potentially inaccurate based on content analysis"
        else:
            assessment = "Requires further verification"
        
        return FactCheckResult(
            overall_assessment=assessment,
            claims=[
                {
                    'claim': statement[:100] + ('...' if len(statement) > 100 else ''),
                    'status': 'unverified',
                    'confidence': confidence
                }
            ],
            sources=[
                {
                    'title': 'Mock Source',
                    'url': 'https://example.com/mock-source',
                    'relevance': 0.5
                }
            ],
            confidence_score=confidence
        )
        
    async def check_fact(self, statement: str) -> FactCheckResult:
        """Check a factual statement using AI and MCP providers (legacy method - deprecated).
        
        This method is kept for backwards compatibility but should not be used.
        Use fact_check() instead.
        
        Args:
            statement: Statement to fact check
            
        Returns:
            Fact check result
        """
        logger.warning("⚠️ Using deprecated check_fact method - use fact_check() instead")
        return await self.fact_check(statement) 