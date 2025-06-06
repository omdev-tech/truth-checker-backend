"""FastAPI application for the Truth Checker service."""

import contextlib
import logging
import os
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

from ..domain.ports.ai_provider import AIAnalysisResult, AIVerificationResult
from ..infrastructure.ai.factory import AIProviderFactory
from ..infrastructure.mcp.factory import MCPProviderFactory
from .endpoints import fact_check, health, stt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str
    version: str
    ai_providers: List[str]
    mcp_providers: List[str]


class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    
    text: str
    provider: str = "ChatGPT"  # Default AI provider


class TextAnalysisResponse(BaseModel):
    """Response model for text analysis."""
    
    analysis: AIAnalysisResult
    metadata: Dict[str, str] = {}


class FactVerificationRequest(BaseModel):
    """Request model for fact verification."""
    
    facts: List[str]
    provider: str = "ChatGPT"  # Default AI provider
    mcp_provider: str = "Wikipedia"  # Default MCP provider


class FactVerificationResponse(BaseModel):
    """Response model for fact verification."""
    
    verification: AIVerificationResult
    metadata: Dict[str, str] = {}


# Create FastAPI application
app = FastAPI(
    title="Truth Checker API",
    description="Real-time fact-checking API with AI and MCP integration",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create provider factories
app.ai_factory = AIProviderFactory()
app.mcp_factory = MCPProviderFactory()

# Include routers
app.include_router(health.router)
app.include_router(fact_check.router)
app.include_router(stt.router)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application.
    
    This replaces the deprecated on_event handlers with the new
    lifespan event handler pattern.
    """
    # Startup: Initialize providers
    if api_key := os.getenv("OPENAI_API_KEY"):
        try:
            await app.ai_factory.create_provider("chatgpt")
        except Exception as e:
            print(f"Failed to initialize ChatGPT provider: {e}")

    try:
        await app.mcp_factory.create_provider("wikipedia")
    except Exception as e:
        print(f"Failed to initialize Wikipedia provider: {e}")

    yield  # Application runs here

    # Shutdown: Cleanup providers
    await app.ai_factory.shutdown()
    await app.mcp_factory.shutdown_all()


# Set lifespan handler
app.router.lifespan_context = lifespan


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health and available providers."""
    try:
        ai_providers = list(app.ai_factory.list_providers().keys())
    except Exception:
        ai_providers = []
    
    try:
        mcp_providers = list(app.mcp_factory.list_providers().keys())
    except Exception:
        mcp_providers = []
    
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        ai_providers=ai_providers,
        mcp_providers=mcp_providers,
    )


@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest) -> TextAnalysisResponse:
    """Analyze text to identify factual claims.

    Args:
        request: Text analysis request

    Returns:
        Analysis results with identified facts

    Raises:
        HTTPException: If provider not found or analysis fails
    """
    # Get AI provider
    provider = app.ai_factory.get_provider(request.provider)
    if not provider:
        raise HTTPException(
            status_code=404,
            detail=f"AI provider not found: {request.provider}",
        )

    try:
        # Analyze text
        analysis = await provider.analyze_statement(request.text)
        return TextAnalysisResponse(
            analysis=analysis,
            metadata={"provider": provider.provider_name},
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


@app.post("/verify", response_model=FactVerificationResponse)
async def verify_facts(request: FactVerificationRequest) -> FactVerificationResponse:
    """Verify facts using AI and MCP providers.

    Args:
        request: Fact verification request

    Returns:
        Verification results with explanation

    Raises:
        HTTPException: If providers not found or verification fails
    """
    # Get providers
    ai_provider = app.ai_factory.get_provider(request.provider)
    if not ai_provider:
        raise HTTPException(
            status_code=404,
            detail=f"AI provider not found: {request.provider}",
        )

    mcp_provider = app.mcp_factory.get_provider(request.mcp_provider)
    if not mcp_provider:
        raise HTTPException(
            status_code=404,
            detail=f"MCP provider not found: {request.mcp_provider}",
        )

    try:
        # Get evidence from MCP provider
        evidence = []
        for fact in request.facts:
            results = await mcp_provider.search(fact)
            evidence.extend(result.content for result in results)

        # Verify facts
        verification = await ai_provider.verify_facts(
            facts=request.facts,
            evidence=evidence,
        )

        return FactVerificationResponse(
            verification=verification,
            metadata={
                "ai_provider": ai_provider.provider_name,
                "mcp_provider": mcp_provider.provider_name,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}",
        )


@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload audio/video file for processing.

    This endpoint will be implemented in Phase 2 with ElevenLabs STT integration.
    """
    raise HTTPException(
        status_code=501,
        detail="File upload support coming in Phase 2",
    ) 