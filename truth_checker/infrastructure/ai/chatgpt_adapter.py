"""ChatGPT implementation of the AI provider interface."""

import json
from typing import Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from ...domain.models.claim import Claim
from ...domain.models.verification import ConfidenceLevel, VerificationResult, VerificationStatus
from ...domain.ports.ai_provider import AIProvider


class ChatGPTConfig(BaseModel):
    """Configuration for ChatGPT adapter."""
    
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4", description="Model to use (gpt-4 recommended)")
    temperature: float = Field(default=0.1, description="Temperature for responses")
    max_tokens: int = Field(default=1000, description="Maximum tokens per response")
    timeout: float = Field(default=30.0, description="API timeout in seconds")


class ChatGPTAdapter(AIProvider):
    """ChatGPT implementation of the AI provider interface."""

    def __init__(
        self,
        config: Optional[ChatGPTConfig] = None,
    ):
        """Initialize the adapter."""
        self._config = config or ChatGPTConfig(api_key="")
        self._client: Optional[httpx.AsyncClient] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the HTTP client and verify API access."""
        try:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    base_url="https://api.openai.com/v1",
                    timeout=self._config.timeout,
                    headers={
                        "Authorization": f"Bearer {self._config.api_key}",
                        "Content-Type": "application/json",
                    },
                )

            # Test connection
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self._config.model,
                    "messages": [{"role": "system", "content": "Test connection"}],
                    "max_tokens": 5,
                },
            )
            response.raise_for_status()
            self._initialized = True
        except Exception as e:
            self._initialized = False
            if self._client:
                await self._client.aclose()
                self._client = None
            raise ConnectionError(f"Failed to initialize ChatGPT provider: {e}")

    async def verify_claim(
        self,
        claim: Claim,
        context: Optional[Dict[str, str]] = None
    ) -> VerificationResult:
        """Verify a single claim using AI reasoning."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        # Create system prompt for verification
        system_prompt = """
        Verify the following claim based on your knowledge.
        Respond in JSON format with:
        {
            "status": "true/false/partially_true/misleading/unverifiable/disputed",
            "confidence": "high/medium/low/insufficient",
            "explanation": "Brief explanation (max 500 chars)",
            "sources": ["List of relevant sources"]
        }
        Focus on objective verification and cite reliable sources.
        """

        # Add context to user prompt if provided
        user_prompt = f"Claim: {claim.text}"
        if context:
            user_prompt += f"\nContext: {json.dumps(context)}"
        if claim.context:
            user_prompt += f"\nAdditional Context: {claim.context}"

        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self._config.temperature,
                "max_tokens": self._config.max_tokens,
            },
        )
        response.raise_for_status()
        result = json.loads(response.json()["choices"][0]["message"]["content"])

        return VerificationResult(
            claim_text=claim.text,
            status=VerificationStatus(result["status"]),
            confidence=ConfidenceLevel(result["confidence"]),
            explanation=result["explanation"],
            sources=result["sources"],
            metadata={
                "ai_model": self._config.model,
                "verification_method": "ai_reasoning",
            }
        )

    async def verify_claims(
        self,
        claims: List[Claim],
        context: Optional[Dict[str, str]] = None
    ) -> List[VerificationResult]:
        """Verify multiple claims in batch."""
        return [await self.verify_claim(claim, context) for claim in claims]

    async def analyze_text(
        self,
        text: str,
        context: Optional[Dict[str, str]] = None
    ) -> List[Claim]:
        """Extract verifiable claims from text."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        system_prompt = """
        Analyze the text and extract ALL factual claims, regardless of whether they appear true or false.
        Extract any statement that makes a claim about reality, facts, or events that can be verified or disproven.
        Include controversial, disputed, or obviously false claims - the verification step will determine their accuracy.
        
        Respond in JSON format with a list of claims:
        {
            "claims": [
                {
                    "text": "The actual claim",
                    "context": "Surrounding context",
                    "metadata": {"category": "type of claim", "confidence": "extraction confidence"}
                }
            ]
        }
        
        Examples of claims to extract:
        - Scientific facts (true or false): "The Earth is flat", "Vaccines cause autism"
        - Historical statements: "Napoleon was short", "The moon landing was fake"
        - Current events: "Company X reported profits", "Country Y invaded Country Z"
        - Statistical claims: "Crime has increased by 50%", "Population is 100 million"
        
        Do NOT filter based on apparent truth value. Extract the claim and let verification determine accuracy.
        """

        user_prompt = f"Text: {text}"
        if context:
            user_prompt += f"\nContext: {json.dumps(context)}"

        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self._config.temperature,
                "max_tokens": self._config.max_tokens,
            },
        )
        response.raise_for_status()
        result = json.loads(response.json()["choices"][0]["message"]["content"])

        return [
            Claim(
                text=claim["text"],
                context=claim.get("context"),
                metadata=claim.get("metadata", {})
            )
            for claim in result["claims"]
        ]

    async def generate_explanation(
        self,
        result: VerificationResult,
        max_length: int = 500
    ) -> str:
        """Generate a concise explanation for a verification result."""
        if not self._client:
            raise RuntimeError("Provider not initialized")

        system_prompt = f"""
        Generate a clear, concise explanation of the fact verification result.
        Keep the explanation under {max_length} characters.
        Focus on the key findings, confidence level, and supporting evidence.
        """

        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(result.model_dump()),
                    },
                ],
                "temperature": self._config.temperature,
                "max_tokens": self._config.max_tokens,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    async def shutdown(self) -> None:
        """Clean up resources and shut down the provider."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False

    @property
    def provider_name(self) -> str:
        """Get the name of the AI provider."""
        return "ChatGPT"

    @property
    def is_available(self) -> bool:
        """Check if the provider is available and ready."""
        return self._initialized and self._client is not None

    @property
    def capabilities(self) -> Dict[str, bool]:
        """Get the provider's capabilities."""
        return {
            "claim_extraction": True,
            "claim_verification": True,
            "batch_processing": True,
            "explanation_generation": True,
            "multilingual": True,
        } 