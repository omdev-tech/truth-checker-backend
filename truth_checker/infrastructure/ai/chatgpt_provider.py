"""ChatGPT implementation of the AI provider interface."""

import json
import os
from typing import List, Tuple

from openai import AsyncOpenAI


class ChatGPTProvider:
    """ChatGPT implementation of the AI provider interface."""
    
    def __init__(self):
        """Initialize the provider."""
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    async def initialize(self) -> None:
        """Initialize the provider."""
        # Nothing to initialize for ChatGPT
        pass
        
    async def shutdown(self) -> None:
        """Clean up resources."""
        # Nothing to clean up for ChatGPT
        pass
        
    async def analyze_statement(self, statement: str) -> str:
        """Analyze a statement and extract key claims."""
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a fact-checking assistant. Extract the key factual claims from the given statement. Return them as a comma-separated list."},
                {"role": "user", "content": statement}
            ]
        )
        return response.choices[0].message.content
        
    async def validate_with_evidence(self, statement: str, evidence: List[str]) -> Tuple[bool, float]:
        """Validate a statement against evidence."""
        evidence_text = "\n".join(f"- {e}" for e in evidence)
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a fact-checking assistant. Validate the statement against the provided evidence. Return a JSON object with 'is_valid' (boolean) and 'confidence' (float between 0 and 1)."},
                {"role": "user", "content": f"Statement: {statement}\n\nEvidence:\n{evidence_text}"}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result["is_valid"], result["confidence"]
        
    async def generate_explanation(self, statement: str, is_valid: bool, evidence: List[str]) -> str:
        """Generate a human-readable explanation."""
        evidence_text = "\n".join(f"- {e}" for e in evidence)
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a fact-checking assistant. Generate a concise explanation (max 500 characters) for why the statement is valid or invalid based on the evidence."},
                {"role": "user", "content": f"Statement: {statement}\nValid: {is_valid}\n\nEvidence:\n{evidence_text}"}
            ]
        )
        return response.choices[0].message.content[:500]  # Truncate to 500 chars 