"""Main script for running the truth checker."""

import asyncio
import os
from typing import Any

from .infrastructure.ai.chatgpt_provider import ChatGPTProvider
from .domain.services.fact_checking_service import FactCheckingService


class MCPWikipediaClient:
    """Temporary MCP Wikipedia client until real one is implemented."""
    
    async def search(self, query: str) -> list[str]:
        """Search Wikipedia articles."""
        # This will be replaced with actual MCP implementation
        return [query]
    
    async def summary(self, article: str) -> str:
        """Get article summary."""
        # This will be replaced with actual MCP implementation
        return f"Summary for {article}"


async def main():
    """Run the truth checker."""
    print("Truth Checker - AI-powered fact checking with Wikipedia")
    print("-----------------------------------------------------")
    
    # Initialize components
    ai_provider = ChatGPTProvider()
    await ai_provider.initialize()
    
    mcp_client = MCPWikipediaClient()
    service = FactCheckingService(ai_provider, mcp_client)
    
    try:
        while True:
            # Get statement from user
            statement = input("\nEnter a statement to fact-check (or 'quit' to exit): ")
            if statement.lower() in ('quit', 'exit', 'q'):
                break
                
            print("\nChecking facts...")
            try:
                # Check the statement
                result = await service.check_fact(statement)
                
                # Print results
                print("\nResults:")
                print(f"Valid: {result.is_valid}")
                print(f"Confidence: {result.confidence:.2%}")
                print(f"\nExplanation: {result.explanation}")
                
                print("\nEvidence:")
                for i, evidence in enumerate(result.evidence, 1):
                    print(f"{i}. {evidence}")
                    
            except Exception as e:
                print(f"\nError checking facts: {e}")
                
    finally:
        # Clean up
        await ai_provider.shutdown()


if __name__ == "__main__":
    asyncio.run(main()) 