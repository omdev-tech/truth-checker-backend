"""Live testing script for Wikipedia MCP adapter."""

import asyncio
from typing import Dict, List

import pytest
from rich import print

from truth_checker.infrastructure.mcp.wikipedia_adapter import (
    WikipediaConfig,
    WikipediaMCPAdapter,
)


async def test_fact_checking():
    """Test fact checking with various statements."""
    
    # Initialize adapter
    config = WikipediaConfig(
        base_url="https://en.wikipedia.org/w/api.php",
        timeout=30.0,
    )
    adapter = WikipediaMCPAdapter(config=config)
    
    try:
        # Initialize connection
        print("\n[bold blue]Initializing Wikipedia adapter...[/bold blue]")
        await adapter.initialize()
        print("[green]✓ Adapter initialized successfully[/green]")
        
        # Test statements
        statements = [
            "Paris is the capital of France",
            "The Earth is flat",
            "Python was created by Guido van Rossum",
            "The moon is made of cheese",
        ]
        
        for statement in statements:
            print(f"\n[bold yellow]Checking statement:[/bold yellow] {statement}")
            
            # Validate fact
            result = await adapter.validate_fact(statement)
            
            # Print results
            print(f"[bold]Valid:[/bold] {result.is_valid}")
            print(f"[bold]Confidence:[/bold] {result.confidence:.2f}")
            
            if result.supporting_evidence:
                print("\n[bold green]Supporting Evidence:[/bold green]")
                for evidence in result.supporting_evidence[:2]:  # Show first 2
                    print(f"- {evidence[:200]}...")
                    
            if result.contradicting_evidence:
                print("\n[bold red]Contradicting Evidence:[/bold red]")
                for evidence in result.contradicting_evidence[:2]:  # Show first 2
                    print(f"- {evidence[:200]}...")
                    
            if result.source_urls:
                print("\n[bold blue]Sources:[/bold blue]")
                for url in result.source_urls[:3]:  # Show first 3
                    print(f"- {url}")
                    
            print("\n[bold cyan]Metadata:[/bold cyan]")
            for key, value in result.metadata.items():
                print(f"- {key}: {value}")
                
    except Exception as e:
        print(f"[bold red]Error:[/bold red] {e}")
        raise
        
    finally:
        # Cleanup
        await adapter.shutdown()
        print("\n[blue]Adapter shutdown complete[/blue]")


async def test_multilingual():
    """Test fact checking in different languages."""
    
    adapter = WikipediaMCPAdapter()
    
    try:
        # Initialize
        await adapter.initialize()
        
        # Test statements in different languages
        tests = [
            ("en", "London is the capital of England"),
            ("es", "Madrid es la capital de España"),
            ("fr", "Paris est la capitale de la France"),
            ("de", "Berlin ist die Hauptstadt von Deutschland"),
        ]
        
        for lang, statement in tests:
            print(f"\n[bold yellow]Testing {lang}:[/bold yellow] {statement}")
            
            try:
                # Set language
                await adapter.set_language(lang)
                print(f"[green]✓ Language set to {lang}[/green]")
                
                # Validate fact
                result = await adapter.validate_fact(statement)
                
                print(f"[bold]Valid:[/bold] {result.is_valid}")
                print(f"[bold]Confidence:[/bold] {result.confidence:.2f}")
                
                if result.supporting_evidence:
                    print("\n[bold green]Evidence:[/bold green]")
                    print(result.supporting_evidence[0][:200] + "...")
                    
            except Exception as e:
                print(f"[bold red]Error testing {lang}:[/bold red] {e}")
                
    finally:
        await adapter.shutdown()


if __name__ == "__main__":
    # Run tests
    print("[bold magenta]Running Wikipedia MCP Adapter Tests[/bold magenta]\n")
    
    asyncio.run(test_fact_checking())
    print("\n" + "="*50 + "\n")
    asyncio.run(test_multilingual()) 