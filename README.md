# Truth Checker

An AI-powered fact-checking system with Model Context Protocol (MCP) integration.

## Project Structure

The project follows hexagonal architecture (ports and adapters) principles:

```
truth_checker/
├── domain/           # Core domain logic and entities
│   ├── models/      # Domain models and value objects
│   ├── ports/       # Interface definitions
│   └── services/    # Domain services
├── application/     # Application services and use cases
│   ├── services/    # Application-specific services
│   └── interfaces/  # Application interfaces
├── infrastructure/  # External adapters and implementations
│   ├── ai/         # AI model adapters (ChatGPT, Claude)
│   ├── mcp/        # MCP provider adapters
│   ├── persistence/# Database and storage adapters
│   └── web/        # Web API adapters
└── config/         # Configuration management
```

## Development Setup

1. Requirements:
   - Python 3.10 or higher
   - pip
   - virtualenv (recommended)

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run tests:
   ```bash
   pytest
   ```

5. Start development server:
   ```bash
   uvicorn truth_checker.infrastructure.web.api:app --reload
   ```

## Configuration

The project uses environment variables for configuration. Copy `.env.example` to `.env` and adjust values:

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# AI Provider Settings
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# MCP Settings
WIKIPEDIA_MCP_URL=http://wikipedia.mcp/
```

## Development Guidelines

1. Follow TDD approach - write tests first
2. Use type hints everywhere
3. Format code with black and isort
4. Run mypy for type checking
5. Follow SOLID principles
6. Keep functions small and focused
7. Document all public interfaces

## License

MIT License - see LICENSE file for details 