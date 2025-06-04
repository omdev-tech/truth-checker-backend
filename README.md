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
   - ElevenLabs API key (for speech-to-text functionality)

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Environment configuration:
   ```bash
   # Copy the environment template
   cp env.example .env
   
   # Edit .env and add your API keys:
   # - ELEVENLABS_API_KEY: Required for speech-to-text functionality
   # - OPENAI_API_KEY: Required for AI fact-checking
   # - ANTHROPIC_API_KEY: Optional, for Claude AI provider
   ```

5. Run tests:
   ```bash
   pytest
   ```

6. Start development server:
   ```bash
   uvicorn truth_checker.api.app:app --reload
   ```

## Configuration

The project uses environment variables for configuration. Copy `env.example` to `.env` and adjust values:

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# AI Provider Settings
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Speech-to-Text Provider Settings (REQUIRED)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# MCP Settings
WIKIPEDIA_MCP_URL=http://wikipedia.mcp/
```

**Note**: The `ELEVENLABS_API_KEY` is required for the speech-to-text functionality. You can get an API key from [ElevenLabs](https://elevenlabs.io/).

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