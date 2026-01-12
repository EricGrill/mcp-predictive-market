# MCP Prediction Market Server

An MCP (Model Context Protocol) server that aggregates prediction market data from multiple platforms, enabling Claude and other AI assistants to query, compare, and track prediction markets.

## Features

The server provides 8 MCP tools:

- **search_markets** - Search for prediction markets across all platforms
- **get_market_odds** - Get current odds for a specific market
- **list_categories** - List available market categories
- **browse_category** - Browse markets by category
- **track_market** - Add a market to your watchlist
- **get_tracked_markets** - Get all tracked markets with current prices
- **find_arbitrage** - Find price discrepancies across platforms
- **compare_platforms** - Side-by-side odds comparison for similar markets

## Supported Platforms

- **Manifold Markets** - manifold.markets
- **Polymarket** - polymarket.com
- **Metaculus** - metaculus.com
- **PredictIt** - predictit.org
- **Kalshi** - kalshi.com

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-predictive-market.git
cd mcp-predictive-market

# Install with uv
uv sync

# Or install with pip
pip install -e .
```

## Configuration

### Claude Code

Add to your Claude Code MCP settings (`~/.config/claude-code/mcp.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "prediction-market": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-predictive-market", "python", "-m", "mcp_predictive_market.server"]
    }
  }
}
```

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "prediction-market": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-predictive-market", "python", "-m", "mcp_predictive_market.server"]
    }
  }
}
```

## Usage Examples

### Search for Markets

```
Search for prediction markets about AI
```

Returns markets from all platforms matching "AI".

### Get Specific Market Odds

```
Get the current odds for Manifold market abc123
```

### Compare Platforms

```
Compare odds across platforms for "Will there be a recession?"
```

Shows side-by-side probabilities from different platforms for similar questions.

### Find Arbitrage Opportunities

```
Find arbitrage opportunities with at least 10% spread
```

Identifies markets where the same question has significantly different odds on different platforms.

### Track Markets

```
Track the Polymarket market xyz789 as "Election 2024"
```

Adds a market to your watchlist for easy monitoring.

## Development

### Setup

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest -v

# Run a specific test
uv run pytest tests/test_server.py -v
```

### Project Structure

```
src/mcp_predictive_market/
├── server.py           # MCP server entry point
├── tools.py            # Tool handler implementations
├── schema.py           # Unified market data models
├── errors.py           # Custom exceptions
├── adapters/           # Platform-specific adapters
│   ├── base.py         # Adapter protocol
│   ├── manifold.py
│   ├── polymarket.py
│   ├── metaculus.py
│   ├── predictit.py
│   └── kalshi.py
└── analysis/           # Market analysis modules
    ├── matching.py     # Cross-platform market matching
    └── arbitrage.py    # Arbitrage detection
```

## License

MIT
