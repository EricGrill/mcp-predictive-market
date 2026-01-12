"""MCP server entry point for prediction market aggregation."""
from mcp.server import Server
from mcp.types import Tool, TextContent

from mcp_predictive_market.adapters.manifold import ManifoldAdapter


def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("mcp-predictive-market")

    # Initialize adapters
    adapters = {
        "manifold": ManifoldAdapter(),
    }

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return [
            Tool(
                name="search_markets",
                description="Search for prediction markets across platforms",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (e.g., 'Will Trump win 2024?')",
                        },
                        "platforms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: filter to specific platforms",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_market_odds",
                description="Get current odds for a specific market",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "platform": {
                            "type": "string",
                            "description": "Platform name (manifold, polymarket, etc.)",
                        },
                        "market_id": {
                            "type": "string",
                            "description": "The market's native ID",
                        },
                    },
                    "required": ["platform", "market_id"],
                },
            ),
            Tool(
                name="list_categories",
                description="List available market categories",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="browse_category",
                description="Browse markets in a specific category",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Category to browse",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max markets to return (default 20)",
                            "default": 20,
                        },
                    },
                    "required": ["category"],
                },
            ),
        ]

    return server


def main() -> None:
    """Run the MCP server."""
    import asyncio
    from mcp.server.stdio import stdio_server

    server = create_server()

    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
