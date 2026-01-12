"""Tests for MCP tool handlers."""
import pytest
from pytest_httpx import HTTPXMock

from mcp_predictive_market.tools import ToolHandlers
from mcp_predictive_market.adapters.manifold import ManifoldAdapter
from tests.fixtures.manifold_responses import SAMPLE_MARKET, SAMPLE_MARKETS_LIST


class TestSearchMarkets:
    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self, httpx_mock: HTTPXMock):
        """search_markets should return formatted market data."""
        httpx_mock.add_response(
            url="https://api.manifold.markets/v0/search-markets?term=AI&limit=20",
            json=SAMPLE_MARKETS_LIST,
        )

        adapters = {"manifold": ManifoldAdapter()}
        handlers = ToolHandlers(adapters)

        result = await handlers.search_markets(query="AI")

        assert "markets" in result
        assert len(result["markets"]) == 2
        assert result["markets"][0]["title"] == "Will AI pass the Turing test by 2025?"
        assert result["markets"][0]["probability"] == 0.4


class TestGetMarketOdds:
    @pytest.mark.asyncio
    async def test_get_market_odds_success(self, httpx_mock: HTTPXMock):
        """get_market_odds should return market details."""
        httpx_mock.add_response(
            url="https://api.manifold.markets/v0/market/abc123xyz",
            json=SAMPLE_MARKET,
        )

        adapters = {"manifold": ManifoldAdapter()}
        handlers = ToolHandlers(adapters)

        result = await handlers.get_market_odds(
            platform="manifold", market_id="abc123xyz"
        )

        assert result["platform"] == "manifold"
        assert result["probability"] == 0.4
        assert result["title"] == "Will AI pass the Turing test by 2025?"
