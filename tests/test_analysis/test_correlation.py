"""Tests for correlation analysis."""
from datetime import datetime, timedelta

import pytest
import numpy as np

from mcp_predictive_market.schema import Market, PricePoint, Outcome
from mcp_predictive_market.analysis.correlation import CorrelationAnalyzer


@pytest.fixture
def analyzer():
    """Create a CorrelationAnalyzer instance."""
    return CorrelationAnalyzer(min_history_points=3)


@pytest.fixture
def correlated_markets():
    """Create two markets with perfectly correlated price histories."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    market_a = Market(
        platform="manifold",
        native_id="abc123",
        url="https://manifold.markets/abc123",
        title="Will it rain?",
        description="Weather prediction",
        category="weather",
        probability=0.6,
        outcomes=[Outcome(name="Yes", probability=0.6)],
        volume=1000,
        created_at=base_time,
        last_fetched=base_time,
        price_history=[
            PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.5 + i * 0.01)
            for i in range(10)
        ],
    )
    
    market_b = Market(
        platform="polymarket",
        native_id="def456",
        url="https://polymarket.com/def456",
        title="Same rain question",
        description="Weather prediction",
        category="weather",
        probability=0.6,
        outcomes=[Outcome(name="Yes", probability=0.6)],
        volume=2000,
        created_at=base_time,
        last_fetched=base_time,
        price_history=[
            PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.5 + i * 0.01)
            for i in range(10)
        ],
    )
    
    return [market_a, market_b]


@pytest.fixture
def uncorrelated_markets():
    """Create two markets with uncorrelated price histories."""
    base_time = datetime(2024, 1, 1, 12, 0, 0)
    
    market_a = Market(
        platform="manifold",
        native_id="abc123",
        url="https://manifold.markets/abc123",
        title="Will it rain?",
        description="Weather prediction",
        category="weather",
        probability=0.6,
        outcomes=[Outcome(name="Yes", probability=0.6)],
        volume=1000,
        created_at=base_time,
        last_fetched=base_time,
        price_history=[
            PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.5 + i * 0.01)
            for i in range(10)
        ],
    )
    
    market_b = Market(
        platform="polymarket",
        native_id="def456",
        url="https://polymarket.com/def456",
        title="Different question",
        description="Politics",
        category="politics",
        probability=0.3,
        outcomes=[Outcome(name="Yes", probability=0.3)],
        volume=5000,
        created_at=base_time,
        last_fetched=base_time,
        price_history=[
            PricePoint(timestamp=base_time + timedelta(hours=i), probability=np.random.random())
            for i in range(10)
        ],
    )
    
    return [market_a, market_b]


class TestCorrelationMatrix:
    """Tests for correlation matrix calculation."""
    
    def test_perfect_correlation(self, analyzer, correlated_markets):
        """Two identical price histories should have correlation ≈ 1.0."""
        result = analyzer.correlation_matrix(correlated_markets)
        
        assert len(result) == 1
        assert result[0].correlation > 0.99
        assert result[0].market_a_id == "manifold:abc123"
        assert result[0].market_b_id == "polymarket:def456"
        
    def test_insufficient_history(self, analyzer):
        """Markets with insufficient history should be skipped."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        market_a = Market(
            platform="manifold",
            native_id="abc123",
            url="https://manifold.markets/abc123",
            title="Test",
            description="Test",
            category="test",
            probability=0.5,
            outcomes=[Outcome(name="Yes", probability=0.5)],
            volume=100,
            created_at=base_time,
            last_fetched=base_time,
            price_history=[
                PricePoint(timestamp=base_time, probability=0.5),
            ],
        )
        
        market_b = Market(
            platform="polymarket",
            native_id="def456",
            url="https://polymarket.com/def456",
            title="Test",
            description="Test",
            category="test",
            probability=0.5,
            outcomes=[Outcome(name="Yes", probability=0.5)],
            volume=100,
            created_at=base_time,
            last_fetched=base_time,
            price_history=[
                PricePoint(timestamp=base_time, probability=0.5),
            ],
        )
        
        result = analyzer.correlation_matrix([market_a, market_b])
        assert len(result) == 0
        
    def test_sample_size_tracking(self, analyzer, correlated_markets):
        """Sample size should reflect number of aligned data points."""
        result = analyzer.correlation_matrix(correlated_markets)
        
        assert result[0].sample_size > 0
        assert result[0].sample_size <= 10
        
    def test_confidence_calculation(self, analyzer, correlated_markets):
        """Confidence should be based on sample size."""
        result = analyzer.correlation_matrix(correlated_markets)
        
        assert 0 <= result[0].confidence <= 1
        assert result[0].confidence == min(1.0, result[0].sample_size / 10.0)


class TestLeadLag:
    """Tests for lead/lag analysis."""
    
    def test_lead_lag_detection(self, analyzer):
        """Should detect when one market leads another."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        # Market A changes, then market B changes 1 hour later
        market_a = Market(
            platform="manifold",
            native_id="lead123",
            url="https://manifold.markets/lead123",
            title="Leading market",
            description="Test",
            category="test",
            probability=0.5,
            outcomes=[Outcome(name="Yes", probability=0.5)],
            volume=1000,
            created_at=base_time,
            last_fetched=base_time,
            price_history=[
                PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.4 + i * 0.02)
                for i in range(10)
            ],
        )
        
        market_b = Market(
            platform="polymarket",
            native_id="lag456",
            url="https://polymarket.com/lag456",
            title="Lagging market",
            description="Test",
            category="test",
            probability=0.5,
            outcomes=[Outcome(name="Yes", probability=0.5)],
            volume=1000,
            created_at=base_time,
            last_fetched=base_time,
            price_history=[
                PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.4 + (i-1) * 0.02)
                for i in range(10)
            ],
        )
        
        result = analyzer.correlation_matrix([market_a, market_b], include_lead_lag=True)
        
        assert len(result) == 1
        # A should lead B (positive lead_lag)
        assert result[0].lead_lag > 0
        
    def test_no_lead_lag_for_sync(self, analyzer, correlated_markets):
        """Synchronous markets should have lead_lag ≈ 0."""
        result = analyzer.correlation_matrix(correlated_markets, include_lead_lag=True)
        
        assert result[0].lead_lag == 0


class TestClusters:
    """Tests for market clustering."""
    
    def test_cluster_detection(self, analyzer, correlated_markets):
        """Highly correlated markets should form a cluster."""
        clusters = analyzer.find_clusters(correlated_markets, min_correlation=0.5)
        
        assert len(clusters) == 1
        assert len(clusters[0].markets) == 2
        assert clusters[0].avg_internal_correlation > 0.5
        
    def test_no_clusters_for_uncorrelated(self, analyzer):
        """Uncorrelated markets shouldn't form clusters at high threshold."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        market_a = Market(
            platform="manifold",
            native_id="abc123",
            url="https://manifold.markets/abc123",
            title="Will it rain?",
            description="Weather prediction",
            category="weather",
            probability=0.6,
            outcomes=[Outcome(name="Yes", probability=0.6)],
            volume=1000,
            created_at=base_time,
            last_fetched=base_time,
            price_history=[
                PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.5 + i * 0.01)
                for i in range(10)
            ],
        )
        
        market_b = Market(
            platform="polymarket",
            native_id="def456",
            url="https://polymarket.com/def456",
            title="Different question",
            description="Politics",
            category="politics",
            probability=0.3,
            outcomes=[Outcome(name="Yes", probability=0.3)],
            volume=5000,
            created_at=base_time,
            last_fetched=base_time,
            price_history=[
                PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.8 if i % 2 == 0 else 0.2)
                for i in range(10)
            ],
        )
        
        clusters = analyzer.find_clusters([market_a, market_b], min_correlation=0.9)
        
        assert len(clusters) == 0
        
    def test_representative_market(self, analyzer, correlated_markets):
        """Cluster should have a representative market."""
        clusters = analyzer.find_clusters(correlated_markets, min_correlation=0.5)
        
        assert clusters[0].representative_market in clusters[0].markets


class TestDiversification:
    """Tests for diversification suggestions."""
    
    def test_portfolio_size(self, analyzer):
        """Portfolio should respect target size when markets are diverse."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        market_a = Market(
            platform="manifold",
            native_id="div1",
            url="https://manifold.markets/div1",
            title="Market A",
            description="Test",
            category="test",
            probability=0.5,
            outcomes=[Outcome(name="Yes", probability=0.5)],
            volume=1000,
            created_at=base_time,
            last_fetched=base_time,
            price_history=[
                PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.3 + i * 0.02)
                for i in range(10)
            ],
        )
        
        market_b = Market(
            platform="polymarket",
            native_id="div2",
            url="https://polymarket.com/div2",
            title="Market B",
            description="Test",
            category="test",
            probability=0.5,
            outcomes=[Outcome(name="Yes", probability=0.5)],
            volume=2000,
            created_at=base_time,
            last_fetched=base_time,
            price_history=[
                PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.7 - i * 0.02)
                for i in range(10)
            ],
        )
        
        result = analyzer.diversification_suggestions(
            [market_a, market_b], target_portfolio_size=2
        )
        
        assert len(result["portfolio"]) == 2
        
    def test_portfolio_diversification(self, analyzer):
        """Portfolio should prefer uncorrelated markets."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        # Create markets with different correlation patterns
        markets = []
        for i in range(5):
            # Alternate between two price patterns
            if i % 2 == 0:
                history = [
                    PricePoint(timestamp=base_time + timedelta(hours=j), probability=0.3 + j * 0.01)
                    for j in range(10)
                ]
            else:
                history = [
                    PricePoint(timestamp=base_time + timedelta(hours=j), probability=0.7 - j * 0.01)
                    for j in range(10)
                ]
            
            market = Market(
                platform="manifold" if i % 2 == 0 else "polymarket",
                native_id=f"market{i}",
                url=f"https://example.com/{i}",
                title=f"Market {i}",
                description="Test",
                category="test",
                probability=0.5,
                outcomes=[Outcome(name="Yes", probability=0.5)],
                volume=1000 * (i + 1),
                created_at=base_time,
                last_fetched=base_time,
                price_history=history,
            )
            markets.append(market)
        
        result = analyzer.diversification_suggestions(markets, target_portfolio_size=3)
        
        assert len(result["portfolio"]) == 3
        # Should include both patterns for diversification
        titles = [m["title"] for m in result["portfolio"]]
        even_markets = [t for t in titles if "Market 0" in t or "Market 2" in t or "Market 4" in t]
        odd_markets = [t for t in titles if "Market 1" in t or "Market 3" in t]
        
        # Should have mix of both patterns
        assert len(even_markets) > 0
        assert len(odd_markets) > 0
        
    def test_hedge_detection(self, analyzer):
        """Should detect negatively correlated pairs as hedges."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        market_a = Market(
            platform="manifold",
            native_id="up123",
            url="https://manifold.markets/up123",
            title="Goes up",
            description="Test",
            category="test",
            probability=0.5,
            outcomes=[Outcome(name="Yes", probability=0.5)],
            volume=1000,
            created_at=base_time,
            last_fetched=base_time,
            price_history=[
                PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.3 + i * 0.05)
                for i in range(10)
            ],
        )
        
        market_b = Market(
            platform="polymarket",
            native_id="down456",
            url="https://polymarket.com/down456",
            title="Goes down",
            description="Test",
            category="test",
            probability=0.5,
            outcomes=[Outcome(name="Yes", probability=0.5)],
            volume=1000,
            created_at=base_time,
            last_fetched=base_time,
            price_history=[
                PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.7 - i * 0.05)
                for i in range(10)
            ],
        )
        
        result = analyzer.diversification_suggestions([market_a, market_b], target_portfolio_size=2)
        
        assert len(result["hedges"]) > 0
        assert result["hedges"][0]["correlation"] < -0.3


class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_markets(self, analyzer):
        """Empty market list should return empty results."""
        result = analyzer.correlation_matrix([])
        assert len(result) == 0
        
    def test_single_market(self, analyzer):
        """Single market should return empty results."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        market = Market(
            platform="manifold",
            native_id="solo123",
            url="https://manifold.markets/solo123",
            title="Solo market",
            description="Test",
            category="test",
            probability=0.5,
            outcomes=[Outcome(name="Yes", probability=0.5)],
            volume=1000,
            created_at=base_time,
            last_fetched=base_time,
            price_history=[
                PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.5)
                for i in range(10)
            ],
        )
        
        result = analyzer.correlation_matrix([market])
        assert len(result) == 0
        
    def test_zero_variance(self, analyzer):
        """Markets with constant prices should return 0 correlation."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        market_a = Market(
            platform="manifold",
            native_id="flat123",
            url="https://manifold.markets/flat123",
            title="Flat market A",
            description="Test",
            category="test",
            probability=0.5,
            outcomes=[Outcome(name="Yes", probability=0.5)],
            volume=1000,
            created_at=base_time,
            last_fetched=base_time,
            price_history=[
                PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.5)
                for i in range(10)
            ],
        )
        
        market_b = Market(
            platform="polymarket",
            native_id="flat456",
            url="https://polymarket.com/flat456",
            title="Flat market B",
            description="Test",
            category="test",
            probability=0.5,
            outcomes=[Outcome(name="Yes", probability=0.5)],
            volume=1000,
            created_at=base_time,
            last_fetched=base_time,
            price_history=[
                PricePoint(timestamp=base_time + timedelta(hours=i), probability=0.5)
                for i in range(10)
            ],
        )
        
        result = analyzer.correlation_matrix([market_a, market_b])
        assert len(result) == 1
        assert result[0].correlation == 0.0
