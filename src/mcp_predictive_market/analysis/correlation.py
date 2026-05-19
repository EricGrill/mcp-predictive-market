"""Market correlation analysis for prediction markets."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from mcp_predictive_market.schema import Market, PricePoint


@dataclass
class CorrelationResult:
    """Correlation between two markets."""

    market_a_id: str
    market_b_id: str
    market_a_title: str
    market_b_title: str
    correlation: float  # Pearson correlation (-1 to 1)
    confidence: float  # Based on data overlap
    sample_size: int  # Number of shared data points
    lead_lag: int  # Positive = a leads b, negative = b leads a


@dataclass
class MarketCluster:
    """A cluster of correlated markets."""

    id: str
    markets: list[str]  # Market IDs
    avg_internal_correlation: float
    representative_market: str  # Most central market


class CorrelationAnalyzer:
    """Analyze correlations between prediction markets."""

    def __init__(self, min_history_points: int = 3) -> None:
        """Initialize analyzer.

        Args:
            min_history_points: Minimum price history points needed for correlation
        """
        self._min_points = min_history_points

    def _align_price_histories(
        self,
        history_a: list[PricePoint],
        history_b: list[PricePoint],
        max_gap_minutes: int = 60,
    ) -> tuple[list[float], list[float]]:
        """Align two price histories by timestamp.

        Returns paired (prices_a, prices_b) at matching timestamps.
        """
        # Convert to dict for fast lookup
        dict_a = {p.timestamp.replace(second=0, microsecond=0): p.probability for p in history_a}
        dict_b = {p.timestamp.replace(second=0, microsecond=0): p.probability for p in history_b}

        # Find common timestamps within tolerance
        pairs_a: list[float] = []
        pairs_b: list[float] = []

        for ts_a, prob_a in dict_a.items():
            # Look for matching timestamp in b
            if ts_a in dict_b:
                pairs_a.append(prob_a)
                pairs_b.append(dict_b[ts_a])
            else:
                # Try nearby timestamps within max_gap
                for offset in range(1, max_gap_minutes + 1):
                    ts_plus = ts_a + timedelta(minutes=offset)
                    ts_minus = ts_a - timedelta(minutes=offset)
                    if ts_plus in dict_b:
                        pairs_a.append(prob_a)
                        pairs_b.append(dict_b[ts_plus])
                        break
                    if ts_minus in dict_b:
                        pairs_a.append(prob_a)
                        pairs_b.append(dict_b[ts_minus])
                        break

        return pairs_a, pairs_b

    def _calculate_correlation(self, x: list[float], y: list[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) < 2 or len(x) != len(y):
            return 0.0

        x_arr = np.array(x)
        y_arr = np.array(y)

        # Check for zero variance
        if np.std(x_arr) == 0 or np.std(y_arr) == 0:
            return 0.0

        try:
            corr = np.corrcoef(x_arr, y_arr)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    def _calculate_lead_lag(
        self,
        history_a: list[PricePoint],
        history_b: list[PricePoint],
        max_lag: int = 5,
    ) -> int:
        """Calculate lead/lag relationship between two markets.

        Returns:
            Positive integer = market_a leads market_b by N time steps
            Negative integer = market_b leads market_a by N time steps
            0 = no clear lead/lag
        """
        if len(history_a) < max_lag + 2 or len(history_b) < max_lag + 2:
            return 0

        # Extract aligned price series
        pairs_a, pairs_b = self._align_price_histories(history_a, history_b)
        if len(pairs_a) < max_lag + 2:
            return 0

        a_arr = np.array(pairs_a)
        b_arr = np.array(pairs_b)

        # Try different lags and find best correlation
        best_corr = abs(np.corrcoef(a_arr, b_arr)[0, 1])
        best_lag = 0

        for lag in range(1, max_lag + 1):
            # A leads B: shift B forward by lag
            if len(a_arr) > lag and len(b_arr) > lag:
                corr_forward = abs(np.corrcoef(a_arr[:-lag], b_arr[lag:])[0, 1])
                if not np.isnan(corr_forward) and corr_forward > best_corr:
                    best_corr = corr_forward
                    best_lag = lag

            # B leads A: shift A forward by lag
            if len(a_arr) > lag and len(b_arr) > lag:
                corr_backward = abs(np.corrcoef(a_arr[lag:], b_arr[:-lag])[0, 1])
                if not np.isnan(corr_backward) and corr_backward > best_corr:
                    best_corr = corr_backward
                    best_lag = -lag

        return best_lag

    def correlation_matrix(
        self,
        markets: list[Market],
        include_lead_lag: bool = True,
    ) -> list[CorrelationResult]:
        """Compute correlation matrix for a set of markets.

        Args:
            markets: Markets with price_history populated
            include_lead_lag: Whether to compute lead/lag analysis

        Returns:
            List of correlation results (one per unique pair)
        """
        results: list[CorrelationResult] = []
        seen_pairs: set[tuple[str, str]] = set()

        for i, market_a in enumerate(markets):
            for market_b in markets[i + 1 :]:
                pair = tuple(sorted([market_a.id, market_b.id]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Skip if either market lacks sufficient history
                if (
                    len(market_a.price_history) < self._min_points
                    or len(market_b.price_history) < self._min_points
                ):
                    continue

                # Align histories and calculate correlation
                prices_a, prices_b = self._align_price_histories(
                    market_a.price_history, market_b.price_history
                )

                if len(prices_a) < self._min_points:
                    continue

                correlation = self._calculate_correlation(prices_a, prices_b)

                # Calculate lead/lag if requested
                lead_lag = 0
                if include_lead_lag:
                    lead_lag = self._calculate_lead_lag(
                        market_a.price_history, market_b.price_history
                    )

                # Confidence based on sample size
                confidence = min(1.0, len(prices_a) / 10.0)

                results.append(
                    CorrelationResult(
                        market_a_id=market_a.id,
                        market_b_id=market_b.id,
                        market_a_title=market_a.title,
                        market_b_title=market_b.title,
                        correlation=correlation,
                        confidence=confidence,
                        sample_size=len(prices_a),
                        lead_lag=lead_lag,
                    )
                )

        # Sort by absolute correlation (strongest first)
        results.sort(key=lambda r: abs(r.correlation), reverse=True)
        return results

    def find_clusters(
        self,
        markets: list[Market],
        min_correlation: float = 0.5,
    ) -> list[MarketCluster]:
        """Find clusters of correlated markets.

        Args:
            markets: Markets with price_history
            min_correlation: Minimum correlation to be in same cluster

        Returns:
            List of market clusters
        """
        correlations = self.correlation_matrix(markets, include_lead_lag=False)

        # Build adjacency list
        adjacency: dict[str, set[str]] = {m.id: set() for m in markets}
        for corr in correlations:
            if abs(corr.correlation) >= min_correlation:
                adjacency[corr.market_a_id].add(corr.market_b_id)
                adjacency[corr.market_b_id].add(corr.market_a_id)

        # Find connected components (clusters)
        visited: set[str] = set()
        clusters: list[MarketCluster] = []
        cluster_id = 0

        for market_id in adjacency:
            if market_id in visited:
                continue

            # BFS to find connected component
            cluster_markets: list[str] = []
            queue = [market_id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                cluster_markets.append(current)
                queue.extend(adjacency[current] - visited)

            if len(cluster_markets) > 1:
                # Find representative (most connected)
                connections = {m: len(adjacency[m] & set(cluster_markets)) for m in cluster_markets}
                representative = max(connections, key=connections.get)

                # Calculate average internal correlation
                internal_corrs = [
                    c.correlation
                    for c in correlations
                    if c.market_a_id in cluster_markets and c.market_b_id in cluster_markets
                ]
                avg_corr = np.mean(internal_corrs) if internal_corrs else 0.0

                clusters.append(
                    MarketCluster(
                        id=f"cluster_{cluster_id}",
                        markets=cluster_markets,
                        avg_internal_correlation=float(avg_corr),
                        representative_market=representative,
                    )
                )
                cluster_id += 1

        # Sort by size (largest first)
        clusters.sort(key=lambda c: len(c.markets), reverse=True)
        return clusters

    def diversification_suggestions(
        self,
        markets: list[Market],
        target_portfolio_size: int = 5,
    ) -> dict[str, Any]:
        """Suggest a diversified set of markets.

        Args:
            markets: Available markets with price_history
            target_portfolio_size: Desired number of markets in portfolio

        Returns:
            Suggested portfolio and reasoning
        """
        correlations = self.correlation_matrix(markets, include_lead_lag=False)

        if not markets:
            return {"portfolio": [], "reasoning": "No markets available"}

        # Build correlation lookup
        corr_map: dict[tuple[str, str], float] = {}
        for c in correlations:
            corr_map[(c.market_a_id, c.market_b_id)] = c.correlation
            corr_map[(c.market_b_id, c.market_a_id)] = c.correlation

        # Start with highest-volume market as anchor
        sorted_by_volume = sorted(
            [m for m in markets if m.volume is not None],
            key=lambda m: m.volume,
            reverse=True,
        )
        if not sorted_by_volume:
            sorted_by_volume = markets

        portfolio: list[Market] = [sorted_by_volume[0]]

        # Greedily add least-correlated markets
        for _ in range(target_portfolio_size - 1):
            best_candidate: Market | None = None
            best_min_correlation = float('inf')

            for candidate in markets:
                if candidate in portfolio:
                    continue

                # Find minimum correlation with existing portfolio
                min_corr = float('inf')
                for held in portfolio:
                    pair = tuple(sorted([candidate.id, held.id]))
                    corr = abs(corr_map.get(pair, 0.0))
                    if corr < min_corr:
                        min_corr = corr

                # Prefer candidate with lowest max correlation to portfolio
                if min_corr < best_min_correlation:
                    best_min_correlation = min_corr
                    best_candidate = candidate

            if best_candidate:
                portfolio.append(best_candidate)

        # Identify hedges (negatively correlated pairs)
        hedges: list[dict[str, Any]] = []
        for c in correlations:
            if c.correlation < -0.3:  # Negative correlation threshold
                hedges.append(
                    {
                        "market_a": c.market_a_id,
                        "market_b": c.market_b_id,
                        "correlation": c.correlation,
                        "reason": "Negative correlation provides natural hedge",
                    }
                )

        return {
            "portfolio": [
                {
                    "id": m.id,
                    "title": m.title,
                    "platform": m.platform,
                    "probability": m.probability,
                    "volume": m.volume,
                }
                for m in portfolio
            ],
            "reasoning": f"Selected {len(portfolio)} markets with minimal cross-correlation",
            "hedges": hedges,
            "total_available": len(markets),
        }

    def format_correlation_matrix(
        self,
        correlations: list[CorrelationResult],
        markets: list[Market],
    ) -> str:
        """Format correlation results as a readable matrix string."""
        market_ids = [m.id for m in markets]
        n = len(market_ids)

        # Build matrix
        matrix = np.zeros((n, n))
        for c in correlations:
            i = market_ids.index(c.market_a_id)
            j = market_ids.index(c.market_b_id)
            matrix[i][j] = c.correlation
            matrix[j][i] = c.correlation

        # Format as string
        lines = ["Correlation Matrix:"]
        lines.append("=" * 60)

        # Header
        header = "Market".ljust(25)
        for i in range(min(5, n)):  # Show first 5 for readability
            short_name = markets[i].title[:15]
            header += f" | {short_name:>15}"
        lines.append(header)
        lines.append("-" * 60)

        # Rows
        for i in range(min(5, n)):
            row = markets[i].title[:23].ljust(25)
            for j in range(min(5, n)):
                if i == j:
                    row += " |        1.000"
                else:
                    row += f" | {matrix[i][j]:>15.3f}"
            lines.append(row)

        if n > 5:
            lines.append(f"\n... and {n - 5} more markets")

        return "\n".join(lines)
