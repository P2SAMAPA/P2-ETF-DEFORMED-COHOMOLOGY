import numpy as np
import warnings
warnings.filterwarnings("ignore")

def deformed_cohomology_score(returns, macro_value, max_distance=0.8, k_neighbors=5):
    """
    Compute per-ETF score: node degree in the k-NN graph, scaled by macro factor.
    Each ETF is connected to its k closest neighbours (by correlation distance).
    """
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    n = dist.shape[0]
    # Build k‑NN graph (symmetric)
    adj = np.zeros((n, n), dtype=int)
    k = min(k_neighbors, n-1)
    for i in range(n):
        # Indices of k smallest distances (excluding self)
        nearest = np.argsort(dist[i])[1:k+1]
        adj[i, nearest] = 1
    # Make symmetric
    adj = np.maximum(adj, adj.T)
    # Node degree
    degrees = np.sum(adj, axis=1)
    # Macro factor (VIX normalized, baseline 20)
    macro_factor = max(0.1, min(3.0, macro_value / 20.0))
    scores = degrees * macro_factor
    tickers = returns.columns
    return {ticker: float(scores[i]) for i, ticker in enumerate(tickers)}

def deformed_cohomology_aggregate_score(returns, macro_df, primary_macro="VIX", max_distance=0.8, steps=5):
    """
    Wrapper for train.py: compute per-ETF scores using k‑NN graph + macro scaling.
    """
    if primary_macro not in macro_df.columns:
        macro_value = 10.0
    else:
        macro_value = macro_df[primary_macro].iloc[-1]
    # Use a fixed k=5 for k‑NN; you can also make it configurable
    scores = deformed_cohomology_score(returns, macro_value, k_neighbors=5)
    return scores
