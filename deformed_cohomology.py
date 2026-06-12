import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def compute_composite_macro_factor(macro_df, returns_etf=None):
    """
    Compute a composite macro factor as a weighted sum of all macro variables.
    If returns_etf is provided, weights are estimated via ridge regression of next-day return on macro.
    Otherwise, use equal weights.
    """
    if returns_etf is not None and len(returns_etf) == len(macro_df):
        # Standardise macro
        scaler = StandardScaler()
        macro_scaled = scaler.fit_transform(macro_df)
        # Ridge regression to estimate importance of each macro for predicting next-day return
        # Use lagged macro to predict next-day return
        target = returns_etf[1:]
        macro_lagged = macro_scaled[:-1] if len(macro_scaled) == len(target) else macro_scaled[:len(target)]
        if len(target) > len(macro_lagged):
            target = target[:len(macro_lagged)]
        if len(target) > 5:
            ridge = Ridge(alpha=1.0)
            ridge.fit(macro_lagged, target)
            weights = np.abs(ridge.coef_)
            weights = weights / (weights.sum() + 1e-8)
        else:
            weights = np.ones(macro_df.shape[1]) / macro_df.shape[1]
    else:
        # Equal weights
        weights = np.ones(macro_df.shape[1]) / macro_df.shape[1]
        scaler = StandardScaler()
        macro_scaled = scaler.fit_transform(macro_df)
    return weights, scaler

def composite_macro_factor_at_time(macro_row, weights, scaler):
    """Compute composite macro factor for a single row of macro data."""
    macro_scaled = scaler.transform(macro_row.reshape(1, -1)).flatten()
    factor = np.dot(weights, macro_scaled)
    # Exponentiate to ensure positivity and amplify differences
    return np.exp(factor)

def deformed_cohomology_score(returns, macro_df, k_neighbors=5, etf_returns_for_weights=None):
    """
    Compute per-ETF score: node degree in the k-NN graph of returns,
    scaled by a composite macro factor from all macro variables.
    """
    # Build distance matrix from correlation
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    n = dist.shape[0]
    # k‑NN graph
    adj = np.zeros((n, n), dtype=int)
    k = min(k_neighbors, n-1)
    for i in range(n):
        nearest = np.argsort(dist[i])[1:k+1]
        adj[i, nearest] = 1
    adj = np.maximum(adj, adj.T)
    degrees = np.sum(adj, axis=1)
    # Compute composite macro factor using all macro variables
    if etf_returns_for_weights is not None:
        # Use the provided ETF returns to estimate macro weights
        weights, scaler = compute_composite_macro_factor(macro_df, etf_returns_for_weights)
    else:
        weights, scaler = compute_composite_macro_factor(macro_df, None)
    # Current macro (last row)
    macro_row = macro_df.iloc[-1].values
    macro_factor = composite_macro_factor_at_time(macro_row, weights, scaler)
    # Score = degree * macro_factor
    scores = degrees * macro_factor
    tickers = returns.columns
    return {ticker: float(scores[i]) for i, ticker in enumerate(tickers)}

def deformed_cohomology_aggregate_score(returns, macro_df, primary_macro="VIX", max_distance=0.8, steps=5):
    """
    Wrapper for train.py: compute per-ETF scores using k‑NN graph and composite macro factor.
    Note: primary_macro is ignored; we use all macros.
    """
    if macro_df is None or macro_df.empty:
        return {ticker: 0.0 for ticker in returns.columns}
    # Use the full returns (universe) to estimate macro weights (global across ETFs)
    # We'll pass the first ETF's returns for weight estimation (or aggregate)
    # To make it more robust, we can average across ETFs? We'll use the mean return of the universe.
    universe_returns = returns.mean(axis=1).values
    scores = deformed_cohomology_score(returns, macro_df, k_neighbors=5, etf_returns_for_weights=universe_returns)
    return scores
