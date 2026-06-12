import numpy as np
import warnings
warnings.filterwarnings("ignore")

def betti_number(dist_matrix, max_distance=0.8, max_dim=1):
    """
    Compute Betti numbers (β₀ and β₁) from a distance matrix using Rips complex.
    Fallback to graph method if gudhi is not available.
    """
    n = dist_matrix.shape[0]
    if n < 2:
        return 1, 0
    try:
        import gudhi as gd
        rips = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_distance)
        st = rips.create_simplex_tree(max_dimension=max_dim)
        st.persistence()
        betti0 = st.persistent_betti_numbers(0, 0)[0]
        betti1 = st.persistent_betti_numbers(1, 0)[0] if max_dim >= 1 else 0
        return betti0, betti1
    except:
        # Fallback: use graph Laplacian to approximate
        adj = (dist_matrix < max_distance).astype(float)
        np.fill_diagonal(adj, 0)
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(adj, directed=False)
        n_edges = np.sum(adj) / 2
        betti1 = max(0, n_edges - n + n_components)
        return n_components, betti1

def deformed_cohomology_score(returns, macro_value, max_distance=0.8):
    """
    Compute per-ETF score: node degree in the distance graph, scaled by macro factor.
    Higher degree = more central, scaled by macro (e.g., VIX).
    """
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Build adjacency: edge if distance < max_distance
    adj = (dist < max_distance).astype(int)
    # Node degree (number of close neighbours)
    degrees = np.sum(adj, axis=1)
    # Macro factor (VIX normalized, baseline 20)
    macro_factor = max(0.1, min(3.0, macro_value / 20.0))
    scores = degrees * macro_factor
    tickers = returns.columns
    return {ticker: float(scores[i]) for i, ticker in enumerate(tickers)}

def deformed_cohomology_aggregate_score(returns, macro_df, primary_macro="VIX", max_distance=0.8, steps=5):
    """
    Wrapper for train.py: compute per-ETF scores using the simple degree + macro method.
    """
    if primary_macro not in macro_df.columns:
        macro_value = 10.0
    else:
        macro_value = macro_df[primary_macro].iloc[-1]
    scores = deformed_cohomology_score(returns, macro_value, max_distance)
    return scores
