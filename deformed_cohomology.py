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
        return 1, 0  # one component, zero loops
    try:
        import gudhi as gd
        rips = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_distance)
        st = rips.create_simplex_tree(max_dimension=max_dim)
        st.persistence()
        betti0 = st.persistent_betti_numbers(0, 0)[0]  # number of components
        betti1 = st.persistent_betti_numbers(1, 0)[0] if max_dim >= 1 else 0
        return betti0, betti1
    except:
        # Fallback: use graph Laplacian to approximate Betti-1
        # Build adjacency: edge if distance < max_distance
        adj = (dist_matrix < max_distance).astype(float)
        np.fill_diagonal(adj, 0)
        # Number of connected components (Betti-0)
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(adj, directed=False)
        # Approximate Betti-1 = number of edges - number of vertices + number of components
        n_edges = np.sum(adj) / 2
        betti1 = max(0, n_edges - n + n_components)
        return n_components, betti1

def deformed_betti_derivative(dist_matrix, macro_factor, steps=5, max_distance=0.8):
    """
    Compute the derivative of Betti-1 with respect to a deformation parameter
    that scales the distance matrix.
    """
    # Deformation: distance' = distance ^ (1 + α * macro_factor) or distance * exp(γ * macro)
    # We'll use a simple scaling: scale = exp(γ * macro), but γ = 1
    # For a given macro factor, compute distance scale
    # We'll vary the deformation by adding a small perturbation to macro
    epsilon = 0.01
    # Baseline macro factor (current macro)
    macro0 = macro_factor
    macro_plus = macro0 + epsilon
    macro_minus = macro0 - epsilon
    
    # Apply deformation: distance_scaled = distance * exp(β * macro)
    # We'll set β = 1 for simplicity
    dist_plus = dist_matrix * np.exp(macro_plus)
    dist_minus = dist_matrix * np.exp(macro_minus)
    # Cap at max_distance to avoid huge values
    dist_plus = np.clip(dist_plus, 0, max_distance)
    dist_minus = np.clip(dist_minus, 0, max_distance)
    
    _, betti1_plus = betti_number(dist_plus, max_distance)
    _, betti1_minus = betti_number(dist_minus, max_distance)
    derivative = (betti1_plus - betti1_minus) / (2 * epsilon)
    return derivative

def deformed_cohomology_score(returns, macro_value, max_distance=0.8, steps=5):
    """
    Compute per-ETF score: derivative of Betti-1 w.r.t macro deformation.
    """
    # Build distance matrix from correlation distance
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Normalise macro factor (e.g., VIX scaled to range 0..1)
    macro_factor = max(0.0, min(1.0, macro_value / 100.0))  # VIX typically 0-100
    derivative = deformed_betti_derivative(dist, macro_factor, steps, max_distance)
    return derivative

def deformed_cohomology_aggregate_score(returns, macro_df, primary_macro="VIX", max_distance=0.8, steps=5):
    """
    Compute aggregated score for a single ETF's universe.
    Since returns here is the full universe returns DataFrame, we compute a per-ETF score
    by treating each ETF's removal? Actually the Betti derivative is global to the universe.
    To get a per-ETF score, we compute the derivative of Betti-1 and then assign
    to each ETF its contribution based on eigenvector centrality or something.
    For simplicity, we return the same derivative for all ETFs, which is not useful.
    Instead, we will compute the change in Betti-1 when each ETF is removed (delete vertex).
    This gives a per-ETF sensitivity: how much the topology changes if that ETF is absent.
    """
    if len(returns.columns) < 3:
        return {ticker: 0.0 for ticker in returns.columns}
    # Get current macro factor
    if primary_macro not in macro_df.columns:
        macro_value = 0.0
    else:
        macro_value = macro_df[primary_macro].iloc[-1]
    # Baseline Betti-1 with all ETFs
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    macro_factor = max(0.0, min(1.0, macro_value / 100.0))
    _, betti1_base = betti_number(dist, max_distance)
    # Per‑ETF: remove each ETF and recompute Betti-1
    scores = {}
    tickers = returns.columns
    for i, ticker in enumerate(tickers):
        # Remove ETF i
        indices = [j for j in range(len(tickers)) if j != i]
        if len(indices) < 2:
            scores[ticker] = 0.0
            continue
        dist_removed = dist[np.ix_(indices, indices)]
        _, betti1_removed = betti_number(dist_removed, max_distance)
        # Score = change in Betti-1 when removed (negative means removal reduces loops)
        delta = betti1_base - betti1_removed
        scores[ticker] = delta
    return scores
