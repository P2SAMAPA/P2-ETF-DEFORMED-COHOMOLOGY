# Deformed Cohomology – Topological Field Theory for ETFs

Applies deformed cohomology (a topological field theory) to ETF correlation distance matrices. The deformation parameter is the macro state (VIX). The per‑ETF score is the derivative of Betti-1 (number of loops) with respect to macro, measured by the change in loop structure when the ETF is removed.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Rips complex from distance matrix = 1 - |correlation|
- Deformation: distance_scaled = distance * exp(macro_factor)
- Score = Betti-1 change when ETF is removed (topological importance)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-deformed-cohomology-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast, only graph computations)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → ETF's removal significantly changes the loop structure under macro deformation → structurally important / systemic.
- Low score → ETF is topologically redundant.

## Requirements

See `requirements.txt`.
