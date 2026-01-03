# Development Notes

## TDA Feature Engineering

After experimenting with Betti numbers (β₀, β₁, β₂) as direct features, Wasserstein
distance between consecutive persistence diagrams turned out to be a better signal.
It captures topology *change* rather than absolute topology, which aligns better with
detecting regime shifts.

The Betti number columns are kept in commented-out code for reference.

## Sentiment Pipeline

FinBERT (ProsusAI/finbert) outperformed the original yiyanghkust/finbert-tone variant
in informal back-testing. Batch size 8 works well on Apple Silicon (MPS) without
triggering MPS memory issues.

## LSTM Architecture

Input shape: `(30 timesteps, 10 features)`.  
The scaler is fitted on training data and saved to `scaler.pkl`; the same scaler is
applied at inference time so the feature distributions stay consistent.

## Known Issues / TODO

- Crisis detection threshold is heuristic (2 std above mean). A rolling z-score with
  a longer lookback might be more stable.
- The Google News RSS feed occasionally returns no results for small-cap tickers.
  Fallback to neutral sentiment (0) is the current behaviour.
