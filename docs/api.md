# Model Backend API

Base URL: `http://localhost:8000`

## Endpoints

### GET /
Health check.

**Response**
```json
{"message": "Hello, World!"}
```

### POST /predict
Runs the full inference pipeline: fetches market data, computes features (RSI, TDA,
sentiment), runs LSTM predictions for all tracked tickers, and executes Alpaca paper
trades.

**Response**
```json
{
  "predictions": [
    {"Ticker": "AAPL", "Prediction": 0.0062},
    ...
  ],
  "crisis_status": false
}
```

### POST /portfolio_history
Returns portfolio equity history from Alpaca.

**Request body**
```json
{"period": "1D"}
```
Valid periods: `1H`, `1D`, `1W`, `1M`, `1Y`.

**Response** — array of `{date, portfolio_value}` objects.
