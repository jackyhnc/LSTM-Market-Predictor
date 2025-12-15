# LSTM Market Predictor

An automated S&P 500 trading system that combines LSTM deep learning, Topological Data Analysis (TDA), and FinBERT sentiment analysis to generate buy/sell signals and execute paper trades via the Alpaca Markets API.

<img width="1465" height="758" alt="image" src="https://github.com/user-attachments/assets/35f4aa47-602a-4417-9fbf-d2b4c7dbec81" />

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
  - [1. Market Data Ingestion](#1-market-data-ingestion)
  - [2. Technical Indicators](#2-technical-indicators)
  - [3. Topological Data Analysis (TDA)](#3-topological-data-analysis-tda)
  - [4. News Sentiment Analysis](#4-news-sentiment-analysis)
  - [5. LSTM Inference](#5-lstm-inference)
  - [6. Crisis Detection](#6-crisis-detection)
  - [7. Trade Execution](#7-trade-execution)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
  - [Prerequisites](#prerequisites)
  - [Backend (Python)](#backend-python)
  - [Frontend (.NET Blazor)](#frontend-net-blazor)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Dashboard](#dashboard)
- [Caching Behavior](#caching-behavior)

---

## Overview

This system runs a full end-to-end automated trading pipeline on a schedule:

1. Downloads 90 days of price/volume history for all tracked S&P 500 tickers via `yfinance`
2. Computes technical indicators (RSI-14, 21-day volatility, volume z-score, log returns)
3. Computes topological features using Vietoris-Rips persistent homology (Wasserstein distance between consecutive persistence diagrams)
4. Fetches recent news headlines for each ticker from Google News RSS and scores them with FinBERT
5. Feeds all features into a trained LSTM model to predict expected log return for the next period
6. Detects potential market crises via Wasserstein distance spikes
7. Executes paper trades on Alpaca Markets: buys predicted high-return tickers, sells predicted low-return tickers already in the portfolio

The Blazor frontend provides a live dashboard showing portfolio equity history and a "Run Predictions" button that triggers the full pipeline on demand.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Blazor Frontend                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Home.razor                                              │   │
│  │  - Portfolio equity chart (MudBlazor line chart)         │   │
│  │  - "Run Predictions" button  →  POST /predict            │   │
│  │  - "Load Portfolio History"  →  POST /portfolio_history  │   │
│  │  - Crisis alert banner (when TDA detects topology spike) │   │
│  └──────────────────────────────────────────────────────────┘   │
│                        .NET 9 / Blazor Server                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP  (http://127.0.0.1:8000)
┌──────────────────────────────▼──────────────────────────────────┐
│                     Python FastAPI Backend                       │
│                                                                  │
│  POST /predict                                                   │
│    │                                                             │
│    ├─► yfinance  ──► OHLCV data (90 days, all tickers)          │
│    ├─► RSI-14, Volatility-21, Volume Z-score, Log Return        │
│    ├─► Ripser (Vietoris-Rips)  ──► Wasserstein distance series  │
│    ├─► Google News RSS  ──► FinBERT  ──► Sentiment score [-1,1] │
│    ├─► LSTM model (lstm.keras)  ──► predicted log return/ticker │
│    ├─► Crisis detection  ──► Wasserstein spike analysis         │
│    └─► Alpaca Paper Trading API  ──► buy / sell / hold          │
│                                                                  │
│  POST /portfolio_history  ──► Alpaca account equity history     │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It Works

### 1. Market Data Ingestion

The backend downloads 90 calendar days of OHLCV (Open, High, Low, Close, Volume) data for all tickers in `valid_tickers.csv` using the `yfinance` library.

```
start_date = today - 90 days
end_date   = today
```

The 90-day window is sized to provide:
- 30 rows for the prediction window (LSTM input)
- 1 row for the first NaN from log return computation
- ~30 rows consumed by the 21-day rolling indicator windows
- ~30 rows consumed by the 30-day Wasserstein window
- Buffer for weekends/holidays (roughly 5/7 of calendar days are trading days)

Data is cached to a date-stamped CSV (`sp500_data_{start}_to_{end}.csv`) so repeated same-day calls skip the download.

---

### 2. Technical Indicators

For each ticker the following features are computed on the Close price and Volume columns:

| Feature | Formula | Window |
|---|---|---|
| `RSI_14` | Wilder's RSI | 14 days |
| `Log_Return` | `ln(1 + pct_change)` | — |
| `Volatility_21` | Rolling std of log returns | 21 days |
| `Volume_Z` | `(Volume - μ) / σ` z-score | 21 days |

**RSI computation:**

```
gain  = mean of positive daily changes over window
loss  = mean of negative daily changes over window
RS    = gain / loss   (inf where loss = 0)
RSI   = 100 - (100 / (1 + RS))
```

These features, along with the raw OHLCV columns and the TDA Wasserstein feature (see below), form the 10-dimensional input vector per timestep fed into the LSTM.

---

### 3. Topological Data Analysis (TDA)

This is the most distinctive part of the feature pipeline. TDA extracts shape-based information from the joint distribution of S&P 500 log returns, which can capture systemic market stress that per-ticker indicators miss.

**Pipeline:**

1. Build a matrix of shape `(n_tickers, n_days)` from all tickers' log returns
2. Slide a 30-day window across the time axis
3. For each window, treat the `(30_days × n_tickers)` matrix as a point cloud in high-dimensional space
4. Fit a **Vietoris-Rips filtration** using `ripser` with `maxdim=2` — this computes the persistent homology (H0, H1, H2) of the point cloud at all scales
5. Compute the **Wasserstein distance** between the H1 persistence diagrams of consecutive windows using `persim`

The Wasserstein distance series measures how much the "topological shape" of the market changed between each consecutive 30-day window. A spike in this series indicates a rapid structural change in cross-asset correlations — historically correlated with market crises (2008, 2020 COVID crash, etc.).

```
Wasserstein[t] = W(PD(window[t-1]), PD(window[t]))
```

This single scalar per day is appended as the `Wasserstein` feature to the LSTM input.

**Betti numbers** (B0, B1, B2 — counting connected components, loops, voids in the point cloud) were also computed but are currently excluded from the LSTM input features.

---

### 4. News Sentiment Analysis

For each ticker, recent news headlines are fetched from the Google News RSS feed using a search query:

```
"{company name} after:{start_date} before:{end_date}"
```

Short tickers (≤ 3 characters) have `" stock"` appended to disambiguate (e.g., `"A stock"` instead of just `"A"`).

Headlines are mapped to trading dates and then scored using **FinBERT** (`ProsusAI/finbert`), a BERT model fine-tuned on financial text. Each headline is classified as:

| Label | Numeric value |
|---|---|
| `positive` | +1 |
| `neutral` | 0 |
| `negative` | -1 |

The sentiment score for each (ticker, date) pair is attached as the `Sentiment` feature column in the finance DataFrame and fed into the LSTM.

Both the raw articles and the scored sentiment results are cached to date-stamped CSVs to avoid re-running the slow BERT inference on repeated calls.

---

### 5. LSTM Inference

The LSTM model (`lstm.keras`) takes a 3D input tensor of shape:

```
(n_tickers, 30, 10)
  │          │    └─ features per timestep:
  │          │        Close, High, Low, Open,
  │          │        RSI_14, Volatility_21,
  │          │        Volume, Volume_Z,
  │          │        Sentiment, Wasserstein
  │          └─ 30-day lookback window
  └─ one sample per ticker
```

Features are scaled with a `StandardScaler` (`scaler.pkl`) that was fitted on the training dataset.

The model outputs one scalar per ticker — the predicted log return for the next trading day. Results are saved to `results_predictions_{start}_to_{end}.csv`.

---

### 6. Crisis Detection

After inference, the full Wasserstein distance time series (from the 90-day window) is analyzed:

```
crisis = (recent_wasserstein > μ + 2σ)  OR  (recent_wasserstein > 95th percentile)
```

If either condition is true, `crisis_status = True` is returned to the frontend, which shows a warning banner advising caution. The trading logic still executes regardless — the crisis flag is informational only.

---

### 7. Trade Execution

Trading is executed through the **Alpaca Paper Trading API** (`https://paper-api.alpaca.markets`). All trades use paper money.

**Logic per ticker:**

```
if predicted_log_return >= 0.004  (≈ +0.4% expected return):
    if not already holding:
        buy $1,000 worth (market order, day)
    else:
        hold (skip)
else:
    if currently holding:
        sell entire position (market order, day)
    else:
        skip
```

Before executing, `has_trades_today()` checks the Alpaca activity log for any fills that already occurred today (US/Eastern timezone). If trades were already made today, the entire trading block is skipped to prevent duplicate executions.

---

## Project Structure

```
LSTM-Market-Predictor/
│
├── README.md
├── .gitignore
├── SP500_tickernames.txt          # Full list of S&P 500 ticker symbols
│
├── model/                         # Training artifacts and notebooks
│   ├── model.ipynb                # LSTM model architecture and training
│   ├── prepare_training_data.ipynb # Feature engineering for training set
│   ├── test_inference_model.ipynb  # Inference validation notebook
│   ├── test.ipynb
│   ├── lstm.keras                 # Trained model weights (gitignored)
│   ├── lstm-old.keras             # Previous model version (gitignored)
│   ├── model_lstm_nosentiment.h5  # Ablation: model without sentiment (gitignored)
│   ├── scaler.pkl                 # Fitted StandardScaler (gitignored)
│   ├── symbol_to_security.pkl     # Ticker → company name mapping (gitignored)
│   ├── X_all.npy.zip              # Training features (gitignored)
│   └── y_all.npy                  # Training targets (gitignored)
│
└── src/
    ├── frontend/                  # .NET 9 Blazor Server app
    │   ├── frontend.csproj
    │   ├── Program.cs             # DI setup, HTTP client configuration
    │   ├── appsettings.json       # App config (logging, allowed hosts)
    │   └── Components/
    │       ├── App.razor
    │       ├── Routes.razor
    │       ├── _Imports.razor
    │       ├── Layout/
    │       │   ├── MainLayout.razor
    │       │   └── NavMenu.razor
    │       └── Pages/
    │           ├── Home.razor     # Main dashboard page
    │           └── Error.razor
    │
    └── model-backend/             # Python FastAPI server
        ├── server.py              # All inference, trading, and API logic
        ├── requirements.txt       # Python dependencies
        ├── valid_tickers.csv      # List of tickers currently being tracked
        └── .env                   # Alpaca credentials (never commit this)
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- .NET 9 SDK
- An [Alpaca Markets](https://alpaca.markets) account (free paper trading account is sufficient)
- The trained model files: `lstm.keras`, `scaler.pkl`, `symbol_to_security.pkl`

---

### Backend (Python)

**1. Create a virtual environment and install dependencies:**

```bash
cd src/model-backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Place the required model files in `src/model-backend/`:**

```
src/model-backend/
├── lstm.keras
├── scaler.pkl
└── symbol_to_security.pkl
```

**3. Create the `.env` file with your Alpaca paper trading credentials:**

```bash
# src/model-backend/.env
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

Get these from your Alpaca dashboard under **Paper Trading → API Keys**.

**4. Start the server:**

```bash
cd src/model-backend
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`. You can explore the auto-generated docs at `http://127.0.0.1:8000/docs`.

---

### Frontend (.NET Blazor)

**1. Restore dependencies and run:**

```bash
cd src/frontend
dotnet run
```

The dashboard will be available at `https://localhost:5001` (or the port shown in the terminal output).

---

## Configuration

### Backend port / URL

The backend URL used by the frontend is configured in `src/frontend/appsettings.json`:

```json
{
  "Backend": {
    "BaseUrl": "http://127.0.0.1:8000"
  }
}
```

Change this if you run the backend on a different host or port.

### Tracked tickers

Edit `src/model-backend/valid_tickers.csv` to change which S&P 500 tickers are tracked:

```csv
ticker
AAPL
MSFT
NVDA
...
```

### Trading parameters

The following constants are hardcoded in `server.py` inside `alpaca_trading()`:

| Parameter | Default | Description |
|---|---|---|
| `buy_threshold` | `0.004` | Minimum predicted log return to trigger a buy (~0.4%) |
| `buy_amount_usd` | `1000` | USD notional per buy order |

### Indicator windows

Also hardcoded in `predict_buy_sell()`:

| Parameter | Default | Description |
|---|---|---|
| `window_rsi` | `14` | RSI lookback period (days) |
| `window_vol` | `21` | Volatility and volume z-score rolling window (days) |
| `window_wasserstein` | `30` | Sliding window for Wasserstein distance computation (days) |
| `eps` | `0.5` | Scale parameter for Betti number computation |
| `maxdim` | `2` | Maximum homology dimension for Rips filtration |

---

## Running the App

1. Start the Python backend: `uvicorn server:app --host 127.0.0.1 --port 8000`
2. Start the Blazor frontend: `dotnet run` (from `src/frontend/`)
3. Open the dashboard in your browser (default `https://localhost:5001`)
4. Click **Load Portfolio History** to view current Alpaca paper portfolio equity
5. Click **Run Predictions** to trigger the full pipeline (data download → features → inference → trades)

> **Note:** The prediction pipeline is long-running. It downloads data for hundreds of tickers, runs BERT inference on thousands of news headlines, and runs LSTM inference on all tickers. Expect the "Run Predictions" call to take anywhere from **10 to 30+ minutes** on the first run. Subsequent same-day calls use cached data and complete much faster.

---

## API Reference

All endpoints are served by the FastAPI backend at `http://127.0.0.1:8000`.

Interactive API docs (Swagger UI) are available at `http://127.0.0.1:8000/docs`.

---

### `GET /`

Health check / sanity ping.

**Response:**
```json
{ "message": "Hello, World!" }
```

---

### `POST /predict`

Runs the full prediction and trading pipeline end-to-end:
1. Downloads/loads market data
2. Computes technical indicators
3. Computes TDA features
4. Fetches and scores news sentiment
5. Runs LSTM inference
6. Detects crisis
7. Executes Alpaca paper trades (buy/sell/hold)

**Request body:** none

**Response:**
```json
{
  "predictions": [
    { "Ticker": "AAPL", "Prediction": 0.0073 },
    { "Ticker": "MSFT", "Prediction": -0.0012 },
    ...
  ],
  "crisis_status": false
}
```

| Field | Type | Description |
|---|---|---|
| `predictions` | array | List of `{Ticker, Prediction}` objects. `Prediction` is the raw LSTM output (predicted log return). |
| `crisis_status` | boolean | `true` if the most recent Wasserstein distance exceeded 2 standard deviations above the mean or the 95th percentile of the current 90-day window. |

---

### `POST /portfolio_history`

Fetches Alpaca paper portfolio equity history for a given time period.

**Request body:**
```json
{ "period": "1D" }
```

| `period` | Timeframe returned | Granularity |
|---|---|---|
| `"1H"` | Last 1 hour | 5-minute bars |
| `"1D"` | Last 1 day | 1-hour bars |
| `"1W"` | Last 1 week | Daily bars |
| `"1M"` | Last 1 month | Weekly bars |
| `"1Y"` | Last 1 year | Monthly bars |

**Response (success):**
```json
[
  { "date": "2025-11-01", "portfolio_value": 100234.56 },
  { "date": "2025-11-02", "portfolio_value": 101102.33 },
  ...
]
```

**Response (error):**
```json
{ "error": "error message string" }
```

---

## Model Details

### Architecture

The LSTM model was trained using Keras/TensorFlow. The input shape is `(30, 10)` — a 30-day lookback window with 10 features per timestep. The model outputs a single scalar (predicted log return).

The 10 input features per day, per ticker:

| # | Feature | Source |
|---|---|---|
| 1 | `Close` | yfinance |
| 2 | `High` | yfinance |
| 3 | `Low` | yfinance |
| 4 | `Open` | yfinance |
| 5 | `RSI_14` | Computed from Close |
| 6 | `Volatility_21` | Rolling std of log returns |
| 7 | `Volume` | yfinance |
| 8 | `Volume_Z` | Rolling z-score of Volume |
| 9 | `Sentiment` | FinBERT score on news headlines |
| 10 | `Wasserstein` | TDA: Wasserstein distance between consecutive H1 persistence diagrams |

All features are standardized using the `StandardScaler` fitted on the training data (`scaler.pkl`).

### Training

See `model/model.ipynb` for the full training pipeline. The model was trained on historical S&P 500 data. Training features were prepared using `model/prepare_training_data.ipynb`.

### Files

| File | Description |
|---|---|
| `lstm.keras` | Current production model weights |
| `lstm-old.keras` | Previous model version (kept for rollback) |
| `model_lstm_nosentiment.h5` | Ablation model trained without the sentiment feature |
| `scaler.pkl` | `sklearn.preprocessing.StandardScaler` fitted on training data |
| `symbol_to_security.pkl` | `dict` mapping ticker symbol → company name (used for news queries) |

---

## Dashboard

The Blazor frontend (`Home.razor`) provides:

- **Period selector** — choose 1H / 1D / 1W / 1M / 1Y for portfolio history granularity
- **Load Portfolio History** — fetches and charts the Alpaca paper portfolio equity curve
- **Run Predictions** — triggers the full inference + trading pipeline, then auto-refreshes the chart
- **Crisis alert** — a dismissible yellow banner shown when `crisis_status = true`
- **Error alert** — a red banner shown on any HTTP or parsing error
- **Line chart** — interactive MudBlazor chart with configurable width, height, line thickness, label rotation, and data markers
- **Progress bar** — shown while any long-running operation is in flight; buttons are disabled to prevent double-submission

---

## Caching Behavior

The backend uses local CSV caches to avoid redundant expensive operations. Cache files are keyed by the date range and live in `src/model-backend/`:

| Cache file | Contents | When skipped |
|---|---|---|
| `sp500_data_{start}_to_{end}.csv` | Raw OHLCV data from yfinance | File exists for today's date range |
| `articles_data_{start}_to_{end}.csv` | News headlines per ticker per date | File exists for today's date range |
| `finbert_sentiment_df_{start}_to_{end}.csv` | FinBERT sentiment labels (pre-numeric) | File exists for today's date range |
| `results_predictions_{start}_to_{end}.csv` | LSTM prediction output | Not cached — always recomputed |

All cache files are gitignored. The date range changes daily (based on `today - 90 days`), so old caches naturally stop being used and can be deleted manually to free disk space.
