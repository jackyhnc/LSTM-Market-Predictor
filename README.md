# LSTM Market Predictor: A Multi-Modal Deep Learning System for Automated Equity Trading

<div align="center">

<img width="1465" height="758" alt="Dashboard" src="https://github.com/user-attachments/assets/35f4aa47-602a-4417-9fbf-d2b4c7dbec81" />

*Figure 1. Live trading dashboard showing portfolio equity curve with crisis detection banner.*

</div>

---

## Abstract

This project presents a fully automated equity trading system targeting the S&P 500 universe. The system fuses three distinct signal modalities — classical technical analysis, topological data analysis (TDA) of cross-asset return manifolds, and transformer-based news sentiment — into a unified feature vector that is consumed by a Long Short-Term Memory (LSTM) recurrent neural network. The model outputs a predicted log return for each ticker over the next trading day, which is thresholded to generate binary buy/hold/sell signals that are then executed as paper trades through the Alpaca Markets API. A Blazor Server frontend provides a real-time dashboard of portfolio equity history and on-demand pipeline invocation. The design emphasizes the novel combination of persistent homology as a market-regime feature alongside per-asset technicals and NLP-derived sentiment, with the hypothesis that topological features capture systemic cross-asset correlation shifts that are invisible to single-asset indicators.

---

## Table of Contents

- [1. Introduction & Motivation](#1-introduction--motivation)
- [2. System Architecture](#2-system-architecture)
- [3. Feature Engineering Pipeline](#3-feature-engineering-pipeline)
  - [3.1 Market Data Ingestion](#31-market-data-ingestion)
  - [3.2 Technical Indicators](#32-technical-indicators)
  - [3.3 Topological Data Analysis](#33-topological-data-analysis)
  - [3.4 News Sentiment Analysis](#34-news-sentiment-analysis)
- [4. LSTM Model](#4-lstm-model)
  - [4.1 Architecture](#41-architecture)
  - [4.2 Input Tensor Construction](#42-input-tensor-construction)
  - [4.3 Target Variable](#43-target-variable)
- [5. Trading Strategy & Signal Logic](#5-trading-strategy--signal-logic)
  - [5.1 Crisis Detection](#51-crisis-detection)
  - [5.2 Order Execution](#52-order-execution)
- [6. System Components & Setup](#6-system-components--setup)
  - [6.1 Project Structure](#61-project-structure)
  - [6.2 Prerequisites](#62-prerequisites)
  - [6.3 Backend Setup](#63-backend-setup)
  - [6.4 Frontend Setup](#64-frontend-setup)
- [7. Configuration Reference](#7-configuration-reference)
- [8. API Reference](#8-api-reference)
- [9. Caching & Performance](#9-caching--performance)
- [10. Dashboard](#10-dashboard)
- [11. Limitations & Future Work](#11-limitations--future-work)

---

## 1. Introduction & Motivation

Quantitative equity trading has historically relied on two broad families of signals: *price-derived technical indicators* and *fundamental/macro data*. The last decade has seen a third pillar emerge — *alternative data*, including satellite imagery, credit card transactions, and web-scraped text. Of these, news sentiment derived from natural language processing has become particularly tractable due to the availability of large pre-trained language models. Simultaneously, the mathematics of algebraic topology has found application in data science through the field of Topological Data Analysis, which can characterize the "shape" of high-dimensional data in ways that traditional statistics cannot.

This system was built to answer a practical question: can the combination of these three distinct information sources — classical technicals, NLP sentiment, and topological market-structure features — be distilled into a single learned representation by an LSTM and turned into a profitable trading strategy?

The key hypotheses motivating the design are:

1. **LSTM over window-based models.** Markets exhibit temporal dependencies that span multiple timescales. An LSTM with a 30-day lookback can, in principle, learn multi-day momentum, reversal, and volatility clustering patterns that a single-day feature vector cannot represent.

2. **Topological features as systemic risk proxies.** During market crises, cross-asset correlations spike and the geometry of the joint return distribution changes dramatically. Persistent homology, measured as the Wasserstein distance between consecutive persistence diagrams of the cross-asset return point cloud, should encode this regime shift as a single scalar that is otherwise absent from per-asset indicators.

3. **Sentiment as a leading indicator.** News headlines reflect the market's information set. FinBERT, trained specifically on financial text, can map sentiment polarity from news to a numerical signal that may lead price action by one or more days.

---

## 2. System Architecture

The system consists of two primary services connected over localhost HTTP, with all ML workloads running in the Python backend.

```
┌──────────────────────────────────────────────────────────────┐
│                     Blazor Server Frontend                    │
│                        (.NET 9)                              │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Home.razor                                            │  │
│  │                                                        │  │
│  │  [Period Selector]  [Load History]  [Run Predictions]  │  │
│  │                                                        │  │
│  │        ┌──────────────────────────────────┐            │  │
│  │        │   Portfolio Equity Line Chart    │            │  │
│  │        │   (MudBlazor, interactive)       │            │  │
│  │        └──────────────────────────────────┘            │  │
│  │                                                        │  │
│  │   ⚠  Crisis Alert  (shown when crisis_status=true)     │  │
│  └────────────────────────────────────────────────────────┘  │
└───────────────────────────┬──────────────────────────────────┘
                            │ HTTP (127.0.0.1:8000)
              POST /predict │ POST /portfolio_history
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                   Python FastAPI Backend                      │
│                                                              │
│  ┌─────────────┐  ┌───────────┐  ┌──────────────────────┐   │
│  │  yfinance   │  │  ripser   │  │  ProsusAI/finbert    │   │
│  │  OHLCV data │  │  Rips TDA │  │  News sentiment NLP  │   │
│  └──────┬──────┘  └─────┬─────┘  └──────────┬───────────┘   │
│         │               │                   │               │
│         └───────────────┴───────────────────┘               │
│                         │                                    │
│                  ┌──────▼──────┐                             │
│                  │  Feature    │                             │
│                  │  Assembly   │  (30 × 10 per ticker)       │
│                  └──────┬──────┘                             │
│                         │                                    │
│                  ┌──────▼──────┐                             │
│                  │ lstm.keras  │  Keras LSTM                 │
│                  └──────┬──────┘                             │
│                         │ predicted log return per ticker    │
│                  ┌──────▼──────┐                             │
│                  │  Alpaca     │  Paper Trading API          │
│                  │  Trading    │  buy / sell / hold          │
│                  └─────────────┘                             │
└──────────────────────────────────────────────────────────────┘
```

*Figure 2. High-level system architecture showing data flow from external sources through the feature pipeline, LSTM inference, and trade execution.*

---

## 3. Feature Engineering Pipeline

### 3.1 Market Data Ingestion

For each prediction run, 90 calendar days of OHLCV data are downloaded for all tickers in `valid_tickers.csv` via the `yfinance` library. The date window is:

$$t_{\text{start}} = t_{\text{today}} - 90 \text{ days}, \quad t_{\text{end}} = t_{\text{today}}$$

The 90-day window is sized to satisfy the following constraints simultaneously, accounting for the fact that only approximately $\frac{5}{7}$ of calendar days are trading days:

| Constraint | Days Required |
|---|---|
| LSTM prediction window | 30 trading days |
| First row dropped (NaN from log return) | 1 |
| 21-day rolling indicator warm-up | 21 |
| 30-day Wasserstein window warm-up | 30 |
| Calendar-to-trading-day buffer | ~13 |
| **Total (calendar days)** | **~90** |

The raw download produces a multi-index DataFrame with columns `(ticker, field)`. This is immediately flattened to a single-level index `{ticker}_{field}` for downstream processing, then re-indexed on the `Date` column.

---

### 3.2 Technical Indicators

Four scalar features are computed per ticker per day from the raw OHLCV data. Combined with the four raw price columns (O, H, L, C) and Volume, these form the per-asset component of the input feature vector.

#### Relative Strength Index (RSI-14)

The RSI, introduced by J. Welles Wilder, is a bounded momentum oscillator in $[0, 100]$. It measures the magnitude of recent gains relative to recent losses:

$$\text{RS}_t = \frac{\text{SMMA}(\text{gain}, 14)_t}{\text{SMMA}(\text{loss}, 14)_t}$$

$$\text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t}$$

where $\text{SMMA}$ denotes the smoothed moving average, and gain/loss are the positive and negative daily price changes respectively. When the denominator is zero (no losses in the window), $\text{RS} = \infty$ and $\text{RSI} = 100$, representing a perfectly trending up asset.

Values above 70 conventionally indicate overbought conditions; values below 30 indicate oversold.

#### Log Return

The continuously compounded one-day log return is:

$$r_t = \ln\!\left(1 + \frac{P_t - P_{t-1}}{P_{t-1}}\right) = \ln\frac{P_t}{P_{t-1}}$$

Log returns are preferred over simple returns because they are additive across time and closer to normally distributed for use in statistical models.

#### Realized Volatility (21-day)

Annualized daily realized volatility is estimated as the rolling standard deviation of log returns:

$$\sigma_t = \sqrt{\frac{1}{W-1} \sum_{i=0}^{W-1} (r_{t-i} - \bar{r}_t)^2}, \quad W = 21$$

A 21-day window corresponds to approximately one calendar month of trading days, capturing medium-term volatility regime.

#### Volume Z-Score

To make volume comparable across tickers with vastly different float sizes, raw volume is standardized relative to its own recent history:

$$Z_t^{\text{vol}} = \frac{V_t - \mu_t^{\text{vol}}}{\sigma_t^{\text{vol}}}, \quad \text{where } \mu, \sigma \text{ are 21-day rolling statistics}$$

A high positive $Z_t^{\text{vol}}$ indicates anomalously heavy trading activity, which often precedes or accompanies significant price moves.

---

### 3.3 Topological Data Analysis

This is the most mathematically novel component of the pipeline. The core intuition is that during normal markets, the joint distribution of cross-asset returns has a stable "shape." During crises — such as the 2008 financial crisis or the March 2020 COVID crash — correlations among all assets suddenly spike toward 1, fundamentally changing the geometry of that distribution. Persistent homology provides a principled, parameter-free way to quantify this shape and detect when it changes.

#### Point Cloud Construction

At each time step $t$, a sliding window of width $W = 30$ days is extracted from the log-return matrix:

$$\mathbf{M}_{t} \in \mathbb{R}^{W \times N}, \quad W = 30, \; N = \text{number of tickers}$$

Each row $\mathbf{M}_{t}^{(d)} \in \mathbb{R}^{N}$ is a vector of all tickers' log returns on day $d$. The 30 rows of $\mathbf{M}_t$ form a *point cloud* of 30 points in $N$-dimensional return space.

#### Vietoris-Rips Filtration

The Vietoris-Rips filtration is a nested sequence of simplicial complexes $\mathcal{R}(\epsilon)$ built from the point cloud. At scale $\epsilon$:

$$\sigma \in \mathcal{R}(\epsilon) \iff d(x_i, x_j) \leq \epsilon \quad \forall\, x_i, x_j \in \sigma$$

That is, a simplex is included when all its vertices are within distance $\epsilon$ of each other. As $\epsilon$ increases from 0 to $\infty$, topological features (connected components, loops, voids) are born and die. The set of (birth, death) pairs for $k$-dimensional features is called the **persistence diagram** $\text{PD}_k$.

```
ε small:          ε medium:         ε large:
  ·  ·  ·           ·──·  ·           ·──·
  · ·   ·           · ·───·           ·──·──·
  ·  ·  ·           ·  ·  ·           ·──·──·

Many components   Loops form       All connected
  H0 births       H1 births         H1 dies
```

*Figure 3. Schematic of the Vietoris-Rips filtration. As ε grows, points connect into edges, triangles, and tetrahedra.*

The `ripser` library computes this filtration efficiently up to `maxdim=2`, yielding diagrams for $H_0$ (connected components), $H_1$ (loops/cycles), and $H_2$ (voids).

#### Wasserstein Distance as a Temporal Feature

To obtain a single time-series feature, the **Wasserstein distance** (also called the Earth Mover's Distance) between consecutive $H_1$ persistence diagrams is computed:

$$W_t = W_2\!\left(\text{PD}_1^{(t)},\; \text{PD}_1^{(t-1)}\right)$$

The $p$-Wasserstein distance between two persistence diagrams is:

$$W_p(\text{PD}_1, \text{PD}_2) = \left[ \inf_{\gamma \in \Gamma} \sum_{x \in \text{PD}_1} \lVert x - \gamma(x) \rVert_\infty^p \right]^{1/p}$$

where $\Gamma$ is the set of all bijections between points in $\text{PD}_1$ and points in $\text{PD}_2$ (extended with the diagonal to handle unequal sizes), and $\lVert \cdot \rVert_\infty$ is the $L^\infty$ norm. Intuitively, $W_t$ measures the minimum "cost" to transport one diagram's point mass to match the other's.

A large $W_t$ means the topological structure of the joint return distribution changed significantly between window $t-1$ and window $t$ — a signal of market structural instability.

The resulting scalar time series $\{W_t\}$ is aligned to the date index and appended as the `Wasserstein` feature column shared across all tickers.

#### Why $H_1$ (Loops)?

While $H_0$ captures clustering (number of connected components in the return graph) and $H_2$ captures voids, $H_1$ captures *cycles* — closed loops in the data manifold. In market terms, $H_1$ features correspond to approximate cyclic relationships among return vectors. These tend to be the features most sensitive to changes in correlation structure: when correlations spike in a crisis, the loops in the point cloud collapse, producing a large Wasserstein distance.

---

### 3.4 News Sentiment Analysis

For each ticker, a Google News RSS query is constructed:

```
"{company_name} after:{start_date} before:{end_date}"
```

For single-letter or very short tickers (e.g., `A` for Agilent Technologies), `" stock"` is appended to disambiguate from common English words. Company names are resolved from `symbol_to_security.pkl`, a pre-built dictionary mapping each ticker symbol to its full security name.

#### FinBERT

Headlines are scored using **ProsusAI/finbert**, a BERT-base model fine-tuned on the Financial PhraseBank dataset of 4,840 manually annotated financial news sentences. The model outputs a three-class probability distribution over `{positive, neutral, negative}`, and the argmax label is extracted:

$$\hat{y} = \arg\max_{c \in \{+, 0, -\}} P(c \mid \text{headline})$$

Labels are then mapped to a numeric score for use as a continuous feature:

| Label | Numeric |
|---|---|
| `positive` | $+1$ |
| `neutral` | $0$ |
| `negative` | $-1$ |

When multiple headlines exist for a ticker on a given date, the first occurrence is used (after deduplication by date). For dates with no articles, the sentiment value is `NaN`, which propagates into the LSTM as a zero after scaling.

Headlines are processed in batches of 8 using the Hugging Face `pipeline` API, with periodic garbage collection to manage memory usage of the ~440M parameter BERT model. Both the raw headlines and the scored sentiment DataFrames are cached to CSVs keyed by the date range.

---

## 4. LSTM Model

### 4.1 Architecture

The model (`lstm.keras`) is a Long Short-Term Memory network, a class of recurrent neural network specifically designed to learn long-range sequential dependencies via its gating mechanism.

An LSTM cell at time step $t$ maintains a hidden state $\mathbf{h}_t$ and a cell state $\mathbf{c}_t$, updated by three learned gates:

$$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(forget gate)}$$

$$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(input gate)}$$

$$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(output gate)}$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

where $\sigma$ is the sigmoid function and $\odot$ denotes element-wise multiplication. The forget gate $\mathbf{f}_t$ controls how much of the previous cell state is retained, allowing the network to learn when to "reset" its memory — important for capturing the difference between trending and mean-reverting regimes.

See `model/model.ipynb` for the full architecture definition, layer sizes, and training configuration.

### 4.2 Input Tensor Construction

For each ticker, the most recent 30 rows of the feature DataFrame are extracted and stacked into a window. All tickers are batched into a 3D tensor:

$$\mathbf{X} \in \mathbb{R}^{N_{\text{tickers}} \times T \times F}, \quad T = 30, \; F = 10$$

Before stacking, the 9 per-asset features (all except Wasserstein, which is market-wide) are standardized using the `StandardScaler` fitted during training:

$$\tilde{x}_{i,t,f} = \frac{x_{i,t,f} - \mu_f}{\sigma_f}$$

where $\mu_f$ and $\sigma_f$ are the training-set mean and standard deviation for feature $f$. This ensures that each feature contributes comparably regardless of its natural scale (e.g., RSI lives in $[0,100]$ while log returns are in $[-0.1, 0.1]$).

The 10 input features per timestep, in their standardized form, are:

| # | Feature | Raw Scale | Captures |
|---|---|---|---|
| 1 | `Close` | USD | Absolute price level |
| 2 | `High` | USD | Intraday range upper bound |
| 3 | `Low` | USD | Intraday range lower bound |
| 4 | `Open` | USD | Gap-open dynamics |
| 5 | `RSI_14` | $[0, 100]$ | Momentum / overbought-oversold |
| 6 | `Volatility_21` | Decimal | Volatility regime |
| 7 | `Volume` | Shares | Liquidity / participation |
| 8 | `Volume_Z` | Std devs | Anomalous volume |
| 9 | `Sentiment` | $\{-1, 0, +1\}$ | News flow direction |
| 10 | `Wasserstein` | Positive real | Market topology / systemic risk |

### 4.3 Target Variable

The model is trained to predict the one-day-ahead log return:

$$y_{i,t} = r_{i, t+1} = \ln \frac{P_{i,t+1}}{P_{i,t}}$$

At inference time, the model outputs $\hat{y}_i$ for each ticker $i$. This is a regression output — the predicted continuously compounded return for the next trading day.

---

## 5. Trading Strategy & Signal Logic

### 5.1 Crisis Detection

After inference, a secondary analysis is run on the Wasserstein distance series to detect potential market crises. Let $\{W_t\}$ be the full series of Wasserstein distances over the 90-day window. The most recent value $W_T$ is compared against two thresholds:

$$\text{crisis} = \mathbb{1}\!\left[W_T > \mu_W + 2\sigma_W\right] \;\lor\; \mathbb{1}\!\left[W_T > Q_{0.95}(\{W_t\})\right]$$

where $\mu_W$ and $\sigma_W$ are the mean and standard deviation of the full series, and $Q_{0.95}$ is the 95th percentile. The OR of these two conditions is used because the standard-deviation threshold is sensitive to the distributional shape of $W$, while the percentile threshold is more robust to outliers from prior crises in the window.

When a crisis is detected, `crisis_status = true` is returned to the frontend and a warning banner is displayed. Importantly, **trading is not automatically suspended** when a crisis is detected — the flag is informational, allowing a human operator to interpret the signal in context.

A historical analysis of this indicator suggests that significant spikes in $H_1$ Wasserstein distance correspond to macro stress events:

```
Wasserstein
Distance
    │
  ▲ │               ╭╮
    │         ╭╮    ││
    │    ╭╮   ││    ││   ╭╮
    │    ││   ││    ││   ││
────┼────┼┼───┼┼────┼┼───┼┼───── time
    │  2008  2011  2020  2022
    │  GFC   EU   COVID  Rate
    │       Crisis Crash Hike
```

*Figure 4. Schematic illustration of Wasserstein distance spikes at known market stress events. Actual magnitudes vary by universe and window size.*

### 5.2 Order Execution

The signal generation and execution logic is straightforward. Given the set of predicted log returns $\{\hat{y}_i\}$ and the current Alpaca portfolio positions $\mathcal{P}$:

$$\text{action}(i) = \begin{cases} \text{BUY} & \hat{y}_i \geq \tau \;\land\; i \notin \mathcal{P} \\ \text{HOLD} & \hat{y}_i \geq \tau \;\land\; i \in \mathcal{P} \\ \text{SELL} & \hat{y}_i < \tau \;\land\; i \in \mathcal{P} \\ \text{SKIP} & \hat{y}_i < \tau \;\land\; i \notin \mathcal{P} \end{cases}$$

where $\tau = 0.004$ (approximately $+0.4\%$ expected log return) is the buy threshold.

For BUY orders, the number of shares is computed as:

$$q_i = \left\lfloor \frac{B}{P_i^{\text{last}}} \right\rfloor$$

where $B = \$1{,}000$ is the fixed notional per trade and $P_i^{\text{last}}$ is the last available closing price. All orders are submitted as **market orders** with `time_in_force = day` — they execute at the open of the next trading session and expire if unfilled.

#### Duplicate-Trade Guard

To prevent re-executing the same day's strategy if the pipeline is invoked multiple times, the Alpaca activity log is queried for `FILL` events prior to execution:

$$\text{execute} \iff \lnot \exists \, a \in \text{Activities} : \text{date}(a) = t_{\text{today}}^{\text{EST}}$$

The date comparison is performed in US/Eastern timezone, consistent with equity market trading hours.

---

## 6. System Components & Setup

### 6.1 Project Structure

```
LSTM-Market-Predictor/
│
├── README.md
├── .gitignore
├── SP500_tickernames.txt              # Reference list of S&P 500 symbols
│
├── model/                             # Training artifacts and research notebooks
│   ├── model.ipynb                    # LSTM architecture definition and training loop
│   ├── prepare_training_data.ipynb    # Full feature engineering for the training set
│   ├── test_inference_model.ipynb     # Inference validation and output inspection
│   ├── test.ipynb                     # Scratch exploration
│   ├── lstm.keras                     # Production model weights (gitignored)
│   ├── lstm-old.keras                 # Prior model version for rollback (gitignored)
│   ├── model_lstm_nosentiment.h5      # Ablation: trained without sentiment feature (gitignored)
│   ├── scaler.pkl                     # Fitted StandardScaler (gitignored)
│   ├── symbol_to_security.pkl         # Ticker → company name mapping (gitignored)
│   ├── X_all.npy.zip                  # Training feature tensor (gitignored)
│   └── y_all.npy                      # Training target vector (gitignored)
│
└── src/
    ├── frontend/                      # .NET 9 Blazor Server application
    │   ├── frontend.csproj
    │   ├── Program.cs                 # Service registration and HTTP client setup
    │   ├── appsettings.json           # Logging config and backend URL
    │   └── Components/
    │       ├── App.razor
    │       ├── Routes.razor
    │       ├── _Imports.razor
    │       ├── Layout/
    │       │   ├── MainLayout.razor
    │       │   └── NavMenu.razor
    │       └── Pages/
    │           ├── Home.razor         # Dashboard: chart, controls, crisis alert
    │           └── Error.razor
    │
    └── model-backend/                 # Python FastAPI inference server
        ├── server.py                  # All pipeline, inference, and trading logic
        ├── requirements.txt           # Pinned Python dependencies
        ├── valid_tickers.csv          # Active ticker universe
        └── .env                       # Alpaca API credentials (never commit)
```

### 6.2 Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Backend runtime |
| .NET SDK | 9.0 | Frontend runtime |
| Alpaca Markets account | — | Paper trading API (free) |
| Model files | — | `lstm.keras`, `scaler.pkl`, `symbol_to_security.pkl` |

The Python backend has significant hardware requirements for the FinBERT sentiment pass. An Apple Silicon Mac (M-series) will use MPS acceleration. A CUDA GPU will accelerate BERT inference substantially. CPU-only is supported but slow.

### 6.3 Backend Setup

```bash
cd src/model-backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place model files in src/model-backend/
# lstm.keras, scaler.pkl, symbol_to_security.pkl

# Create .env with Alpaca paper trading credentials
# (available at alpaca.markets → Paper Trading → API Keys)
echo "ALPACA_API_KEY=your_key_here" >> .env
echo "ALPACA_SECRET_KEY=your_secret_here" >> .env

# Start the server
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation is auto-generated by FastAPI at `http://127.0.0.1:8000/docs`.

### 6.4 Frontend Setup

```bash
cd src/frontend
dotnet run
```

The dashboard will be available at `https://localhost:5001` (exact port shown in terminal output).

---

## 7. Configuration Reference

### Backend URL

Configured in `src/frontend/appsettings.json`:

```json
{
  "Backend": {
    "BaseUrl": "http://127.0.0.1:8000"
  }
}
```

### Ticker Universe

Edit `src/model-backend/valid_tickers.csv`. Each row after the header is a tracked ticker:

```csv
ticker
AAPL
MSFT
NVDA
```

### Model & Strategy Parameters

| Parameter | Location | Default | Description |
|---|---|---|---|
| `buy_threshold` $\tau$ | `server.py` `alpaca_trading()` | `0.004` | Minimum $\hat{y}_i$ to trigger a buy |
| `buy_amount_usd` $B$ | `server.py` `alpaca_trading()` | `1000` | Notional per buy order (USD) |
| `window_rsi` | `server.py` `predict_buy_sell()` | `14` | RSI lookback (days) |
| `window_vol` | `server.py` `predict_buy_sell()` | `21` | Volatility / volume z-score window (days) |
| `window_wasserstein` $W$ | `server.py` `predict_buy_sell()` | `30` | Sliding window for TDA computation (days) |
| `maxdim` | `server.py` `predict_buy_sell()` | `2` | Max homology dimension for Rips filtration |
| `eps` | `server.py` `predict_buy_sell()` | `0.5` | Scale parameter for Betti number computation |

---

## 8. API Reference

All endpoints served at `http://127.0.0.1:8000`. Full Swagger UI at `/docs`.

---

### `GET /`

Sanity check endpoint.

**Response** `200 OK`:
```json
{ "message": "Hello, World!" }
```

---

### `POST /predict`

Executes the full pipeline: market data → features → LSTM → crisis detection → Alpaca trades.

**Request body:** none

**Response** `200 OK`:
```json
{
  "predictions": [
    { "Ticker": "AAPL", "Prediction": 0.0073 },
    { "Ticker": "MSFT", "Prediction": -0.0012 },
    { "Ticker": "NVDA", "Prediction": 0.0051 }
  ],
  "crisis_status": false
}
```

| Field | Type | Description |
|---|---|---|
| `predictions[].Ticker` | string | Ticker symbol |
| `predictions[].Prediction` | float | Raw LSTM output — predicted log return $\hat{y}_i$ |
| `crisis_status` | boolean | Whether the Wasserstein spike threshold was exceeded |

> **Note:** This endpoint is long-running. First-call latency is 10–30 minutes due to data download, BERT inference across hundreds of tickers' headlines, and LSTM forward pass. Subsequent same-day calls use cached data and return in seconds.

---

### `POST /portfolio_history`

Fetches Alpaca paper account equity history.

**Request body:**
```json
{ "period": "1D" }
```

| `period` | Data Span | Granularity |
|---|---|---|
| `"1H"` | Last hour | 5-minute bars |
| `"1D"` | Last day | 1-hour bars |
| `"1W"` | Last week | Daily bars |
| `"1M"` | Last month | Weekly bars |
| `"1Y"` | Last year | Monthly bars |

**Response** `200 OK` (success):
```json
[
  { "date": "2025-11-01", "portfolio_value": 100234.56 },
  { "date": "2025-11-04", "portfolio_value": 101102.33 }
]
```

**Response** `200 OK` (error):
```json
{ "error": "descriptive error string" }
```

---

## 9. Caching & Performance

The pipeline employs a file-based caching strategy keyed on the date range `{start_date}_to_{end_date}`. Because `end_date = today` and `start_date = today - 90 days`, the cache key changes every calendar day, naturally expiring old entries.

| Cache File | Content | Typical Size | Regeneration Cost |
|---|---|---|---|
| `sp500_data_{start}_to_{end}.csv` | OHLCV for all tickers, 90 days | ~50 MB | ~2–5 min (yfinance API) |
| `articles_data_{start}_to_{end}.csv` | News headlines per ticker per date | ~10 MB | ~20–40 min (Google News RSS) |
| `finbert_sentiment_df_{start}_to_{end}.csv` | FinBERT labels (pre-numeric conversion) | ~5 MB | ~10–30 min (BERT inference) |

Prediction outputs (`results_predictions_*.csv`) are written but not read back — they serve as a persistent audit trail of model outputs over time.

All cache files are gitignored. Old cache files accumulate in `src/model-backend/` and can be deleted freely — they will be regenerated on the next run.

---

## 10. Dashboard

The Blazor frontend (`Home.razor`) renders a live trading dashboard with the following components:

| Component | Description |
|---|---|
| **Period Selector** | Dropdown for `1H / 1D / 1W / 1M / 1Y` portfolio history granularity |
| **Load Portfolio History** | Fetches Alpaca equity history and populates the chart |
| **Run Predictions** | Triggers the full inference pipeline, then auto-refreshes the chart |
| **Progress Bar** | Indeterminate MudBlazor progress bar shown during any in-flight HTTP call |
| **Line Chart** | Interactive MudBlazor chart with configurable width, height, line stroke width, X-axis label rotation, and data markers |
| **Crisis Alert** | Yellow dismissible banner shown when `crisis_status = true` from the backend |
| **Error Alert** | Red dismissible banner shown on any HTTP or JSON parsing failure |

Both action buttons are disabled while a request is in flight, preventing double-submission. On load, `FetchPortfolioHistory()` is called automatically from `OnInitializedAsync()` so the chart is populated immediately on page open.

---

## 11. Limitations & Future Work

**Model staleness.** The LSTM is a static artifact trained on historical data. Market dynamics evolve — volatility regimes shift, correlation structures change, and new macroeconomic factors emerge. The model should be retrained periodically (monthly or quarterly) on a rolling training window to remain calibrated to current market conditions.

**Synchronous long-running endpoint.** The `/predict` endpoint currently blocks for the duration of the full pipeline (up to 30+ minutes). A production-grade implementation should use an async task queue (e.g., Celery with Redis) to immediately return a job ID and allow the client to poll for completion.

**No authentication.** The FastAPI backend exposes trade-execution endpoints with no authentication. Any process on localhost can trigger trades. An API key header or JWT authentication layer should be added before exposing the backend to any network beyond loopback.

**Single sentiment score per ticker per day.** The current implementation takes only the first deduplicated headline per ticker per date. A more robust approach would average sentiment across all articles for that day, weight by source credibility, or use a finer-grained temporal alignment.

**Betti numbers disabled.** The TDA pipeline computes $B_0$, $B_1$, $B_2$ (connected components, loops, voids) but these are excluded from the final feature set. Ablation experiments could determine whether including them improves model performance, and at what computational cost.

**No position sizing.** The current strategy uses a fixed $1,000 notional per trade regardless of predicted return magnitude, volatility, or portfolio concentration. Kelly criterion or volatility-scaled position sizing would produce a more theoretically sound allocation.

**Backtesting.** There is no systematic backtesting framework. The notebooks in `model/` contain inference validation, but a rigorous walk-forward backtest with realistic transaction costs and slippage modeling is needed before drawing conclusions about strategy profitability.
