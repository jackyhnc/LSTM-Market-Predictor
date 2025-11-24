import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from ripser import Rips
import persim
from keras.models import load_model
from fastapi import FastAPI
import pickle
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import gc
import joblib
import os


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


"""
- predicts buy or sell for all tracked tickers
-then performs buy hold sell logic on alpaca api
"""


@app.get("/predict_buy_sell")
def predict_buy_sell():
    tickers_df = pd.read_csv("valid_tickers.csv")
    tickers = tickers_df["ticker"].tolist()

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=90)
    # (30 days for prediction, 1 day for first day na log return, 30 days for window + 1 day for first day of window) / (5/7 trading days)

    # caching data seen before (for testing only, not prod)
    period_name = f"sp500_data_{start_date}_to_{end_date}.csv"
    if not os.path.exists(period_name):
        sp500_data = yf.download(
            tickers, start=start_date, end=end_date, group_by="ticker"
        )
        sp500_data.to_csv(period_name, index=True)
    else:
        sp500_data = pd.read_csv(
            period_name, header=[0, 1], index_col=0, parse_dates=True
        )

    df_finance = sp500_data.copy()
    df_finance.columns = [f"{col[0]}_{col[1]}" for col in df_finance.columns]

    df_finance = df_finance.reset_index()
    df_finance = df_finance.set_index("Date")

    ######calculate the indeicators
    window_rsi = 14
    window_vol = 21

    def compute_rsi(series, window):
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain, index=series.index).rolling(window=window).mean()
        roll_down = pd.Series(loss, index=series.index).rolling(window=window).mean()

        RS = np.where(roll_down == 0, np.inf, roll_up / roll_down)
        RSI = 100.0 - (100.0 / (1.0 + RS))

        return RSI

    for ticker in tickers:
        close_col = f"{ticker}_Close"
        volume_col = f"{ticker}_Volume"

        if close_col not in df_finance.columns or volume_col not in df_finance.columns:
            continue

        df_finance[f"{ticker}_RSI_14"] = compute_rsi(
            df_finance[close_col], window=window_rsi
        )

        log_return = np.log(1 + df_finance[close_col].pct_change())
        df_finance[f"{ticker}_Log_Return"] = log_return
        df_finance[f"{ticker}_Volatility_21"] = log_return.rolling(
            window=window_vol
        ).std()

        df_finance[f"{ticker}_Volume_Z"] = (
            df_finance[volume_col] - df_finance[volume_col].rolling(window_vol).mean()
        ) / df_finance[volume_col].rolling(window_vol).std()

    df_finance = df_finance.iloc[1:]

    #### tda features and stuff
    window_wasserstein = 30
    eps = 0.5
    maxdim = 2
    rips = Rips(maxdim=maxdim, verbose=False)

    def betti_numbers_at_scale(diagrams, eps):
        bettis = []
        for dgm in diagrams:
            alive = np.sum((dgm[:, 0] <= eps) & (dgm[:, 1] > eps))
            bettis.append(int(alive))
        return bettis

    log_returns = np.array(
        [df_finance[f"{ticker}_Log_Return"].values for ticker in tickers]
    )

    persistences = []
    betti_list = []
    wasserstein_list = []

    for start in range(0, len(log_returns.T) - window_wasserstein + 1):
        window_data = log_returns[start : start + window_wasserstein].T
        diagrams = rips.fit_transform(window_data)
        persistences.append(diagrams)

        bettis = betti_numbers_at_scale(diagrams, eps=eps)
        while len(bettis) < 3:
            bettis.append(0)
        betti_list.append(bettis)

    for i in range(1, len(persistences)):
        dgm_prev = persistences[i - 1][1]
        dgm_curr = persistences[i][1]
        W = persim.wasserstein(dgm_prev, dgm_curr)
        wasserstein_list.append(W)

    betti_df = pd.DataFrame(
        betti_list,
        columns=["Betti0", "Betti1", "Betti2"],
        index=df_finance.index[window_wasserstein - 1 :],
    )
    wasserstein_df = pd.DataFrame(
        {"Wasserstein": wasserstein_list}, index=df_finance.index[window_wasserstein:]
    )

    df_finance["Wasserstein"] = wasserstein_df.reindex(df_finance.index)["Wasserstein"]

    """
    for col in ["Betti0", "Betti1", "Betti2"]:
        df_finance[col] = betti_df.reindex(df_finance.index)[col]
    """

    df_finance = df_finance[sorted(df_finance.columns)]
    df_finance = df_finance.iloc[window_wasserstein:]

    df_finance = df_finance[sorted(df_finance.columns)]
    df_finance.columns = pd.MultiIndex.from_tuples(
        [
            (col.split("_")[0], "_".join(col.split("_")[1:]))
            for col in df_finance.columns
        ]
    )

    train_features = [
        "Close",
        "High",
        "Low",
        "Open",
        "RSI_14",
        "Volatility_21",
        "Volume",
        "Volume_Z",
        "Sentiment",
    ]

    with open("symbol_to_security.pkl", "rb") as f:
        symbol_to_security = pickle.load(f)

    df_articles = pd.DataFrame().reindex(df_finance.index)

    ##### news articles fetching

    import requests
    import feedparser

    import warnings

    warnings.filterwarnings("ignore")

    articles_cache_name = f"articles_data_{start_date}_to_{end_date}.csv"
    if os.path.exists(articles_cache_name):
        df_articles = pd.read_csv(articles_cache_name, index_col=0, parse_dates=True)
        df_articles = df_articles.reindex(df_finance.index)
        print(f"Loaded cached articles from {articles_cache_name}")
    else:
        count = 0

        for ticker in tickers:
            if ticker in df_articles.columns:
                print(f"Skipping {ticker}, already in df_articles")
                continue

            keyword = symbol_to_security[ticker]
            print(keyword)
            if len(ticker) <= 3:
                keyword += " stock"

            query = f"{keyword} after:{start_date} before:{end_date}"
            feed_url = f"https://news.google.com/rss/search?q={query}"

            try:
                response = requests.get(feed_url, headers={"User-Agent": "Mozilla/5.0"})
                feed = feedparser.parse(response.text)

                titles = []
                dates = []

                for entry in feed.entries:
                    if "title" in entry and "published" in entry:
                        titles.append(entry["title"])
                        try:
                            article_date = pd.to_datetime(
                                entry["published"], errors="coerce"
                            ).normalize()
                        except Exception:
                            continue
                        dates.append(article_date)

                if titles and dates:
                    articles_df = pd.DataFrame({"title": titles, "date": dates})

                    articles_df.set_index("date", inplace=True)
                    articles_df.rename(columns={"title": ticker}, inplace=True)
                    articles_df.index.name = "Date"

                    articles_df = articles_df[
                        ~articles_df.index.duplicated(keep="first")
                    ]
                    valid_dates = articles_df.index.intersection(df_articles.index)

                    if len(valid_dates) > 0:
                        df_articles.loc[valid_dates, ticker] = articles_df.loc[
                            valid_dates, ticker
                        ]
                    else:
                        df_articles[ticker] = np.nan

                    print(f"{keyword} done - found {len(titles)} articles")
                else:
                    df_articles[ticker] = np.nan
                    print(f"{keyword} done - no valid articles or no results")

            except Exception as e:
                print(f"Error for {ticker}: {e}")
                df_articles[ticker] = np.nan

            count += 1

        df_articles.to_csv(articles_cache_name, index=True)
        print(f"Saved articles to {articles_cache_name}")

    ##### sentiment analysis

    SENTIMENT_CSV = f"finbert_sentiment_df_{start_date}_to_{end_date}.csv"

    if os.path.exists(SENTIMENT_CSV):
        print(f"Loading cached sentiment results from {SENTIMENT_CSV}")
        finbert_sentiment_df = pd.read_csv(SENTIMENT_CSV, index_col=0)
        finbert_sentiment_df.index = pd.to_datetime(finbert_sentiment_df.index)
        finbert_sentiment_df.columns = df_articles.columns
    else:

        finbert = BertForSequenceClassification.from_pretrained(
            "yiyanghkust/finbert-tone", num_labels=3
        )
        tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")

        nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

        def print_nlp(text):
            if text is None or pd.isna(text):
                return "Neutral"
            res = nlp(text)
            return res

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using device: CUDA (GPU: {torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using device: Apple Silicon (mps)")
        else:
            device = torch.device("cpu")
            print("Using device: CPU")

        model_name = "ProsusAI/finbert"  # adjust if needed
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)

        bert_model = bert_model.to(device)

        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=bert_model,
            tokenizer=tokenizer,
            device=-1,  # device=-1 for CPU and MPS with PyTorch >1.13
            batch_size=8,  # Keep batch small for compatibility
            return_all_scores=True,
        )

        def batch_process_texts(texts, batch_size=8):
            """Batch process texts; works for M1/M2 Mac (MPS) or CPU"""
            results = []

            valid_texts = [
                str(text) if text and str(text).strip() else "" for text in texts
            ]

            for i in tqdm(
                range(0, len(valid_texts), batch_size), desc="Processing batches"
            ):
                batch = valid_texts[i : i + batch_size]

                # Process batch
                batch_results = sentiment_pipeline(batch)

                batch_labels = []
                for result in batch_results:
                    if isinstance(result, list) and len(result) > 0:
                        # Get the label with highest score
                        best_result = max(result, key=lambda x: x["score"])
                        batch_labels.append(best_result["label"])
                    else:
                        batch_labels.append("NEUTRAL")  # fallback

                results.extend(batch_labels)

                # Clear cache periodically (MPS no empty_cache, just collect)
                if i % (batch_size * 10) == 0:
                    gc.collect()

            return results

        # Flatten dataframe to array
        text_array = df_articles.values.flatten()
        print(f"Processing {len(text_array)} texts...")

        results = batch_process_texts(text_array, batch_size=8)

        # Reshape to dataframe as before
        finbert_sentiment_df = pd.DataFrame(
            np.array(results).reshape(df_articles.shape),
            index=df_articles.index,
            columns=df_articles.columns,
        )
        finbert_sentiment_df.to_csv(SENTIMENT_CSV)

    gc.collect()  # extra cleanup

    # finbert_sentiment_num_df = pd.read_csv("finbert_sentiment_df.csv")

    finbert_sentiment_num_df = finbert_sentiment_df.replace(
        {"neutral": 0, "positive": 1, "negative": -1}
    )
    finbert_sentiment_num_df.index = pd.to_datetime(finbert_sentiment_num_df.index)
    for ticker in tickers:
        df_finance[(ticker, "Sentiment")] = finbert_sentiment_num_df[ticker]

    scaler = joblib.load("scaler.pkl")

    window_prediction = 30

    X_all = []
    ticker_names = []
    print(df_finance.head())
    # Process each ticker separately for window extraction,
    # using the same scaler that was fitted during training
    for ticker in tickers:
        try:
            df_ticker_finance = df_finance[ticker]
            df_predict = df_ticker_finance[train_features].copy()
            df_predict = df_predict.merge(
                df_finance["Wasserstein"].loc[df_predict.index],
                left_index=True,
                right_index=True,
            )
            df_predict[train_features] = scaler.transform(df_predict[train_features])
        except Exception as e:
            print(f"Error {ticker}: {e}")
            continue

        # Use all 30 days, if available
        if len(df_predict) >= window_prediction:
            X_window = df_predict[train_features].iloc[-window_prediction:].values
            X_all.append(X_window)
            ticker_names.append(ticker)
        else:
            print(
                f"Not enough data for ticker {ticker} (has {len(df_predict)} rows) to form the required window."
            )

    X_all = np.array(X_all)
    print(f"Prepared data for {len(ticker_names)} tickers")
    print(f"Input shape: {X_all.shape}")

    model = load_model("lstm.keras", compile=False)
    predictions = model.predict(X_all)

    results_df = pd.DataFrame(
        {"Ticker": ticker_names, "Prediction": predictions.flatten()}
    )
    print(results_df.head())
    results_df.to_csv(f"results_predictions_{start_date}_to_{end_date}.csv", index=False)
    
    #### alpaca api
    import requests
    from dotenv import load_dotenv
    load_dotenv()

    ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
    
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
    }
    
    def alpaca_trading(results_df, df_finance):

        buy_threshold = 0.004  # log returns
        buy_amount_usd = 1000


        positions_url = f"{ALPACA_BASE_URL}/v2/positions"
        try:
            response = requests.get(positions_url, headers=headers)
            response.raise_for_status()
            positions_data = response.json()
            current_positions = {position['symbol']: float(position['qty']) for position in positions_data}
        except Exception as e:
            print(f"error fetching positions: {e}")
            current_positions = {}

        for idx, row in results_df.iterrows():
            ticker = row["Ticker"]
            pred = row["Prediction"]

            if pred >= buy_threshold:
                # buy 1k worth
                if ticker not in current_positions or current_positions[ticker] == 0:
                    try:
                        last_price = df_finance[(ticker, "Close")].iloc[-1]
                        qty = int(buy_amount_usd // last_price)
                        if qty > 0:
                            order = {
                                "symbol": ticker,
                                "qty": qty,
                                "side": "buy",
                                "type": "market",
                                "time_in_force": "day"
                            }
                            order_url = f"{ALPACA_BASE_URL}/v2/orders"
                            order_response = requests.post(order_url, json=order, headers=headers)
                            if order_response.status_code == 200 or order_response.status_code == 201:
                                print(f"bought {qty} of {ticker} at ${last_price:.2f}")
                            else:
                                print(f"error submitting buy order for {ticker}: {order_response.text}")
                    except Exception as e:
                        print(f"error buying {ticker}: {e}")
                else:
                    print(f"holding {ticker}: already in portfolio ({current_positions[ticker]} shares)")
            else:
                if ticker in current_positions and current_positions[ticker] > 0:
                    try:
                        qty = int(current_positions[ticker])
                        order = {
                            "symbol": ticker,
                            "qty": qty,
                            "side": "sell",
                            "type": "market",
                            "time_in_force": "day"}
                        order_url = f"{ALPACA_BASE_URL}/v2/orders"
                        order_response = requests.post(order_url, json=order, headers=headers)
                        if order_response.status_code == 200 or order_response.status_code == 201:
                            print(f"sold {qty} of {ticker}")
                        else:
                            print(f"error submitting sell order for {ticker}: {order_response.text}")
                    except Exception as e:
                        print(f"error selling {ticker}: {e}")

    import pytz
    def has_trades_today():
        eastern = pytz.timezone("US/Eastern")
        today_eastern = datetime.now(eastern).date().isoformat()
        activities_url = f"{ALPACA_BASE_URL}/v2/account/activities"
        params = {
            "activity_types": "FILL"
        }
        response = requests.get(activities_url, headers=headers, params=params)
        if response.status_code == 200:
            activities = response.json()
            for act in activities:
                trade_date = act.get("transaction_time", act.get("settlement_date", ""))
                trade_date = trade_date[:10]
                if trade_date == today_eastern:
                    return True
        else:
            print(f"rrror fetching account activities: {response.text}")
        return False

    if not has_trades_today():
        alpaca_trading(results_df, df_finance)
    else:
        print("Trades have already been made today, don't wanna repeat auto trades.")
    
    return results_df.to_dict(orient="records")
