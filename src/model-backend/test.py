from fastapi import FastAPI
from alpaca.trading.client import TradingClient
import os

app = FastAPI()

# Retrieve Alpaca API credentials from environment variables
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

# Create Alpaca trading client (set paper=True for paper trading)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

@app.get("/alpaca/account")
async def get_alpaca_account():
    """
    Fetch Alpaca account information
    """
    account = trading_client.get_account()
    return account.to_dict() if hasattr(account, "to_dict") else dict(account)
