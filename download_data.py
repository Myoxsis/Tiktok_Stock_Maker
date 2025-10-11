import os
import argparse
import yfinance as yf
import pandas as pd

def download_stock_data(ticker: str, period: str = "1y"):
    """
    Download historical stock data for a given ticker and save Date, Close, and Dividends to a CSV.
    
    Args:
        ticker (str): Stock ticker (e.g. 'AAPL' for Apple, 'BN.PA' for Danone in Paris).
        period (str): Time period for data (default 1y).
    """
    # Create "data" folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Download stock data including dividend information
    df = yf.download(ticker, period=period, actions=True)

    if df.empty:
        print(f"❌ No data found for ticker {ticker}")
        return
    
    # Reset index to turn the Date into a column
    df = df.reset_index()

    # Ensure a Dividends column is available (may be absent for some tickers/periods)
    if "Dividends" not in df.columns:
        df["Dividends"] = 0.0

    # Keep only the relevant columns
    df = df[["Date", "Close", "Dividends"]]

    # Ensure date format YYYY-MM-DD
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # Ensure numeric columns use dot decimal
    df["Close"] = df["Close"].astype(float)
    df["Dividends"] = df["Dividends"].fillna(0).astype(float)

    # Save to CSV with dot as decimal separator
    file_path = os.path.join("data", f"{ticker}.csv")
    df.to_csv(file_path, index=False, float_format="%.6f")

    print(f"✅ Data for {ticker} saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data and save as CSV.")
    parser.add_argument("ticker", type=str, help="Stock ticker (e.g. AAPL, BN.PA)")
    parser.add_argument("--period", type=str, default="1y", 
                        help="Data period (default: 1y). Options: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max")

    args = parser.parse_args()

    download_stock_data(args.ticker, args.period)
