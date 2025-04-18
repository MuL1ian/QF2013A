import numpy as np
import pandas as pd

def calculate_log_returns(data: pd.DataFrame, stock_ticker: str = 'AAPL') -> pd.Series:
    close_column = f'Close_{stock_ticker}'
    log_returns = np.log(data[close_column] / data[close_column].shift(1))
    return pd.Series(log_returns, index=data.index)


def calculate_simple_returns(data: pd.DataFrame, stock_ticker: str = 'AAPL') -> float:
    close_column = f'Close_{stock_ticker}'
    ST = data[close_column].iloc[-1]
    S0 = data[close_column].iloc[0]
    return (ST - S0) / S0


def preprocess_project_data(filepath,
                            sma_short_window=12,
                            sma_long_window=25,
                            volatility_window=5,
                            rsi_period=12,
                            ema_span=7,
                            boll_window=20,
                            boll_std_factor=2,
                            macd_short=12,
                            macd_long=26,
                            macd_signal=9,
                            split_date="2024-03-01"):
    """
    Loading the 'Trading_Project_Data_Cleaned.csv' which we cleaned in the data cleaning section. This function 
    computes the dividend-adjusted log returns and several technical indicators for each stock,
     and the data is split into training and testing sets.

    For each ticker (extracted from columns starting with "Close_"):
      - Dividend-adjusted log return is computed as:
              Adj_Return = ln((Close + Dividend) / Close.shift(1))
      - SMA_short and SMA_long (with periods 12 and 25, respectively) are computed.
      - Rolling volatility is computed on the adjusted return using a 5-day window.
      - RSI is computed over a 12-day period.
      - EMA of the closing price is computed with a span of 7.
      - Bollinger Bands (upper and lower) are computed using a 20-day window and a multiplier of 2.
      - MACD indicators computed using standard parameters:
            EMA12 (span=12), EMA26 (span=26), then MACD = EMA12 - EMA26, and
            MACD_Signal = EMA(MACD, span=9).


    Returns:
        tuple: (train_df, test_df) with all the computed technical indicators and the original data.
    """
    # Load and sort the base data.
    df = pd.read_csv(filepath)
    df["Price_Ticker"] = pd.to_datetime(df["Price_Ticker"])
    df = df.sort_values("Price_Ticker")
    df.set_index("Price_Ticker", inplace=True)
    print("Data loaded and sorted by date.")

    # Identify tickers from columns beginning with "Close_"
    tickers = sorted(set(col.replace("Close_", "") for col in df.columns if col.startswith("Close_")))
    # print("Tickers found:", tickers)

    ticker_dfs = []

    def compute_RSI(series, period=7):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(100)

    for ticker in tickers:
        close_col = f"Close_{ticker}"
        if close_col not in df.columns:
            print(f"{close_col} not found; skipping {ticker}.")
            continue

        div_col = f"Dividends_{ticker}"
        dividends = df[div_col] if div_col in df.columns else pd.Series(0, index=df.index)
        dividends = dividends.fillna(0)

        # Compute dividend-adjusted log returns.
        adj_return = np.log((df[close_col] + dividends) / df[close_col].shift(1))
        df_adj = adj_return.to_frame(name=f"Adj_Return_{ticker}")

        # Compute SMA indices.
        sma_short = df[close_col].rolling(window=sma_short_window, min_periods=1).mean()
        sma_long = df[close_col].rolling(window=sma_long_window, min_periods=1).mean()
        df_sma = pd.concat([sma_short.rename(f"SMA_short_{ticker}"),
                            sma_long.rename(f"SMA_long_{ticker}")], axis=1)

        # Compute rolling volatility on dividend-adjusted return.
        volatility = df_adj[f"Adj_Return_{ticker}"].rolling(window=volatility_window, min_periods=1).std()
        df_vol = volatility.to_frame(name=f"Volatility_{ticker}")

        # Compute RSI.
        rsi = compute_RSI(df[close_col], period=rsi_period)
        df_rsi = rsi.to_frame(name=f"RSI_{ticker}")

        # Compute EMA.
        ema = df[close_col].ewm(span=ema_span, adjust=False).mean()
        df_ema = ema.to_frame(name=f"EMA_{ticker}")

        # Compute Bollinger Bands.
        boll_mid = df[close_col].rolling(window=boll_window, min_periods=1).mean()
        boll_std = df[close_col].rolling(window=boll_window, min_periods=1).std()
        boll_upper = boll_mid + boll_std_factor * boll_std
        boll_lower = boll_mid - boll_std_factor * boll_std
        df_boll = pd.concat([boll_upper.rename(f"BOLL_upper_{ticker}"),
                             boll_lower.rename(f"BOLL_lower_{ticker}")], axis=1)

        # Compute MACD
        exp1 = df[close_col].ewm(span=macd_short, adjust=False).mean()
        exp2 = df[close_col].ewm(span=macd_long, adjust=False).mean()
        macd_line = exp1 - exp2
        signal = macd_line.ewm(span=macd_signal, adjust=False).mean()
        df_macd = pd.concat([macd_line.rename(f"MACD_line_{ticker}"),
                             signal.rename(f"MACD_signal_{ticker}")], axis=1)

        # Concatenate indicators for this ticker.
        ticker_df = pd.concat([df_adj, df_sma, df_vol, df_rsi, df_ema, df_boll, df_macd], axis=1)
        ticker_dfs.append(ticker_df)
        # print(f"Computed technical indicators for {ticker}.")

    # Join all ticker indicator DataFrames.
    technical_df = pd.concat(ticker_dfs, axis=1).copy()
    # print("Technical indicators concatenated.")

    # Join with the original data.
    newframe = pd.concat([df, technical_df], axis=1).copy()

    # Split into training and testing sets.
    train_df = newframe.loc[newframe.index < split_date].copy()
    test_df  = newframe.loc[newframe.index >= split_date].copy()
    full_df = newframe.copy()
    print(f"Data split into {train_df.shape[0]} training rows and {test_df.shape[0]} testing rows.")

    return full_df, train_df, test_df


def stock_selection(train_df):
    """
    Selects stocks based on three criteria computed from the preprocessed data:
      1. Volatility: Standard deviation of the dividend-adjusted log returns.
      2. Lower average correlation: Each stock's average correlation with other stocks based on
         dividend-adjusted returns (lower is better).
      3. Average return: Mean of the dividend-adjusted log returns (higher is better).

    For each ticker (identified from columns "Adj_Return_{ticker}"):
      - Compute volatility and average return.
      - Build a DataFrame from all dividend-adjusted returns to compute the correlation matrix,
        and then compute each stock's average correlation with the others.

    Stocks are then ranked by:
      - Volatility (higher is better → descending rank).
      - Average return (higher is better → descending rank).
      - Average correlation (lower is better → ascending rank).

    The composite ranking is the average of these three individual ranks.
    The function prints the rankings for each metric and returns the top 10 tickers based on
    the composite rank.

    Return:
             the top 10 tickers based on the composite rank.
    """
    tickers = sorted(set(col.replace("Adj_Return_", "") for col in train_df.columns if col.startswith("Adj_Return_")))

    metrics = []
    adj_return_dict = {}

    for ticker in tickers:
        col = f"Adj_Return_{ticker}"
        if col in train_df.columns:
            series = train_df[col].dropna()
            if len(series) > 1:
                vol = series.std()
                avg_ret = series.mean()
                adj_return_dict[ticker] = series
                metrics.append({
                    "ticker": ticker,
                    "volatility": vol,
                    "avg_return": avg_ret
                })
            else:
                print(f"Not enough data for {ticker}; skipping.")
        else:
            print(f"{col} not found for {ticker}.")

    metrics_df = pd.DataFrame(metrics)

    # Build DataFrame of all dividend-adjusted returns for correlation calculation.
    adj_returns_df = pd.DataFrame({ticker: adj_return_dict[ticker] for ticker in adj_return_dict}).dropna()
    corr_matrix = adj_returns_df.corr()

    avg_corr = {}
    for ticker in corr_matrix.columns:
        avg_corr[ticker] = (corr_matrix[ticker].sum() - 1) / (len(corr_matrix) - 1)
    metrics_df["avg_corr"] = metrics_df["ticker"].map(avg_corr)

    # Ranking:
    metrics_df["vol_rank"] = metrics_df["volatility"].rank(ascending=False, method="min")
    metrics_df["ret_rank"] = metrics_df["avg_return"].rank(ascending=False, method="min")
    metrics_df["corr_rank"] = metrics_df["avg_corr"].rank(ascending=True, method="min")

    metrics_df["composite_rank"] = (metrics_df["vol_rank"] + metrics_df["ret_rank"] + metrics_df["corr_rank"]) / 3

    print("Ranking by Volatility (higher is better):")
    for _, row in metrics_df.sort_values("vol_rank").iterrows():
        pass
        # print(f"{row['ticker']}: Volatility = {row['volatility']:.5f}, Vol_Rank = {row['vol_rank']:.1f}")

    print("\nRanking by Average Correlation (lower is better):")
    for _, row in metrics_df.sort_values("corr_rank").iterrows():
        pass
        # print(f"{row['ticker']}: Avg Corr = {row['avg_corr']:.5f}, Corr_Rank = {row['corr_rank']:.1f}")

    print("\nRanking by Average Return (higher is better):")
    for _, row in metrics_df.sort_values("ret_rank").iterrows():
        pass
        # print(f"{row['ticker']}: Avg Return = {row['avg_return']:.5f}, Ret_Rank = {row['ret_rank']:.1f}")

    print("\nComposite Ranking (lower is better):")
    comp_sorted = metrics_df.sort_values("composite_rank")
    for _, row in comp_sorted.iterrows():
        pass
        # print(f"{row['ticker']}: Composite Rank = {row['composite_rank']:.1f}")

    top10 = comp_sorted.head(10)["ticker"].tolist()
    print("\nTop 10 Selected Tickers:")
    # print(top10)
    return top10