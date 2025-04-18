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


def preprocess_project_data(
        filepath: str,
        sma_short_window: int = 12,
        sma_long_window:  int = 25,
        volatility_window: int = 5,
        rsi_period: int = 12,
        ema_span: int = 7,
        boll_window: int = 7,
        boll_std_factor: int = 2,
        macd_short: int = 12,
        macd_long: int = 26,
        macd_signal: int = 9,
        split_date: str = "2024-03-01"
    ):
    """
    Load cleaned CSV ➜ build per ticker technical indicators ➜
    return (train_df, test_df) where test_df starts on `split_date`.

    For every ticker (detected via Close_<ticker>):
        • keep raw High_, Low_, Close_  (and Volume_ / Dividends_ already in file)
        • add Adj_Return_, SMA_short/long_, Volatility_, RSI_, EMA_,
          BOLL_upper/lower_, MACD_line/signal_
    """
    # 1) read & sort ------------------------------------------------
    df = pd.read_csv(filepath)
    df["Price_Ticker"] = pd.to_datetime(df["Price_Ticker"])
    df = df.sort_values("Price_Ticker").set_index("Price_Ticker")
    print("Loaded", df.shape, "rows.")

    # 2) discover tickers ------------------------------------------
    tickers = sorted(c.replace("Close_", "") for c in df.columns if c.startswith("Close_"))
    print("Tickers:", tickers)

    # helper for RSI
    def _rsi(series, p=14):
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(p, 1).mean()
        loss  = -delta.clip(upper=0).rolling(p, 1).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    per_ticker_frames = []

    # 3) loop over tickers -----------------------------------------
    for tk in tickers:
        close = df[f"Close_{tk}"]
        div   = df.get(f"Dividends_{tk}", 0).fillna(0)

        # a) adj‑log‑return
        adj_ret = np.log((close + div) / close.shift(1))
        df_adj  = adj_ret.rename(f"Adj_Return_{tk}")

        # b) SMA
        sma_s  = close.rolling(sma_short_window, 1).mean().rename(f"SMA_short_{tk}")
        sma_l  = close.rolling(sma_long_window, 1).mean().rename(f"SMA_long_{tk}")

        # c) volatility of adj ret
        vol5   = adj_ret.rolling(volatility_window, 1).std().rename(f"Volatility_{tk}")

        # d) RSI
        rsi12  = _rsi(close, rsi_period).rename(f"RSI_{tk}")

        # e) EMA
        ema7   = close.ewm(span=ema_span, adjust=False).mean().rename(f"EMA_{tk}")

        # f) Bollinger
        mid    = close.rolling(boll_window, 1).mean()
        std    = close.rolling(boll_window, 1).std()
        boll_u = (mid + boll_std_factor*std).rename(f"BOLL_upper_{tk}")
        boll_l = (mid - boll_std_factor*std).rename(f"BOLL_lower_{tk}")

        # g) MACD
        exp1   = close.ewm(span=macd_short, adjust=False).mean()
        exp2   = close.ewm(span=macd_long,  adjust=False).mean()
        macd   = (exp1-exp2).rename(f"MACD_line_{tk}")
        macd_s = macd.ewm(span=macd_signal, adjust=False).mean().rename(f"MACD_signal_{tk}")

        # gather for this ticker  ----------------------------------
        keep_cols = df[[f"High_{tk}", f"Low_{tk}", f"Close_{tk}", f"Volume_{tk}"]].copy() # high/low/close
        tk_frame  = pd.concat(
            [keep_cols, df_adj, sma_s, sma_l, vol5, rsi12,
             ema7, boll_u, boll_l, macd, macd_s],
            axis=1
        )
        per_ticker_frames.append(tk_frame)
        print(f"  added indicators for {tk}")

    # 4) merge everything ------------------------------------------
    full = pd.concat(per_ticker_frames, axis=1)

    # 5) train / test split ----------------------------------------
    train_df = full.loc[full.index < split_date].copy()
    test_df  = full.loc[full.index >= split_date].copy()
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    return train_df, test_df


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
        print(f"{row['ticker']}: Volatility = {row['volatility']:.5f}, Vol_Rank = {row['vol_rank']:.1f}")

    print("\nRanking by Average Correlation (lower is better):")
    for _, row in metrics_df.sort_values("corr_rank").iterrows():
        print(f"{row['ticker']}: Avg Corr = {row['avg_corr']:.5f}, Corr_Rank = {row['corr_rank']:.1f}")

    print("\nRanking by Average Return (higher is better):")
    for _, row in metrics_df.sort_values("ret_rank").iterrows():
        print(f"{row['ticker']}: Avg Return = {row['avg_return']:.5f}, Ret_Rank = {row['ret_rank']:.1f}")

    print("\nComposite Ranking (lower is better):")
    comp_sorted = metrics_df.sort_values("composite_rank")
    for _, row in comp_sorted.iterrows():
        print(f"{row['ticker']}: Composite Rank = {row['composite_rank']:.1f}")

    top10 = comp_sorted.head(10)["ticker"].tolist()
    return top10