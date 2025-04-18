import pandas as pd


def update_daily_portfolio_state(date, simulation_results, allocation_per_stock):
    """
    For the given date, this function extracts the per-stock net position (the "Position" column)
    and equity from each ticker’s simulation results (each is a DataFrame with index being dates).

    For each stock:
      - If Position > 0, the stock is in a long state; if Position < 0, in a short state.
      - Free (uninvested) capital = 1 - |Position|
      - The value associated with the stock is given by its simulated equity.

    Using these values (assuming each stock is allocated 'allocation_per_stock'),
    the function computes:
      • Long Value = (Position if positive) * allocation_per_stock
      • Short Value = (absolute Position if negative) * allocation_per_stock
      • Cash (free capital) = (1 - |Position|) * allocation_per_stock
      • Total Value = the simulated equity for that stock.

    Then, aggregates across stocks:
      - Total Long = sum over stocks with positive positions,
      - Total Short = sum over stocks with negative positions (expressed as a positive number),
      - Total Cash = sum of free capital across stocks,
      - Total Portfolio Value = sum of simulated equity for all stocks.

    Returns:
        dict: A dictionary with keys:
            'Date', 'Long', 'Short', 'Cash', 'Total'
          and also (optionally) a per-stock breakdown dictionary.
    """
    state = {"Date": date, "Long": 0, "Short": 0, "Cash": 0, "Total": 0}
    per_stock = {}

    for ticker, sim_df in simulation_results.items():
        # Check if the given date exists in the simulation result
        if date not in sim_df.index:
            continue
        pos = sim_df.loc[date, "Position"]
        equity = sim_df.loc[date, "Equity"]
        # For each stock, assume allocated capital is normalized to 1; then scale using allocation_per_stock.
        long_val = max(pos, 0) * allocation_per_stock
        short_val = -min(pos, 0) * allocation_per_stock  # note: pos < 0 gives positive short value
        cash = (1 - abs(pos)) * allocation_per_stock  # free or uninvested capital
        bench = sim_df.loc[date, "BenchmarkEquity"]
        per_stock[ticker] = {"Position": pos, "Equity": equity, "Long": long_val, "Short": short_val, "Cash": cash, "BenchmarkEquity": bench}

        # Aggregate values
        state["Long"] += long_val
        state["Short"] += short_val
        state["Cash"] += cash
        state["Total"] += equity  # total portfolio value for that stock is its simulated equity

    # Optionally, one might decide only to record days where there is a change in position.
    return {"daily_state": state, "per_stock": per_stock}

def simulate_portfolio_over_period(tickers, simulation_results, allocation_per_stock):
    """
    Given a list of tickers and simulation_results (a dictionary mapping each ticker to its simulation DataFrame),
    iterate over all dates in the union of simulation indices and compute the daily portfolio state using
    update_daily_portfolio_state.

    Returns:
         pd.DataFrame: A DataFrame with columns Date, Long, Short, Cash, Total.
    """
    all_dates = set()
    for ticker in tickers:
        df = simulation_results.get(ticker)
        if df is not None:
            all_dates = all_dates.union(set(df.index))
    all_dates = sorted(list(all_dates))

    daily_states = []
    for dt in all_dates:
        state_info = update_daily_portfolio_state(dt, simulation_results, allocation_per_stock)
        daily_states.append(state_info)

    # Build aggregated DataFrame for portfolio-level records.
    rows = []
    for entry in daily_states:
        row = entry["daily_state"].copy()
        # Also include each stock’s position with column name "ticker_Position".
        for ticker, stats in entry["per_stock"].items():
            row[f"{ticker}_Position"] = stats.get("Position")
            row[f"{ticker}_Equity"] = stats.get("Equity")
            row[f"{ticker}_Benchmark_Equity"] = stats.get("BenchmarkEquity")
        rows.append(row)

    trade_doc_df = pd.DataFrame(rows)
    trade_doc_df.sort_values("Date", inplace=True)
    trade_doc_df.reset_index(drop=True, inplace=True)
    return trade_doc_df


def generate_trade_documentation_table(trade_doc_df, only_active=False, activity_threshold=1e-4):
    """
    Processes the trade documentation DataFrame to include positions by stock as separate columns.

    Parameters:
        trade_doc_df (pd.DataFrame): DataFrame with aggregated columns (Date, Long, Short, Cash, Total)
                                     and additional columns for per-stock positions (e.g. "NVDA_Position", etc.).
        only_active (bool): If True, only keep rows where total activity (Long+Short) changes significantly.
        activity_threshold (float): Threshold to filter out minor changes.

    Returns:
        pd.DataFrame: The final trade documentation table.
    """
    if only_active:
        active_rows = [True]
        prev_invested = None
        for i, row in trade_doc_df.iterrows():
            current_invested = row["Long"] + row["Short"]
            if prev_invested is None or abs(current_invested - prev_invested) > activity_threshold:
                active_rows.append(True)
            else:
                active_rows.append(False)
            prev_invested = current_invested
        filtered_df = trade_doc_df[active_rows].copy()
        filtered_df.reset_index(drop=True, inplace=True)
        return filtered_df
    else:
        return trade_doc_df