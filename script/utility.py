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