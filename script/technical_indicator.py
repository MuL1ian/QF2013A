import pandas as pd
import numpy as np
from script.BaseStrategy import BaseStrategy

class SMAStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        sma_short_col = f"SMA_short_{self.ticker}"
        sma_long_col = f"SMA_long_{self.ticker}"
        if sma_short_col not in data.columns or sma_long_col not in data.columns:
            raise ValueError(f"SMA columns missing for {self.ticker}.")
        sma_short = data[sma_short_col]
        sma_long = data[sma_long_col]
        signals = np.where(sma_short > sma_long, 1, -1)
        return pd.Series(signals, index=data.index)

    def simulate(self, data: pd.DataFrame) -> pd.DataFrame:
        close_col = f"Close_{self.ticker}"
        div_col = f"Dividends_{self.ticker}"
        prices = data[close_col]
        dividends = data.get(div_col, pd.Series(0, index=data.index)).fillna(0)
        log_returns = np.log((prices + dividends) / prices.shift(1))
        simple_ret = np.exp(log_returns) - 1
        signals = self.generate_signals(data)
        positions = signals

        equity = pd.Series(index=data.index, dtype=float)
        equity.iloc[0] = self.initial_investment
        for t in range(1, len(prices)):
            equity.iloc[t] = equity.iloc[t-1] * (1 + positions.iloc[t-1] * simple_ret.iloc[t])

        results = pd.DataFrame({
            "Signal": signals,
            "SimpleReturn": simple_ret,
            "Position": positions,
            "Equity": equity
        }, index=data.index)
        self.results = results
        return results
    
class RSIStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        rsi_col = f"RSI_{self.ticker}"
        if rsi_col not in data.columns:
            raise ValueError(f"RSI column for {self.ticker} missing.")
        rsi_values = data[rsi_col]
        signals = []
        prev_signal = 1  # default long signal for the first day
        for rsi in rsi_values:
            if rsi < 30:
                signal = 1
            elif rsi > 70:
                signal = -1
            else:
                signal = prev_signal
            signals.append(signal)
            prev_signal = signal
        return pd.Series(signals, index=data.index)

    def simulate(self, data: pd.DataFrame) -> pd.DataFrame:
        close_col = f"Close_{self.ticker}"
        div_col = f"Dividends_{self.ticker}"
        prices = data[close_col]
        dividends = data.get(div_col, pd.Series(0, index=data.index)).fillna(0)
        log_returns = np.log((prices + dividends) / prices.shift(1))
        simple_ret = np.exp(log_returns) - 1
        signals = self.generate_signals(data)
        positions = signals

        equity = pd.Series(index=data.index, dtype=float)
        equity.iloc[0] = self.initial_investment
        for t in range(1, len(prices)):
            equity.iloc[t] = equity.iloc[t-1] * (1 + positions.iloc[t-1] * simple_ret.iloc[t])

        results = pd.DataFrame({
            "Signal": signals,
            "SimpleReturn": simple_ret,
            "Position": positions,
            "Equity": equity
        }, index=data.index)
        self.results = results
        return results
    

class EMAStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        ema_col = f"EMA_{self.ticker}"
        close_col = f"Close_{self.ticker}"
        if ema_col not in data.columns:
            raise ValueError(f"EMA column for {self.ticker} missing.")
        prices = data[close_col]
        ema = data[ema_col]
        signals = np.where(prices > ema, 1, -1)
        return pd.Series(signals, index=data.index)

    def simulate(self, data: pd.DataFrame) -> pd.DataFrame:
        close_col = f"Close_{self.ticker}"
        div_col = f"Dividends_{self.ticker}"
        prices = data[close_col]
        dividends = data.get(div_col, pd.Series(0, index=data.index)).fillna(0)
        log_returns = np.log((prices + dividends) / prices.shift(1))
        simple_ret = np.exp(log_returns) - 1
        signals = self.generate_signals(data)
        positions = signals

        equity = pd.Series(index=data.index, dtype=float)
        equity.iloc[0] = self.initial_investment
        for t in range(1, len(prices)):
            equity.iloc[t] = equity.iloc[t-1] * (1 + positions.iloc[t-1] * simple_ret.iloc[t])

        results = pd.DataFrame({
            "Signal": signals,
            "SimpleReturn": simple_ret,
            "Position": positions,
            "Equity": equity
        }, index=data.index)
        self.results = results
        return results

class BollStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close_col = f"Close_{self.ticker}"
        upper_col = f"BOLL_upper_{self.ticker}"
        lower_col = f"BOLL_lower_{self.ticker}"
        if close_col not in data.columns or upper_col not in data.columns or lower_col not in data.columns:
            raise ValueError(f"Bollinger Bands columns missing for {self.ticker}.")
        prices = data[close_col]
        upper = data[upper_col]
        lower = data[lower_col]
        signals = []
        init_signal = 1  # default long on the first day
        prev_signal = init_signal
        for price, up, low in zip(prices, upper, lower):
            if price < low:
                sig = 1
            elif price > up:
                sig = -1
            elif prev_signal == -1 and price < up:
                sig = 1
            else:
                sig = prev_signal
            signals.append(sig)
            prev_signal = sig
        return pd.Series(signals, index=data.index)

    def simulate(self, data: pd.DataFrame) -> pd.DataFrame:
        close_col = f"Close_{self.ticker}"
        div_col = f"Dividends_{self.ticker}"
        prices = data[close_col]
        dividends = data.get(div_col, pd.Series(0, index=data.index)).fillna(0)
        log_returns = np.log((prices + dividends) / prices.shift(1))
        simple_ret = np.exp(log_returns) - 1
        signals = self.generate_signals(data)
        positions = signals

        equity = pd.Series(index=data.index, dtype=float)
        equity.iloc[0] = self.initial_investment
        for t in range(1, len(prices)):
            equity.iloc[t] = equity.iloc[t-1] * (1 + positions.iloc[t-1] * simple_ret.iloc[t])

        results = pd.DataFrame({
            "Signal": signals,
            "SimpleReturn": simple_ret,
            "Position": positions,
            "Equity": equity
        }, index=data.index)
        self.results = results
        return results
    
class MACDStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close_col = f"Close_{self.ticker}"
        if close_col not in data.columns:
            raise ValueError(f"Close column for {self.ticker} missing.")
        prices = data[close_col]
        # Use default MACD parameters or ones passed via self.params.
        macd_short = self.params.get("macd_short", 12)
        macd_long = self.params.get("macd_long", 26)
        macd_signal = self.params.get("macd_signal", 9)
        ema_short = prices.ewm(span=macd_short, adjust=False).mean()
        ema_long = prices.ewm(span=macd_long, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()

        signals = []
        prev_signal = 1  # default long signal
        for m, s in zip(macd_line, signal_line):
            if m > s:
                sig = 1
            elif m < s:
                sig = -1
            else:
                sig = prev_signal
            signals.append(sig)
            prev_signal = sig
        return pd.Series(signals, index=data.index)

    def simulate(self, data: pd.DataFrame) -> pd.DataFrame:
        close_col = f"Close_{self.ticker}"
        div_col = f"Dividends_{self.ticker}"
        if close_col not in data.columns:
            raise ValueError(f"Close column for {self.ticker} missing.")
        prices = data[close_col]
        dividends = data.get(div_col, pd.Series(0, index=data.index)).fillna(0)
        log_returns = np.log((prices + dividends) / prices.shift(1))
        simple_ret = np.exp(log_returns) - 1
        signals = self.generate_signals(data)
        positions = signals

        equity = pd.Series(index=data.index, dtype=float)
        equity.iloc[0] = self.initial_investment
        for t in range(1, len(prices)):
            equity.iloc[t] = equity.iloc[t-1] * (1 + positions.iloc[t-1] * simple_ret.iloc[t])

        results = pd.DataFrame({
            "Signal": signals,
            "SimpleReturn": simple_ret,
            "Position": positions,
            "Equity": equity
        }, index=data.index)
        self.results = results
        return results