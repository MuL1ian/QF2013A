import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseStrategy():
    def __init__(self, ticker: str, initial_investment: float, **kwargs):
        """
        Base class for trading strategies.

        Parameters:
            ticker (str): Stock symbol (e.g., "AAPL").
            initial_investment (float): Capital allocated for the strategy.
            kwargs: Additional strategy-specific parameters.
        """
        self.ticker = ticker
        self.initial_investment = initial_investment
        self.params = kwargs
        self.name = self.__class__.__name__
        self.results = None

    #abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        Should return a pandas Series indexed by date, with +1 for long and -1 for short.
        """
        pass

    #abstractmethod
    def simulate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Run a simulation on historical data.

        Returns a DataFrame with columns: 'Signal', 'SimpleReturn', 'Position', 'Equity'.
        """
        pass

    def simulate_benchmark(self, data: pd.DataFrame) -> pd.Series:
        """
        Simulate a benchmark strategy (always long).

        Uses effective log returns computed as:
            ln((Close + Dividend) / Close.shift(1))
        and then updates the equity curve.
        """
        close_col = f"Close_{self.ticker}"
        div_col = f"Dividends_{self.ticker}"
        prices = data[close_col]
        dividends = data.get(div_col, pd.Series(0, index=data.index)).fillna(0)
        log_returns = np.log((prices + dividends) / prices.shift(1))
        simple_ret = np.exp(log_returns) - 1

        benchmark_equity = pd.Series(index=prices.index, dtype=float)
        benchmark_equity.iloc[0] = self.initial_investment
        for t in range(1, len(prices)):
            benchmark_equity.iloc[t] = benchmark_equity.iloc[t-1] * (1 + simple_ret.iloc[t])
        return benchmark_equity

    def plot_results(self, data: pd.DataFrame):
        """
        Plot the strategy's equity curve alongside the benchmark, as well as positions and signals.
        """
        if self.results is None:
            print("No simulation results available. Run simulate() first.")
            return

        dates = self.results.index
        strategy_equity = self.results["Equity"]
        signals = self.results["Signal"]
        benchmark_equity = self.simulate_benchmark(data)

        fig, ax = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

        # Equity Curve Plot
        ax[0].plot(dates, strategy_equity, label=f"{self.name} Equity")
        ax[0].plot(dates, benchmark_equity, label="Benchmark (Always Long)", linestyle="--")
        ax[0].set_title(f"Equity Curve for {self.ticker}")
        ax[0].set_ylabel("Equity ($)")
        ax[0].legend()
        ax[0].grid(True)

        # Position Plot
        ax[1].plot(dates, self.results["Position"], label="Position", color="tab:orange")
        ax[1].set_title("Daily Trading Position")
        ax[1].set_ylabel("Position (Fraction)")
        ax[1].legend()
        ax[1].grid(True)


        plt.tight_layout()
        plt.show()

 
class BenchmarkStrategy:
    def __init__(self, ticker: str, initial_investment: float):
        """
        Benchmark strategy: always long.
        """
        self.ticker = ticker
        self.initial_investment = initial_investment
        self.name = "BenchmarkStrategy"
        self.results = None

    def simulate(self, data: pd.DataFrame) -> pd.DataFrame:
        close_col = f"Close_{self.ticker}"
        div_col = f"Dividends_{self.ticker}"
        prices = data[close_col]
        dividends = data.get(div_col, pd.Series(0, index=data.index)).fillna(0)
        log_returns = np.log((prices + dividends) / prices.shift(1))
        simple_ret = np.exp(log_returns) - 1

        equity = pd.Series(index=data.index, dtype=float)
        equity.iloc[0] = self.initial_investment
        for t in range(1, len(prices)):
            equity.iloc[t] = equity.iloc[t-1] * (1 + simple_ret.iloc[t])
        signals = pd.Series(1, index=data.index)  # always long
        results = pd.DataFrame({
            "Signal": signals,
            "SimpleReturn": simple_ret,
            "Position": pd.Series(1, index=data.index),
            "Equity": equity
        }, index=data.index)
        self.results = results
        return results

    def plot_results(self, data: pd.DataFrame):
        if self.results is None:
            print("No simulation results available. Run simulate() first.")
            return
        dates = self.results.index
        equity = self.results["Equity"]
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(dates, equity, label="Benchmark Equity")
        ax.set_title(f"Benchmark Equity Curve for {self.ticker}")
        ax.set_ylabel("Equity ($)")
        ax.set_xlabel("Date")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()
