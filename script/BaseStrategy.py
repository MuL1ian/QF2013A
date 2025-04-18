import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BaseStrategy:
    """
    Parent skeleton for all strategies.
    • generate_signals() must be supplied by the subclass.
    • simulate() must be supplied by the subclass and must create
      a DataFrame with:  Signal · LogReturn · StrategyReturn · CumRet
    """

    def __init__(self, ticker: str, **kwargs):
        self.ticker   = ticker
        self.params   = kwargs          # strategy‑specific hyper‑parameters
        self.name     = self.__class__.__name__
        self.results  = None            # populated by simulate()

    # ----------- to be implemented in each concrete strategy ------------------
    def generate_signals(self, data):
        raise NotImplementedError("subclass must implement")

    def simulate(self, data):
        raise NotImplementedError("subclass must implement")

    # --------------------- always‑long benchmark ------------------------------
    def simulate_benchmark(self, data):
        close = data[f"Close_{self.ticker}"]
        div   = data.get(f"Dividends_{self.ticker}", close.mul(0))  # zeros if absent
        log_r = ((close + div) / close.shift(1)).pipe(
            lambda s: s.apply(lambda x: 0 if x <= 0 else __import__('math').log(x))
        ).fillna(0)
        cum   = log_r.cumsum()
        cum.name = "BenchmarkCumRet"
        return cum

    # --------------------------- Plot helper ----------------------------------
    def plot_results(self, data):
        if self.results is None:
            print("Run simulate() first.")
            return

        import numpy as np, matplotlib.pyplot as plt

        strat_pct = (np.exp(self.results["CumRet"]) - 1) * 100
        bench_pct = (np.exp(self.simulate_benchmark(data)) - 1) * 100

        fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        ax[0].plot(strat_pct.index, strat_pct, label=self.name)
        ax[0].plot(bench_pct.index, bench_pct, '--', label="Benchmark")
        ax[0].set_ylabel("Cum Return (%)"); ax[0].legend(); ax[0].grid(True)

        ax[1].plot(self.results.index, self.results["Signal"], color="tab:orange")
        ax[1].set_ylabel("Signal"); ax[1].grid(True)
        ax[1].set_title(self.name + " daily position")

        plt.tight_layout(); plt.show()

class BenchmarkStrategy(BaseStrategy):
    """
    Always‑long benchmark: Signal == 1 every day.
    """

    def generate_signals(self, data):
        import pandas as pd
        return pd.Series(1.0, index=data.index, name="Signal")

    def simulate(self, data):
        import pandas as pd, math

        signal = self.generate_signals(data)
        close  = data[f"Close_{self.ticker}"]
        div    = data.get(f"Dividends_{self.ticker}", close.mul(0))

        log_r  = ((close + div) / close.shift(1)).pipe(
            lambda s: s.apply(lambda x: 0 if x <= 0 else math.log(x))
        ).fillna(0)

        strat = signal.shift(1).fillna(0) * log_r

        self.results = pd.DataFrame({
            "Signal": signal,
            "LogReturn": log_r,
            "StrategyReturn": strat,
            "CumRet": strat.cumsum()
        })
        return self.results