"""
验证动态3σ止损：低波异象168h × {无止损 / 固定10% / 动态3σ}
========================================================
全周期+三段，确认动态止损是否真正改善风险调整后收益。
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.strategy.low_vol_parity import LowVolRiskParityStrategy
from rooster_trader.backtest.engine import Backtester

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "history_binance")


def load_data():
    data = {}
    coins = [l.strip() for l in open(os.path.join(DATA_DIR, "universe.txt")) if l.strip()]
    for c in coins:
        fp = os.path.join(DATA_DIR, f"{c}_1h.csv")
        if os.path.exists(fp):
            data[f"{c}/USD"] = pd.read_csv(fp, index_col=0, parse_dates=True)
    common = None
    for s, df in data.items():
        common = df.index if common is None else common.intersection(df.index)
    return {s: df.loc[common] for s, df in data.items()}


def run(symbols, data, stop, dyn):
    strat = LowVolRiskParityStrategy(symbols, lookback_hours=168, top_n_pct=0.5)
    bt = Backtester(strategy=strat, initial_capital=50000, fee_rate=0.001,
                     min_rebalance_bars=72, rebalance_threshold=0.01,
                     stop_loss_pct=stop, dynamic_stop=dyn)
    r = bt.run(data)
    return r, bt.stop_loss_count


def seg_metrics(symbols, data, a, b, stop, dyn):
    sub = {x: df.iloc[a:b] for x, df in data.items()}
    r, n = run(symbols, sub, stop, dyn)
    eq = r.equity_curve
    fnf = next((i for i in range(len(eq)) if abs(eq.iloc[i]-eq.iloc[0]) > 0.01), 0)
    real = eq.iloc[fnf:]
    ret = real.iloc[-1]/real.iloc[0]-1
    dd = (real/real.cummax()-1).min()
    vol = real.pct_change().std()*np.sqrt(24*365)
    sh = real.pct_change().mean()*24*365/vol if vol > 0 else 0
    return ret, sh, dd, n


def main():
    data = load_data()
    symbols = list(data.keys())
    total = len(data[symbols[0]])
    win = 720
    bounds = [(s, min(s+win+win, total)) for s in range(0, total-win-100, win)][:3]

    modes = [("无止损", 0.0, False), ("固定10%", 0.10, False),
             ("固定12%", 0.12, False), ("固定15%", 0.15, False),
             ("动态3σ", 0.10, True)]

    print("=" * 100)
    print("低波异象168h：止损方式对比")
    print("=" * 100)
    print("\n【全周期】")
    print("-" * 100)
    print(f"{'止损方式':<12}{'收益':>9}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'回撤':>9}{'止损次数':>8}")
    for name, st, dyn in modes:
        r, n = run(symbols, data, st, dyn)
        print(f"{name:<12}{r.total_return*100:>+8.2f}%{r.sharpe_ratio:>+8.2f}"
              f"{r.sortino_ratio:>+9.2f}{r.calmar_ratio:>+8.2f}{r.max_drawdown*100:>+8.2f}%{n:>8}")

    print("\n【三段】")
    print("-" * 100)
    print(f"{'止损方式':<12}{'段1':>14}{'段2':>14}{'段3':>14}{'最差回撤':>9}")
    for name, st, dyn in modes:
        res = [seg_metrics(symbols, data, a, b, st, dyn) for a, b in bounds]
        wdd = min(x[2] for x in res)
        print(f"{name:<12}" + "".join(f"{x[0]*100:>+7.1f}%/{x[1]:>4.1f}" for x in res)
              + f"{wdd*100:>+8.1f}%")
    print("=" * 100)


if __name__ == "__main__":
    main()
