"""
策略对比：低波异象Risk Parity vs 时序动量
======================================
全周期 + 三段独立窗口，Binance真实数据，0.1%手续费，72h调仓。
用数据判断"低波异象"是否优于当前时序动量，而非凭导师一面之词。
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.strategy.trend import TrendFollowingStrategy
from rooster_trader.strategy.low_vol_parity import LowVolRiskParityStrategy
from rooster_trader.backtest.engine import Backtester

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "history_binance")


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


def run(strat, data, stop=0.0):
    bt = Backtester(strategy=strat, initial_capital=50000, fee_rate=0.001,
                     min_rebalance_bars=72, rebalance_threshold=0.01,
                     stop_loss_pct=stop)
    return bt.run(data)


def seg(strat, data, a, b, stop=0.0):
    sub = {x: df.iloc[a:b] for x, df in data.items()}
    r = run(strat, sub, stop)
    eq = r.equity_curve
    fnf = next((i for i in range(len(eq)) if abs(eq.iloc[i]-eq.iloc[0]) > 0.01), 0)
    real = eq.iloc[fnf:]
    ret = real.iloc[-1]/real.iloc[0]-1
    dd = (real/real.cummax()-1).min()
    vol = real.pct_change().std()*np.sqrt(24*365)
    sh = real.pct_change().mean()*24*365/vol if vol > 0 else 0
    return ret, sh, dd


def main():
    data = load_data()
    symbols = list(data.keys())
    total = len(data[symbols[0]])
    win = 720
    bounds = [(s, min(s+win+win, total)) for s in range(0, total-win-100, win)][:3]

    cands = [
        ("时序动量30天(当前)", TrendFollowingStrategy(
            symbols, lookback_hours=720, ma_hours=720,
            use_ma_filter=False, max_total_exposure=1.0, rebalance_hours=72)),
        ("低波异象 72h top50%", LowVolRiskParityStrategy(
            symbols, lookback_hours=72, top_n_pct=0.5)),
        ("低波异象 168h top50%", LowVolRiskParityStrategy(
            symbols, lookback_hours=168, top_n_pct=0.5)),
        ("低波异象 72h top30%", LowVolRiskParityStrategy(
            symbols, lookback_hours=72, top_n_pct=0.3)),
    ]

    print("=" * 108)
    print("策略对比：低波异象 vs 时序动量（全周期+三段）")
    print("=" * 108)
    print("\n【全周期120天】")
    print("-" * 108)
    print(f"{'策略':<22}{'收益':>9}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'回撤':>9}")
    for name, s in cands:
        r = run(s, data)
        print(f"{name:<22}{r.total_return*100:>+8.2f}%{r.sharpe_ratio:>+8.2f}"
              f"{r.sortino_ratio:>+9.2f}{r.calmar_ratio:>+8.2f}{r.max_drawdown*100:>+8.2f}%")

    print("\n【三段独立窗口】")
    print("-" * 108)
    print(f"{'策略':<22}{'段1收益/Sharpe':>18}{'段2收益/Sharpe':>18}{'段3收益/Sharpe':>18}{'正段':>6}{'最差回撤':>9}")
    for name, s in cands:
        res = [seg(s, data, a, b) for a, b in bounds]
        pos = sum(1 for x in res if x[0] > 0)
        wdd = min(x[2] for x in res)
        print(f"{name:<22}" + "".join(f"{x[0]*100:>+9.1f}%/{x[1]:>+4.1f}" for x in res)
              + f"{pos:>5}/3{wdd*100:>+8.1f}%")
    print("=" * 108)


if __name__ == "__main__":
    main()
