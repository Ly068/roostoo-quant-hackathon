"""
最终方案验证：时序趋势（只做多） + 周期内止损
============================================
在 Binance 真实数据上，对最终候选做全周期 + 三段独立窗口验证，
选择"震荡/下跌月下行可控、上涨月充分参与"的稳健配置。

候选：lookback(14/30天) × MA过滤(开/关) × 止损(无/10%)
评分重点：段1（震荡市）下行、三段正收益数、全周期收益、回撤
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.strategy.trend import TrendFollowingStrategy
from rooster_trader.backtest.engine import Backtester

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "history_binance")


def load_data():
    data = {}
    coins = [l.strip() for l in open(os.path.join(DATA_DIR, "universe.txt")) if l.strip()]
    for coin in coins:
        fp = os.path.join(DATA_DIR, f"{coin}_1h.csv")
        if os.path.exists(fp):
            data[f"{coin}/USD"] = pd.read_csv(fp, index_col=0, parse_dates=True)
    common = None
    for s, df in data.items():
        common = df.index if common is None else common.intersection(df.index)
    return {s: df.loc[common] for s, df in data.items()}


def run(strat, data, stop):
    bt = Backtester(strategy=strat, initial_capital=50000, fee_rate=0.001,
                     min_rebalance_bars=72, rebalance_threshold=0.01,
                     stop_loss_pct=stop)
    return bt.run(data), bt.stop_loss_count


def seg_metrics(strat, data, a, b, stop):
    sub = {x: df.iloc[a:b] for x, df in data.items()}
    r, nstop = run(strat, sub, stop)
    eq = r.equity_curve
    fnf = 0
    for i in range(len(eq)):
        if abs(eq.iloc[i] - eq.iloc[0]) > 0.01:
            fnf = i
            break
    real = eq.iloc[fnf:]
    ret = real.iloc[-1] / real.iloc[0] - 1
    dd = (real / real.cummax() - 1).min()
    vol = real.pct_change().std() * np.sqrt(24 * 365)
    sh = (real.pct_change().mean() * 24 * 365) / vol if vol > 0 else 0
    return ret, sh, dd, nstop


def main():
    data = load_data()
    symbols = list(data.keys())
    total = len(data[symbols[0]])
    win = 720
    bounds = [(s, min(s + win + win, total)) for s in range(0, total - win - 100, win)][:3]

    print("=" * 112)
    print("最终方案验证：时序趋势（只做多、无趋势空仓）+ 止损")
    print("=" * 112)

    cand = []
    for lb in [336, 720]:
        for maf in [True, False]:
            for stop in [0.0, 0.10]:
                t = TrendFollowingStrategy(symbols, lookback_hours=lb, ma_hours=lb,
                                           use_ma_filter=maf, max_total_exposure=1.0,
                                           rebalance_hours=72)
                name = f"{lb//24}天 {'MA' if maf else '无MA'} 止损{'10%' if stop else '无'}"
                cand.append((name, t, stop))

    print("\n【全周期】")
    print("-" * 112)
    print(f"{'配置':<22}{'收益':>9}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'回撤':>9}{'止损次数':>8}")
    full = []
    for name, t, stop in cand:
        r, nstop = run(t, data, stop)
        print(f"{name:<22}{r.total_return*100:>+8.2f}%{r.sharpe_ratio:>+8.2f}"
              f"{r.sortino_ratio:>+9.2f}{r.calmar_ratio:>+8.2f}"
              f"{r.max_drawdown*100:>+8.2f}%{nstop:>8}")
        full.append((name, t, stop, r))

    top = sorted([x for x in full if x[3].total_return > 0],
                 key=lambda z: z[3].sharpe_ratio, reverse=True)[:4]

    print("\n【三段独立窗口】（段1=7月震荡，段2=8月涨，段3=9月山寨暴涨）")
    print("-" * 112)
    print(f"{'配置':<22}{'段1收益/Sharpe':>18}{'段2收益/Sharpe':>18}{'段3收益/Sharpe':>18}{'正段':>6}{'最差回撤':>9}")
    for name, t, stop, _ in top:
        seg = [seg_metrics(t, data, a, b, stop) for a, b in bounds]
        pos = sum(1 for x in seg if x[0] > 0)
        worst_dd = min(x[2] for x in seg)
        print(f"{name:<22}" + "".join(f"{x[0]*100:>+9.1f}%/{x[1]:>+4.1f}" for x in seg)
              + f"{pos:>5}/3{worst_dd*100:>+8.1f}%")
    print("=" * 112)


if __name__ == "__main__":
    main()
