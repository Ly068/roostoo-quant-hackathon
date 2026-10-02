"""
夺冠优化网格回测
================
控制变量：总多头敞口=总空头敞口=50%（总敞口100%）保持不变，
逐一对比三个维度：
  1. 持仓分散度 top_n：2 / 3 / 4（单仓 = 0.5/top_n）
  2. 加权方式 sizing：equal（等权）/ inverse_vol（波动率倒数）
  3. 周期内止损 stop_loss：0（无）/ 0.08 / 0.10

全周期（120天数据，30天预热后约88个有效交易日）
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.strategy.momentum import MomentumStrategy
from rooster_trader.backtest.engine import Backtester

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LARGE_DIR = os.path.join(PROJECT_ROOT, "data", "history_binance")


def load_data():
    data = {}
    uf = os.path.join(LARGE_DIR, "universe.txt")
    coins = [l.strip() for l in open(uf) if l.strip()]
    for coin in coins:
        fp = os.path.join(LARGE_DIR, f"{coin}_1h.csv")
        if os.path.exists(fp):
            data[f"{coin}/USD"] = pd.read_csv(fp, index_col=0, parse_dates=True)
    common = None
    for s, df in data.items():
        common = df.index if common is None else common.intersection(df.index)
    return {s: df.loc[common] for s, df in data.items()}


def run_cfg(data, symbols, top_n, sizing, stop):
    # 控制总敞口：单仓 = 50% / top_n（满仓口径，与历史分段验证一致）
    ps = 0.5 / top_n
    mom = MomentumStrategy(
        symbols=symbols, lookback_hours=720, top_n=top_n,
        rebalance_hours=72, position_size=ps, sizing_method=sizing,
    )
    bt = Backtester(
        strategy=mom, initial_capital=50000, fee_rate=0.001,
        min_rebalance_bars=72, rebalance_threshold=0.01,
        stop_loss_pct=stop,
    )
    r = bt.run(data)
    return r, bt.stop_loss_count


def main():
    data = load_data()
    symbols = list(data.keys())
    total_bars = len(data[symbols[0]])
    print("=" * 108)
    print("夺冠优化网格（30天动量，72h调仓，0.1%手续费，总敞口100%固定）")
    print("=" * 108)
    print(f"币种 {len(symbols)}，总K线 {total_bars}（{total_bars/24:.0f}天），"
          f"区间 {data[symbols[0]].index[0]:%Y-%m-%d} ~ {data[symbols[0]].index[-1]:%Y-%m-%d}")

    top_ns = [2, 3, 4]
    sizings = ["equal", "inverse_vol"]
    stops = [0.0, 0.08, 0.10]

    rows = []
    print("\n" + "-" * 108)
    print(f"{'配置':<34}{'收益':>9}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'回撤':>9}{'交易':>5}{'止损':>5}")
    print("-" * 108)

    for top_n, sizing, stop in itertools.product(top_ns, sizings, stops):
        r, n_stop = run_cfg(data, symbols, top_n, sizing, stop)
        sl_label = "无" if stop == 0 else f"{stop*100:.0f}%"
        name = f"top{top_n} {sizing[:6]} 止损{sl_label}"
        print(f"{name:<34}{r.total_return*100:>+8.2f}%{r.sharpe_ratio:>+8.2f}"
              f"{r.sortino_ratio:>+9.2f}{r.calmar_ratio:>+8.2f}"
              f"{r.max_drawdown*100:>+8.2f}%{r.total_trades:>5}{n_stop:>5}")
        rows.append({
            "top_n": top_n, "sizing": sizing, "stop": stop,
            "return": r.total_return, "sharpe": r.sharpe_ratio,
            "sortino": r.sortino_ratio, "calmar": r.calmar_ratio,
            "dd": r.max_drawdown, "trades": r.total_trades, "stops": n_stop,
        })

    # 综合排序：Sharpe 为主，兼顾收益与回撤
    print("\n" + "=" * 108)
    print("按 Sharpe 排序 Top5：")
    for x in sorted(rows, key=lambda z: z["sharpe"], reverse=True)[:5]:
        sl = "无" if x["stop"] == 0 else f"{x['stop']*100:.0f}%"
        print(f"  top{x['top_n']} {x['sizing']:<11} 止损{sl:<3} | "
              f"收益{x['return']*100:+.2f}% Sharpe{x['sharpe']:.2f} "
              f"回撤{x['dd']*100:.2f}% Calmar{x['calmar']:.2f}")

    print("\n按 收益 排序 Top5：")
    for x in sorted(rows, key=lambda z: z["return"], reverse=True)[:5]:
        sl = "无" if x["stop"] == 0 else f"{x['stop']*100:.0f}%"
        print(f"  top{x['top_n']} {x['sizing']:<11} 止损{sl:<3} | "
              f"收益{x['return']*100:+.2f}% Sharpe{x['sharpe']:.2f} "
              f"回撤{x['dd']*100:.2f}% Calmar{x['calmar']:.2f}")
    print("=" * 108)


if __name__ == "__main__":
    main()
