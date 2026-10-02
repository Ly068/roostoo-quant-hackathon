"""
配对选择与优化回测
==================
1. 测试所有可能的配对组合
2. 对每个配对跑回测
3. 找出 Sharpe 最高的配对
4. 测试最优配对的参数网格
"""

import os
import sys
import itertools
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.backtest.engine import Backtester
from rooster_trader.strategy.pairs_v2 import PairTradingV2


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_data(symbol: str, interval: str = "4h") -> pd.DataFrame:
    filepath = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")
    return pd.read_csv(filepath, index_col=0, parse_dates=True)


def main():
    print("=" * 60)
    print("配对交易优化 - 4小时级别")
    print("=" * 60)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    data = {}
    for sym in symbols:
        data[sym] = load_data(sym, interval="4h")

    min_len = min(len(df) for df in data.values())
    for sym in data:
        data[sym] = data[sym].iloc[-min_len:]

    print(f"\n加载完成: {len(symbols)} 个标的，{min_len} 根4小时K线（约{min_len*4/24:.0f}天）\n")

    # ---- 第一步：测试所有配对组合 ----
    print("=" * 60)
    print("第一步：测试所有配对组合")
    print("=" * 60)

    all_pairs = list(itertools.combinations(symbols, 2))
    pair_results = []

    for pair in all_pairs:
        sym_a, sym_b = pair
        strategy = PairTradingV2(
            pair_symbols=[pair],
            window=30,
            entry_z=2.0,
            exit_z=0.5,
            stop_loss_z=3.5,
            pair_weight=0.15,
            trend_filter=True,
        )

        bt = Backtester(strategy, initial_capital=100_000, fee_rate=0.001, bars_per_year=6*365)
        result = bt.run(data)

        pair_results.append({
            "pair": f"{sym_a}/{sym_b}",
            "total_return": result.total_return,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "calmar": result.calmar_ratio,
            "trades": result.total_trades,
        })

        print(f"  {sym_a}/{sym_b:<12} 收益:{result.total_return:>7.2%}  "
              f"Sharpe:{result.sharpe_ratio:>7.3f}  回撤:{result.max_drawdown:>7.2%}  "
              f"交易:{result.total_trades:>4d}次")

    # 按 Sharpe 排序
    pair_results.sort(key=lambda x: x["sharpe"], reverse=True)
    best_pair = pair_results[0]
    print(f"\n★ 最优配对: {best_pair['pair']} (Sharpe = {best_pair['sharpe']:.3f})")

    # ---- 第二步：对最优配对做参数网格搜索 ----
    print("\n" + "=" * 60)
    print(f"第二步：对最优配对 {best_pair['pair']} 做参数网格搜索")
    print("=" * 60)

    best_pair_symbols = tuple(best_pair["pair"].split("/"))

    window_options = [20, 30, 50, 80]
    entry_z_options = [1.5, 2.0, 2.5]
    stop_loss_options = [3.0, 3.5, 4.0]

    grid_results = []
    total = len(window_options) * len(entry_z_options) * len(stop_loss_options)
    count = 0

    for window, entry_z, stop_loss in itertools.product(window_options, entry_z_options, stop_loss_options):
        count += 1
        print(f"  进度 {count}/{total}: window={window}, entry_z={entry_z}, stop_loss={stop_loss}", end="\r")

        strategy = PairTradingV2(
            pair_symbols=[best_pair_symbols],
            window=window,
            entry_z=entry_z,
            exit_z=0.5,
            stop_loss_z=stop_loss,
            pair_weight=0.15,
            trend_filter=True,
        )

        bt = Backtester(strategy, initial_capital=100_000, fee_rate=0.001, bars_per_year=6*365)
        result = bt.run(data)

        grid_results.append({
            "window": window,
            "entry_z": entry_z,
            "stop_loss_z": stop_loss,
            "total_return": result.total_return,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "calmar": result.calmar_ratio,
            "trades": result.total_trades,
        })

    print()  # 换行

    # 展示 Top 10
    grid_results.sort(key=lambda x: x["sharpe"], reverse=True)
    print("\n参数网格搜索 - Sharpe Top 10:")
    print("-" * 80)
    for r in grid_results[:10]:
        print(f"  window={r['window']:>3d}  entry_z={r['entry_z']:.1f}  "
              f"stop_loss={r['stop_loss_z']:.1f}  收益:{r['total_return']:>7.2%}  "
              f"Sharpe:{r['sharpe']:>7.3f}  回撤:{r['max_drawdown']:>7.2%}")

    best_params = grid_results[0]
    print(f"\n★ 最优参数: window={best_params['window']}, entry_z={best_params['entry_z']}, "
          f"stop_loss={best_params['stop_loss_z']}")
    print(f"  → Sharpe = {best_params['sharpe']:.3f}, 收益 = {best_params['total_return']:.2%}, "
          f"回撤 = {best_params['max_drawdown']:.2%}")

    # 保存结果
    pd.DataFrame(pair_results).to_csv("data/pair_selection_results.csv", index=False)
    pd.DataFrame(grid_results).to_csv("data/pair_grid_search_results.csv", index=False)
    print("\n✓ 结果已保存到 data/pair_selection_results.csv 和 data/pair_grid_search_results.csv")


if __name__ == "__main__":
    main()
