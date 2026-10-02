"""
参数网格搜索（Grid Search）
=============================
自动遍历不同参数组合，找出 Sharpe 最高的参数配置。

输出：
1. 所有参数组合的绩效表格
2. 按 Sharpe 排序的 Top 10
3. 最优参数建议
"""

import os
import sys
import itertools
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.backtest.engine import Backtester
from rooster_trader.strategy.momentum import MomentumStrategy
from rooster_trader.strategy.pairs import PairTradingStrategy


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_data(symbol: str, interval: str = "1h") -> pd.DataFrame:
    filepath = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def grid_search_momentum(data: dict) -> pd.DataFrame:
    """
    动量策略参数网格搜索。
    遍历：lookback_hours × top_n × rebalance_hours
    """
    print("=" * 60)
    print("动量策略 - 参数网格搜索")
    print("=" * 60)

    # 参数网格
    lookback_options = [72, 168, 336, 720]      # 3天 / 7天 / 14天 / 30天
    top_n_options = [2, 3]                        # 做多/做空各选几个
    rebalance_options = [72, 168, 336]           # 3天 / 7天 / 14天

    results = []
    total_combos = len(lookback_options) * len(top_n_options) * len(rebalance_options)
    count = 0

    for lookback, top_n, rebal in itertools.product(lookback_options, top_n_options, rebalance_options):
        count += 1
        print(f"  进度 {count}/{total_combos}: lookback={lookback}h, top_n={top_n}, rebal={rebal}h", end="\r")

        strategy = MomentumStrategy(
            symbols=list(data.keys()),
            lookback_hours=lookback,
            top_n=top_n,
            rebalance_hours=rebal,
        )

        bt = Backtester(strategy, initial_capital=100_000, fee_rate=0.001)
        result = bt.run(data)

        results.append({
            "lookback_hours": lookback,
            "top_n": top_n,
            "rebalance_hours": rebal,
            "total_return": result.total_return,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "calmar": result.calmar_ratio,
            "trades": result.total_trades,
        })

    print()  # 换行
    return pd.DataFrame(results)


def grid_search_pairs(data: dict) -> pd.DataFrame:
    """
    配对交易参数网格搜索。
    遍历：window × entry_z × exit_z
    """
    print("\n" + "=" * 60)
    print("配对交易策略 - 参数网格搜索")
    print("=" * 60)

    window_options = [10, 20, 50, 100]
    entry_z_options = [1.5, 2.0, 2.5, 3.0]
    exit_z_options = [0.3, 0.5, 1.0]

    results = []
    total_combos = len(window_options) * len(entry_z_options) * len(exit_z_options)
    count = 0

    for window, entry_z, exit_z in itertools.product(window_options, entry_z_options, exit_z_options):
        count += 1
        print(f"  进度 {count}/{total_combos}: window={window}, entry_z={entry_z}, exit_z={exit_z}", end="\r")

        strategy = PairTradingStrategy(
            pair_symbols=[("BTCUSDT", "ETHUSDT")],
            window=window,
            entry_z=entry_z,
            exit_z=exit_z,
            pair_weight=0.15,
        )

        bt = Backtester(strategy, initial_capital=100_000, fee_rate=0.001)
        result = bt.run(data)

        results.append({
            "window": window,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "total_return": result.total_return,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "calmar": result.calmar_ratio,
            "trades": result.total_trades,
        })

    print()
    return pd.DataFrame(results)


def main():
    # 加载数据
    crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    data = {}
    for sym in crypto_symbols:
        data[sym] = load_data(sym)

    min_len = min(len(df) for df in data.values())
    for sym in data:
        data[sym] = data[sym].iloc[-min_len:]

    print(f"加载完成: {len(crypto_symbols)} 个标的，{min_len} 根1小时K线\n")

    # ---- 动量策略网格搜索 ----
    mom_results = grid_search_momentum(data)
    mom_results.to_csv("data/grid_search_momentum.csv", index=False)

    # 按 Sharpe 排序，展示 Top 10
    print("\n动量策略 - Sharpe Top 10:")
    print("-" * 80)
    top_mom = mom_results.nlargest(10, "sharpe")[
        ["lookback_hours", "top_n", "rebalance_hours", "total_return", "sharpe", "sortino", "max_drawdown", "calmar"]
    ]
    print(top_mom.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 找最优组合
    best_mom = mom_results.loc[mom_results["sharpe"].idxmax()]
    print(f"\n★ 动量策略最优参数:")
    print(f"   lookback = {int(best_mom['lookback_hours'])}h ({best_mom['lookback_hours']/24:.0f}天)")
    print(f"   top_n = {int(best_mom['top_n'])}")
    print(f"   rebalance = {int(best_mom['rebalance_hours'])}h ({best_mom['rebalance_hours']/24:.0f}天)")
    print(f"   → Sharpe = {best_mom['sharpe']:.3f}, 收益 = {best_mom['total_return']:.2%}, 最大回撤 = {best_mom['max_drawdown']:.2%}")

    # ---- 配对交易网格搜索 ----
    pair_results = grid_search_pairs(data)
    pair_results.to_csv("data/grid_search_pairs.csv", index=False)

    print("\n配对交易 - Sharpe Top 10:")
    print("-" * 80)
    top_pair = pair_results.nlargest(10, "sharpe")[
        ["window", "entry_z", "exit_z", "total_return", "sharpe", "sortino", "max_drawdown", "calmar"]
    ]
    print(top_pair.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best_pair = pair_results.loc[pair_results["sharpe"].idxmax()]
    print(f"\n★ 配对交易最优参数:")
    print(f"   window = {int(best_pair['window'])}根K线")
    print(f"   entry_z = {best_pair['entry_z']}")
    print(f"   exit_z = {best_pair['exit_z']}")
    print(f"   → Sharpe = {best_pair['sharpe']:.3f}, 收益 = {best_pair['total_return']:.2%}, 最大回撤 = {best_pair['max_drawdown']:.2%}")

    print("\n" + "=" * 60)
    print("网格搜索完成！结果已保存到 data/grid_search_*.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
