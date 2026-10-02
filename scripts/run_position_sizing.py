"""
主力策略最终回测
================
对比：
1. 满仓动量策略
2. 80%动量 + 20%现金缓冲
3. 70%动量 + 30%现金缓冲

目标：找到收益和回撤的最佳平衡点
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.backtest.engine import Backtester
from rooster_trader.strategy.momentum import MomentumStrategy
from rooster_trader.strategy.momentum_with_buffer import MomentumWithCashBuffer


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_data(symbol: str, interval: str = "1h") -> pd.DataFrame:
    filepath = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")
    return pd.read_csv(filepath, index_col=0, parse_dates=True)


def print_result(name: str, result):
    print(f"\n{'='*60}")
    print(f"策略: {name}")
    print(f"{'='*60}")
    print(f"  总收益率:       {result.total_return:>8.2%}")
    print(f"  年化收益率:     {result.annual_return:>8.2%}")
    print(f"  年化波动率:     {result.annual_volatility:>8.2%}")
    print(f"  Sharpe 比率:    {result.sharpe_ratio:>8.3f}")
    print(f"  Sortino 比率:   {result.sortino_ratio:>8.3f}")
    print(f"  Calmar 比率:    {result.calmar_ratio:>8.3f}")
    print(f"  最大回撤:       {result.max_drawdown:>8.2%}")
    print(f"  总交易次数:     {result.total_trades:>8d}")


def main():
    print("=" * 60)
    print("主力策略优化：仓位大小对比")
    print("=" * 60)

    crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    data = {}
    for sym in crypto_symbols:
        data[sym] = load_data(sym)

    min_len = min(len(df) for df in data.values())
    for sym in data:
        data[sym] = data[sym].iloc[-min_len:]

    print(f"\n加载完成: {len(crypto_symbols)} 个标的，{min_len} 根1小时K线\n")

    results = {}

    # ---- 测试不同仓位比例 ----
    for ratio in [1.0, 0.8, 0.7, 0.6, 0.5]:
        label = f"投资{int(ratio*100)}% + 现金{int((1-ratio)*100)}%"
        print(f"→ 回测: {label}...")

        mom = MomentumStrategy(
            symbols=crypto_symbols,
            lookback_hours=720,
            top_n=2,
            rebalance_hours=72,
        )
        strategy = MomentumWithCashBuffer(mom, investment_ratio=ratio)

        bt = Backtester(strategy, initial_capital=100_000, fee_rate=0.001)
        result = bt.run(data)
        results[label] = result
        print_result(label, result)

    # ---- 对比总结 ----
    print(f"\n{'='*75}")
    print("仓位比例对比（找Sharpe最高的）")
    print(f"{'='*75}")
    print(f"{'仓位方案':<25} {'总收益':>10} {'Sharpe':>10} {'Sortino':>10} {'最大回撤':>10} {'Calmar':>10}")
    print("-" * 75)

    best_sharpe = -999
    best_label = ""
    for label, r in results.items():
        print(f"{label:<25} {r.total_return:>10.2%} {r.sharpe_ratio:>10.3f} "
              f"{r.sortino_ratio:>10.3f} {r.max_drawdown:>10.2%} {r.calmar_ratio:>10.3f}")
        if r.sharpe_ratio > best_sharpe:
            best_sharpe = r.sharpe_ratio
            best_label = label

    print(f"\n★ 最优仓位方案: {best_label} (Sharpe = {best_sharpe:.3f})")
    print("  → 这个方案在收益和风险之间取得了最佳平衡")


if __name__ == "__main__":
    main()
