"""
组合策略回测
============
测试动量 + 配对交易的组合效果。
对比：
1. 单独动量策略
2. 单独配对交易
3. 组合策略（波动率倒数加权）
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.backtest.engine import Backtester
from rooster_trader.strategy.momentum import MomentumStrategy
from rooster_trader.strategy.pairs import PairTradingStrategy
from rooster_trader.strategy.multi_strategy import MultiStrategyEngine


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
    print("组合策略回测：动量 + 配对交易")
    print("=" * 60)

    # ---- 加载数据 ----
    crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    data = {}
    for sym in crypto_symbols:
        data[sym] = load_data(sym)

    min_len = min(len(df) for df in data.values())
    for sym in data:
        data[sym] = data[sym].iloc[-min_len:]

    print(f"\n加载完成: {len(crypto_symbols)} 个标的，{min_len} 根1小时K线")

    # ============================================================
    # 1. 单独动量策略（最优参数）
    # ============================================================
    print("\n→ 回测：单独动量策略...")
    momentum = MomentumStrategy(
        symbols=crypto_symbols,
        lookback_hours=720,       # 30天
        top_n=2,
        rebalance_hours=72,       # 3天
    )
    bt_mom = Backtester(momentum, initial_capital=100_000, fee_rate=0.001)
    result_mom = bt_mom.run(data)
    print_result("单独动量策略", result_mom)

    # ============================================================
    # 2. 单独配对交易（调整后参数：换配对、降阈值、降权重）
    # ============================================================
    print("\n→ 回测：单独配对交易（调整后）...")
    pairs = PairTradingStrategy(
        pair_symbols=[("BTCUSDT", "SOLUSDT")],   # 换配对：BTC/SOL
        window=20,
        entry_z=1.8,                              # 降低入场阈值
        exit_z=0.5,
        pair_weight=0.10,                         # 降低权重到10%
    )
    bt_pair = Backtester(pairs, initial_capital=100_000, fee_rate=0.001)
    result_pair = bt_pair.run(data)
    print_result("单独配对交易", result_pair)

    # ============================================================
    # 3. 组合策略：波动率倒数加权
    # ============================================================
    print("\n→ 回测：组合策略（动量 + 配对，波动率加权）...")

    # 重新创建策略实例（之前的已经被回测跑过，有内部状态）
    momentum2 = MomentumStrategy(
        symbols=crypto_symbols,
        lookback_hours=720,
        top_n=2,
        rebalance_hours=72,
    )
    pairs2 = PairTradingStrategy(
        pair_symbols=[("BTCUSDT", "SOLUSDT")],
        window=20,
        entry_z=1.8,
        exit_z=0.5,
        pair_weight=0.10,
    )

    combo = MultiStrategyEngine(
        strategies=[momentum2, pairs2],
        weight_method="vol_inverse",
        vol_window=500,
        max_total_weight=0.75,   # 总敞口控制在75%，留25%现金
    )
    bt_combo = Backtester(combo, initial_capital=100_000, fee_rate=0.001)
    result_combo = bt_combo.run(data)
    print_result("组合策略（波动率加权）", result_combo)

    # ============================================================
    # 对比总结
    # ============================================================
    print(f"\n{'='*70}")
    print("三策略对比总结")
    print(f"{'='*70}")
    print(f"{'指标':<18} {'单独动量':>12} {'单独配对':>12} {'组合策略':>12}")
    print("-" * 70)
    print(f"{'总收益率':<18} {result_mom.total_return:>12.2%} {result_pair.total_return:>12.2%} {result_combo.total_return:>12.2%}")
    print(f"{'年化波动率':<18} {result_mom.annual_volatility:>12.2%} {result_pair.annual_volatility:>12.2%} {result_combo.annual_volatility:>12.2%}")
    print(f"{'Sharpe 比率':<18} {result_mom.sharpe_ratio:>12.3f} {result_pair.sharpe_ratio:>12.3f} {result_combo.sharpe_ratio:>12.3f}")
    print(f"{'Sortino 比率':<18} {result_mom.sortino_ratio:>12.3f} {result_pair.sortino_ratio:>12.3f} {result_combo.sortino_ratio:>12.3f}")
    print(f"{'最大回撤':<18} {result_mom.max_drawdown:>12.2%} {result_pair.max_drawdown:>12.2%} {result_combo.max_drawdown:>12.2%}")
    print(f"{'Calmar 比率':<18} {result_mom.calmar_ratio:>12.3f} {result_pair.calmar_ratio:>12.3f} {result_combo.calmar_ratio:>12.3f}")

    # 保存净值曲线
    result_combo.equity_curve.to_csv("data/combo_equity_curve.csv")
    print(f"\n✓ 组合策略净值曲线已保存到 data/combo_equity_curve.csv")


if __name__ == "__main__":
    main()
