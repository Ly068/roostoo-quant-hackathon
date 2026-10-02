"""
最终组合策略回测
================
动量策略（主力）+ 配对交易（减震器）
用波动率倒数加权，看组合 Sharpe 能否超过单独动量。
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.backtest.engine import Backtester
from rooster_trader.strategy.momentum import MomentumStrategy
from rooster_trader.strategy.momentum_with_buffer import MomentumWithCashBuffer
from rooster_trader.strategy.pairs_v2 import PairTradingV2
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
    print("最终组合策略回测：动量 + 配对交易")
    print("=" * 60)

    crypto_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    data = {}
    for sym in crypto_symbols:
        data[sym] = load_data(sym, interval="1h")

    min_len = min(len(df) for df in data.values())
    for sym in data:
        data[sym] = data[sym].iloc[-min_len:]

    print(f"\n加载完成: {len(crypto_symbols)} 个标的，{min_len} 根1小时K线\n")

    # ============================================================
    # 1. 单独动量策略（70%仓位）
    # ============================================================
    print("→ 回测：单独动量策略（70%仓位）...")
    mom = MomentumStrategy(
        symbols=crypto_symbols,
        lookback_hours=720,
        top_n=2,
        rebalance_hours=72,
    )
    mom_with_buffer = MomentumWithCashBuffer(mom, investment_ratio=0.70)
    bt1 = Backtester(mom_with_buffer, initial_capital=100_000, fee_rate=0.001)
    result_mom = bt1.run(data)
    print_result("单独动量策略（70%仓位）", result_mom)

    # ============================================================
    # 2. 单独配对交易（优化后参数，4小时级别在1小时数据上近似）
    # ============================================================
    print("\n→ 回测：单独配对交易（优化后）...")
    pairs = PairTradingV2(
        pair_symbols=[("BTCUSDT", "ETHUSDT")],
        window=80,           # 优化后最优
        entry_z=1.5,         # 优化后最优
        exit_z=0.5,
        stop_loss_z=4.0,     # 优化后最优
        pair_weight=0.15,
        trend_filter=True,
    )
    bt2 = Backtester(pairs, initial_capital=100_000, fee_rate=0.001)
    result_pair = bt2.run(data)
    print_result("单独配对交易（优化后）", result_pair)

    # ============================================================
    # 3. 组合策略：动量 + 配对，波动率倒数加权
    # ============================================================
    print("\n→ 回测：组合策略（动量70% + 配对，波动率加权）...")

    mom2 = MomentumStrategy(
        symbols=crypto_symbols,
        lookback_hours=720,
        top_n=2,
        rebalance_hours=72,
    )
    mom_buffer2 = MomentumWithCashBuffer(mom2, investment_ratio=0.70)

    pairs2 = PairTradingV2(
        pair_symbols=[("BTCUSDT", "ETHUSDT")],
        window=80,
        entry_z=1.5,
        exit_z=0.5,
        stop_loss_z=4.0,
        pair_weight=0.15,
        trend_filter=True,
    )

    combo = MultiStrategyEngine(
        strategies=[mom_buffer2, pairs2],
        weight_method="vol_inverse",
        vol_window=500,
        max_total_weight=0.85,
    )
    bt3 = Backtester(combo, initial_capital=100_000, fee_rate=0.001)
    result_combo = bt3.run(data)
    print_result("组合策略（动量+配对）", result_combo)

    # ============================================================
    # 对比总结
    # ============================================================
    print(f"\n{'='*75}")
    print("三策略对比总结")
    print(f"{'='*75}")
    print(f"{'指标':<18} {'单独动量':>12} {'单独配对':>12} {'组合策略':>12}")
    print("-" * 75)
    print(f"{'总收益率':<18} {result_mom.total_return:>12.2%} {result_pair.total_return:>12.2%} {result_combo.total_return:>12.2%}")
    print(f"{'年化波动率':<18} {result_mom.annual_volatility:>12.2%} {result_pair.annual_volatility:>12.2%} {result_combo.annual_volatility:>12.2%}")
    print(f"{'Sharpe 比率':<18} {result_mom.sharpe_ratio:>12.3f} {result_pair.sharpe_ratio:>12.3f} {result_combo.sharpe_ratio:>12.3f}")
    print(f"{'Sortino 比率':<18} {result_mom.sortino_ratio:>12.3f} {result_pair.sortino_ratio:>12.3f} {result_combo.sortino_ratio:>12.3f}")
    print(f"{'最大回撤':<18} {result_mom.max_drawdown:>12.2%} {result_pair.max_drawdown:>12.2%} {result_combo.max_drawdown:>12.2%}")
    print(f"{'Calmar 比率':<18} {result_mom.calmar_ratio:>12.3f} {result_pair.calmar_ratio:>12.3f} {result_combo.calmar_ratio:>12.3f}")

    # 判断组合是否优于单独动量
    if result_combo.sharpe_ratio > result_mom.sharpe_ratio:
        print(f"\n✅ 组合策略 Sharpe ({result_combo.sharpe_ratio:.3f}) > 单独动量 ({result_mom.sharpe_ratio:.3f})")
        print("   配对交易成功降低了组合波动率，提升了风险调整后收益！")
    else:
        print(f"\n⚠ 组合策略 Sharpe ({result_combo.sharpe_ratio:.3f}) < 单独动量 ({result_mom.sharpe_ratio:.3f})")
        print("   配对交易在1小时级别上效果有限，建议继续用单独动量策略")

    # 保存净值曲线
    result_combo.equity_curve.to_csv("data/final_combo_equity_curve.csv")
    print(f"\n✓ 组合策略净值曲线已保存到 data/final_combo_equity_curve.csv")


if __name__ == "__main__":
    main()
