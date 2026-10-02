"""
快速策略扫描：测试不同策略和参数在当前回测区间的表现
"""
import os, sys, yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.strategy.momentum import MomentumStrategy
from rooster_trader.strategy.volatility_breakout import VolatilityBreakoutStrategy
from rooster_trader.strategy.enhanced_multi import EnhancedMultiStrategy
from rooster_trader.strategy.dynamic_position import DynamicPositionStrategy
from rooster_trader.backtest.engine import Backtester

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "history")

def load_data():
    data = {}
    for coin in ["BTC", "ETH", "SOL", "BNB", "XRP"]:
        fp = os.path.join(DATA_DIR, f"{coin}_1h.csv")
        if os.path.exists(fp):
            data[f"{coin}/USD"] = pd.read_csv(fp, index_col=0, parse_dates=True)
    return data

def quick_backtest(strategy, data, name):
    bt = Backtester(strategy=strategy, initial_capital=50000, fee_rate=0.001,
                     min_rebalance_bars=72, rebalance_threshold=0.01)
    r = bt.run(data)
    print(f"  {name:40s} | 收益{r.total_return*100:+6.2f}% | Sharpe{r.sharpe_ratio:+6.2f} | 回撤{r.max_drawdown*100:+6.2f}% | 交易{r.total_trades:3d}")
    return r

def main():
    data = load_data()
    symbols = list(data.keys())
    print("=" * 90)
    print("策略快速扫描（回测区间2026-08-03~10-01，72h调仓，0.1%手续费）")
    print("=" * 90)

    # ---- 1. 不同动量窗口测试 ----
    print("\n【1】动量策略：不同窗口+top1+25%仓位")
    print("-" * 90)
    for lookback in [168, 336, 504, 720]:
        for top_n in [1, 2]:
            mom = MomentumStrategy(symbols=symbols, lookback_hours=lookback,
                                   top_n=top_n, rebalance_hours=72, position_size=0.25)
            quick_backtest(mom, data, f"动量{lookback//24}天top{top_n}")

    # ---- 2. 纯波动率突破策略 ----
    print("\n【2】波动率突破策略（单独）")
    print("-" * 90)
    for max_pos in [1, 2]:
        for atr_stop in [1.5, 2.0, 3.0]:
            bo = VolatilityBreakoutStrategy(symbols=symbols, bb_window=20, bb_std=2.0,
                                              squeeze_percentile=0.2, atr_window=14,
                                              atr_stop_mult=atr_stop, atr_target_mult=atr_stop*1.5,
                                              max_positions=max_pos)
            quick_backtest(bo, data, f"突破max{max_pos}止损{atr_stop}ATR")

    # ---- 3. 动量+突破组合（不同权重） ----
    print("\n【3】动量+突破组合（不同权重，动量14天top2）")
    print("-" * 90)
    mom = MomentumStrategy(symbols=symbols, lookback_hours=336, top_n=2,
                           rebalance_hours=72, position_size=0.15)
    bo = VolatilityBreakoutStrategy(symbols=symbols, bb_window=20, bb_std=2.0,
                                      squeeze_percentile=0.2, atr_window=14,
                                      atr_stop_mult=2.0, atr_target_mult=3.0, max_positions=2)
    for mom_w, bo_w in [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]:
        combo = EnhancedMultiStrategy(symbols=symbols,
                                       strategies={"momentum": mom, "breakout": bo},
                                       base_weights={"momentum": mom_w, "breakout": bo_w},
                                       use_regime_adjust=True, use_xgb_filter=False,
                                       use_strategy_circuit_breaker=True)
        strat = DynamicPositionStrategy(underlying_strategy=combo,
                                         competition_start_date="2026-10-04", competition_days=14)
        quick_backtest(strat, data, f"动量{int(mom_w*100)}%+突破{int(bo_w*100)}%")

    # ---- 4. 纯突破+动态仓位 ----
    print("\n【4】纯突破策略+动态仓位（确认期100%）")
    print("-" * 90)
    bo2 = VolatilityBreakoutStrategy(symbols=symbols, bb_window=20, bb_std=2.0,
                                       squeeze_percentile=0.2, atr_window=14,
                                       atr_stop_mult=2.0, atr_target_mult=3.0, max_positions=2)
    combo_bo = EnhancedMultiStrategy(symbols=symbols,
                                      strategies={"breakout": bo2},
                                      base_weights={"breakout": 1.0},
                                      use_regime_adjust=True, use_xgb_filter=False,
                                      use_strategy_circuit_breaker=True)
    strat_bo = DynamicPositionStrategy(underlying_strategy=combo_bo,
                                        competition_start_date="2026-10-04", competition_days=14)
    quick_backtest(strat_bo, data, "纯突破100%+动态仓位")

    print("\n" + "=" * 90)
    print("扫描完成")
    print("=" * 90)

if __name__ == "__main__":
    main()
