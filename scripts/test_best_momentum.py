"""
精确回测：30天动量top1的最佳配置 + 动态仓位
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.strategy.momentum import MomentumStrategy
from rooster_trader.strategy.volatility_breakout import VolatilityBreakoutStrategy
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

def detailed_backtest(strategy, data, name):
    bt = Backtester(strategy=strategy, initial_capital=50000, fee_rate=0.001,
                     min_rebalance_bars=72, rebalance_threshold=0.01)
    r = bt.run(data)
    print(f"\n{'='*70}")
    print(f"回测: {name}")
    print(f"{'='*70}")
    print(f"  总收益率:     {r.total_return*100:+.2f}%")
    print(f"  年化收益率:   {r.annual_return*100:+.2f}%")
    print(f"  年化波动率:   {r.annual_volatility*100:.2f}%")
    print(f"  Sharpe比率:   {r.sharpe_ratio:+.2f}")
    print(f"  Sortino比率:  {r.sortino_ratio:+.2f}")
    print(f"  最大回撤:     {r.max_drawdown*100:+.2f}%")
    print(f"  Calmar比率:   {r.calmar_ratio:+.2f}")
    print(f"  交易次数:     {r.total_trades}")
    print(f"  最终净值:     ${r.equity_curve.iloc[-1]:,.2f}")
    print(f"  胜率:         {r.win_rate*100:.1f}%")

    # 分阶段收益
    eq = r.equity_curve
    n = len(eq)
    print(f"\n  分阶段收益（每10天）:")
    for i in range(0, n, 240):
        end = min(i+240, n-1)
        if end > i:
            ret = (eq.iloc[end] / eq.iloc[i] - 1) * 100
            ds = eq.index[i]
            de = eq.index[end]
            ds_str = ds.strftime('%m-%d') if hasattr(ds, 'strftime') else str(ds)[:5]
            de_str = de.strftime('%m-%d') if hasattr(de, 'strftime') else str(de)[:5]
            print(f"    {ds_str} ~ {de_str}: {ret:+.2f}%")

    # 14天预期
    daily_ret = r.total_return / (n / 24)
    print(f"\n  14天比赛预期收益估算:")
    print(f"    策略日收益能力: {daily_ret*100:+.4f}%/天")
    for pos_label, pos in [("50%仓位", 0.5), ("67.9%动态仓位", 0.679), ("100%仓位", 1.0)]:
        exp = daily_ret * 14 * pos * 100
        print(f"    {pos_label}: {exp:+.2f}%")

    return r

def main():
    data = load_data()
    symbols = list(data.keys())

    # ================================================================
    # 方案A：30天动量top1，25%单仓位（50%总敞口），动态仓位
    # ================================================================
    mom_a = MomentumStrategy(symbols=symbols, lookback_hours=720, top_n=1,
                              rebalance_hours=72, position_size=0.25)
    strat_a = DynamicPositionStrategy(underlying_strategy=mom_a,
                                       competition_start_date="2026-10-04", competition_days=14)
    detailed_backtest(strat_a, data, "方案A: 30天动量top1 + 25%仓位 + 动态仓位")

    # ================================================================
    # 方案B：30天动量top1，30%单仓位（60%总敞口），动态仓位
    # ================================================================
    mom_b = MomentumStrategy(symbols=symbols, lookback_hours=720, top_n=1,
                              rebalance_hours=72, position_size=0.30)
    strat_b = DynamicPositionStrategy(underlying_strategy=mom_b,
                                       competition_start_date="2026-10-04", competition_days=14)
    detailed_backtest(strat_b, data, "方案B: 30天动量top1 + 30%仓位 + 动态仓位")

    # ================================================================
    # 方案C：30天动量top1（25%）+ 突破max1+1.5ATR（25%），简单等权
    # ================================================================
    mom_c = MomentumStrategy(symbols=symbols, lookback_hours=720, top_n=1,
                              rebalance_hours=72, position_size=0.20)
    bo_c = VolatilityBreakoutStrategy(symbols=symbols, bb_window=20, bb_std=2.0,
                                        squeeze_percentile=0.2, atr_window=14,
                                        atr_stop_mult=1.5, atr_target_mult=2.25, max_positions=1)

    class SimpleCombo:
        def __init__(self, s1, s2, w1=0.5, w2=0.5):
            self.s1, self.s2, self.w1, self.w2 = s1, s2, w1, w2
            self.name = "SimpleCombo"
            self.symbols = s1.symbols
        def warmup_bars(self):
            return max(self.s1.warmup_bars(), self.s2.warmup_bars())
        def on_bar(self, bars):
            w1 = self.s1.on_bar(bars)
            w2 = self.s2.on_bar(bars)
            result = {}
            for s in set(list(w1.keys()) + list(w2.keys())):
                result[s] = self.w1 * w1.get(s, 0) + self.w2 * w2.get(s, 0)
            return result

    combo_c = SimpleCombo(mom_c, bo_c, 0.6, 0.4)
    strat_c = DynamicPositionStrategy(underlying_strategy=combo_c,
                                       competition_start_date="2026-10-04", competition_days=14)
    detailed_backtest(strat_c, data, "方案C: 30天动量top1(60%) + 突破max1(40%) + 动态仓位")

    # ================================================================
    # 方案D：纯30天动量top1，无动态仓位（满仓）
    # ================================================================
    mom_d = MomentumStrategy(symbols=symbols, lookback_hours=720, top_n=1,
                              rebalance_hours=72, position_size=0.50)  # 50%单仓位=100%总敞口
    detailed_backtest(mom_d, data, "方案D: 30天动量top1 + 50%单仓位(满仓) + 无动态仓位")

    print("\n" + "="*70)
    print("全部方案回测完成")
    print("="*70)

if __name__ == "__main__":
    main()
