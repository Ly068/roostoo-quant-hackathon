"""
真实数据策略重扫（Binance，0.1%手续费，72h调仓）
================================================
对比三类方案，寻找在 6-10 月真实市场稳健有效的策略：
  A. 买入持有 BTC（基准）
  B. 横截面只做多动量（去空头）：窗口 × topN
  C. 时序趋势跟踪（无趋势空仓避险）：窗口 × MA过滤开关

先全周期扫描，再对 Sharpe Top3 做三段独立窗口验证。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rooster_trader.strategy.momentum import MomentumStrategy
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


def run(strat, data, stop=0.0):
    bt = Backtester(strategy=strat, initial_capital=50000, fee_rate=0.001,
                     min_rebalance_bars=72, rebalance_threshold=0.01,
                     stop_loss_pct=stop)
    return bt.run(data)


def bh_btc(data, warmup=720):
    """买入持有BTC基准（一次买入，扣一次手续费）"""
    px = data["BTC/USD"]["close"]
    ret = px.iloc[-1] / px.iloc[warmup] * (1 - 0.001) - 1
    return ret


def configs(symbols):
    cfgs = []
    # B. 横截面只做多
    for lb in [168, 336, 600]:
        for tn in [2, 3]:
            ps = 1.0 / tn  # 总多头≈100%
            m = MomentumStrategy(symbols, lookback_hours=lb, top_n=tn,
                                 rebalance_hours=72, position_size=ps,
                                 sizing_method="equal", long_only=True)
            cfgs.append((f"XS只做多 {lb//24}天 top{tn}", m))
    # C. 时序趋势
    for lb in [168, 336, 720]:
        for maf in [True, False]:
            t = TrendFollowingStrategy(symbols, lookback_hours=lb, ma_hours=lb,
                                       use_ma_filter=maf, max_total_exposure=1.0,
                                       rebalance_hours=72)
            tag = "MA过滤" if maf else "无MA"
            cfgs.append((f"时序趋势 {lb//24}天 {tag}", t))
    return cfgs


def main():
    data = load_data()
    symbols = list(data.keys())
    total = len(data[symbols[0]])
    print("=" * 104)
    print(f"真实数据策略重扫（{len(symbols)}币种，{total/24:.0f}天）")
    print("=" * 104)

    bh = bh_btc(data)
    print(f"\n【基准】买入持有BTC 全周期收益: {bh*100:+.2f}%")

    print("\n" + "-" * 104)
    print(f"{'配置':<26}{'收益':>9}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'回撤':>9}{'交易':>5}")
    print("-" * 104)
    rows = []
    for name, strat in configs(symbols):
        r = run(strat, data)
        print(f"{name:<26}{r.total_return*100:>+8.2f}%{r.sharpe_ratio:>+8.2f}"
              f"{r.sortino_ratio:>+9.2f}{r.calmar_ratio:>+8.2f}"
              f"{r.max_drawdown*100:>+8.2f}%{r.total_trades:>5}")
        rows.append((name, strat, r))

    # Top3 by Sharpe（且要求正收益）
    top = sorted([x for x in rows if x[2].total_return > 0],
                 key=lambda z: z[2].sharpe_ratio, reverse=True)[:3]

    # ---- 三段独立窗口 ----
    print("\n" + "=" * 104)
    print("Top3 候选 + B&H 的三段独立窗口验证（每段约29天，自带30天预热）")
    print("=" * 104)
    win = 720
    step = 720
    bounds = [(s, min(s + win + win, total)) for s in range(0, total - win - 100, step)][:3]

    def seg_run(strat, a, b):
        sub = {x: df.iloc[a:b] for x, df in data.items()}
        rr = run(strat, sub)
        eq = rr.equity_curve
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
        return ret, sh, dd

    print(f"{'配置':<26}{'段1':>14}{'段2':>14}{'段3':>14}{'正收益段':>9}")
    for name, strat, _ in top:
        seg = [seg_run(strat, a, b) for a, b in bounds]
        pos = sum(1 for x in seg if x[0] > 0)
        cells = "".join(f"{x[0]*100:>+7.2f}%/{x[1]:>+4.1f}" for x in seg)
        print(f"{name:<26}" + "".join(f"{x[0]*100:>+8.2f}%" for x in seg) + f"{pos:>7}/3")
    # B&H分段
    bhseg = []
    for a, b in bounds:
        px = data["BTC/USD"]["close"].iloc[a:b]
        bhseg.append(px.iloc[-1] / px.iloc[win] - 1)
    print(f"{'B&H BTC':<26}" + "".join(f"{x*100:>+8.2f}%" for x in bhseg) +
          f"{sum(1 for x in bhseg if x>0):>7}/3")
    print("=" * 104)


if __name__ == "__main__":
    main()
