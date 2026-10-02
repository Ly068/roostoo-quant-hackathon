"""
测试环境完整调仓验证（时序趋势只做多 v5）
流程：
  1. 记录初始状态
  2. Binance冷启动数据计算目标权重（只做多）
  3. 执行调仓（买入处于上升趋势的币，等权）
  4. 验证持仓与多头敞口
  5. 复位（卖出现货，恢复空仓）
  6. 汇总
运行：/opt/anaconda3/bin/python3 scripts/test_live_rebalance.py
"""
import os, sys, time, yaml
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rooster_trader.client import RoosterClient
from rooster_trader.execution import ExecutionEngine
from rooster_trader.strategy.trend import TrendFollowingStrategy
from rooster_trader.strategy.dynamic_position import DynamicPositionStrategy

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "history_binance")


def snapshot(client, label):
    pv = client.get_portfolio_value()
    bal = client.get_balance().get("SpotWallet", {})
    print(f"\n--- {label} ---")
    print(f"  组合市值: ${pv:,.2f}")
    held, usd_free = {}, 0
    for coin, info in bal.items():
        free = float(info.get("Free", 0))
        if coin == "USD":
            usd_free = free
        elif free > 0:
            held[coin] = round(free, 6)
    print(f"  现货持仓: {held}")
    print(f"  USD余额: ${usd_free:,.2f}")
    return pv, held


def flatten_all(client, executor, held):
    """复位：卖出现货全部"""
    print("\n--- 复位：卖出现货全部 ---")
    for coin, qty in held.items():
        if coin != "USD" and qty > 0:
            q = executor._adjust_quantity(f"{coin}/USD", qty)
            r = client.place_order(pair=f"{coin}/USD", side="SELL",
                                   quantity=q, order_type="MARKET")
            print(f"  卖 {coin}: {'成功' if r.get('Success') else r.get('ErrMsg')}")
            time.sleep(0.5)


def main():
    cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config.yaml"), encoding="utf-8"))
    client = RoosterClient(cfg["TEST_API_KEY"], cfg["TEST_API_SECRET"])
    executor = ExecutionEngine(client=client, min_trade_pct=0.003)

    print("=" * 70)
    print("测试环境完整调仓验证（时序趋势只做多 v5）")
    print("=" * 70)

    # 1. 初始状态
    pv0, held0 = snapshot(client, "初始状态")

    # 2. Binance冷启动数据 + 目标权重
    coins = [l.strip() for l in open(os.path.join(DATA_DIR, "universe.txt"),
                                    encoding="utf-8") if l.strip()]
    symbols = [f"{c}/USD" for c in coins]
    data = {}
    for c in coins:
        data[f"{c}/USD"] = pd.read_csv(os.path.join(DATA_DIR, f"{c}_1h.csv"),
                                       index_col=0, parse_dates=True)
    common = None
    for s, df in data.items():
        common = df.index if common is None else common.intersection(df.index)
    data = {s: df.loc[common] for s, df in data.items()}

    trend = TrendFollowingStrategy(symbols=symbols, lookback_hours=720, ma_hours=720,
                                   use_ma_filter=False, max_total_exposure=1.0,
                                   rebalance_hours=72)
    strat = DynamicPositionStrategy(underlying_strategy=trend,
                                    competition_start_date="2026-10-04",
                                    competition_days=14)
    target = strat.on_bar(data)
    active = {k: f"{v:.0%}" for k, v in target.items() if v != 0}
    print(f"\n目标仓位（只做多，{len(active)}币）: {active}")

    # 3. 执行调仓
    summary = executor.rebalance(target)
    time.sleep(2)

    # 4. 验证
    pv1, held1 = snapshot(client, "调仓后状态")

    # 5. 复位
    flatten_all(client, executor, held1)
    time.sleep(2)
    pv2, held2 = snapshot(client, "复位后状态")

    # 汇总
    print("\n" + "=" * 70)
    print("验证汇总")
    print("=" * 70)
    checks = [
        ("调仓计划执行", summary["orders_failed"] == 0,
         f"现货成功{summary['orders_executed']}, 失败{summary['orders_failed']}"),
        ("调仓后建立多头", len(held1) >= 1, f"现货{list(held1.keys())}"),
        ("手续费已扣除", abs(pv1 - pv0) > 0, f"市值变动{pv1-pv0:+.2f}（含手续费）"),
        ("复位后现货清空", len(held2) == 0, f"残留{held2 if held2 else '无'}"),
        ("复位后市值接近初始", abs(pv2-pv0) < pv0*0.02,
         f"初始${pv0:,.0f}→复位${pv2:,.2f}（手续费/滑点{pv2-pv0:+.2f}）"),
    ]
    passed = 0
    for name, ok, detail in checks:
        print(f"  {'✓' if ok else '✗'} {name}: {detail}")
        passed += ok
    print(f"\n  结果: {passed}/{len(checks)} 通过")
    print("=" * 70)


if __name__ == "__main__":
    main()
