"""
赛前检查清单（金奖版 v6：低波异象 + 风险平价）
运行：/opt/anaconda3/bin/python3 scripts/pre_flight_check.py
逐项确认所有模块、数据、参数、API正常。
"""
import os, sys, yaml
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

results = []

def check(name, ok, detail=""):
    results.append((name, ok, detail))
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}: {detail}")

print("=" * 70)
print("赛前检查清单（低波异象 v6）")
print("=" * 70)

# 1. 核心文件
print("\n[1/7] 核心文件检查")
core_files = [
    "main.py", "config.yaml",
    "rooster_trader/client.py", "rooster_trader/execution.py",
    "rooster_trader/risk/manager.py",
    "rooster_trader/strategy/low_vol_parity.py",
    "rooster_trader/strategy/dynamic_position.py",
    "rooster_trader/data/binance_feed.py",
    "rooster_trader/backtest/engine.py",
    "dashboard/app.py", "deploy.sh",
]
for f in core_files:
    exists = os.path.exists(os.path.join(PROJECT_ROOT, f))
    check(f"文件 {f}", exists, "存在" if exists else "缺失")

# 2. 配置
print("\n[2/7] 配置检查")
cfg = yaml.safe_load(open(os.path.join(PROJECT_ROOT, "config.yaml"), encoding="utf-8"))
check("测试API Key", bool(cfg.get("TEST_API_KEY")), "已配置")
check("正式API Key", bool(cfg.get("COMPETITION_API_KEY")), "已配置")
check("环境开关", "USE_TESTNET" in cfg, f"USE_TESTNET={cfg.get('USE_TESTNET')}")
check("风控参数", "RISK_CONFIG" in cfg, str(cfg.get("RISK_CONFIG", {})))

# 3. Binance universe数据
print("\n[3/7] Binance历史数据检查")
data_dir = os.path.join(PROJECT_ROOT, "data", "history_binance")
uf = os.path.join(data_dir, "universe.txt")
coins = [l.strip() for l in open(uf, encoding="utf-8") if l.strip()] if os.path.exists(uf) else []
check("universe列表", len(coins) >= 20, f"{len(coins)}个币种")
min_bars, bad, zero_vol = 10**9, [], []
for c in coins:
    fp = os.path.join(data_dir, f"{c}_1h.csv")
    if os.path.exists(fp):
        d = pd.read_csv(fp)
        n = len(d)
        min_bars = min(min_bars, n)
        if n < 1000:
            bad.append(c)
        if "volume" in d.columns and (d["volume"] <= 0).mean() > 0.1:
            zero_vol.append(c)
    else:
        bad.append(c)
check("历史K线完整性", len(bad) == 0, f"最少{min_bars}根, 异常{bad if bad else '无'}")
check("volume完整性", len(zero_vol) == 0, f"volume异常币{zero_vol if zero_vol else '无'}")

# 4. 模块导入
print("\n[4/7] 模块导入检查")
mods = [
    ("rooster_trader.client", "RoosterClient"),
    ("rooster_trader.execution", "ExecutionEngine"),
    ("rooster_trader.risk.manager", "RiskManager"),
    ("rooster_trader.strategy.low_vol_parity", "LowVolRiskParityStrategy"),
    ("rooster_trader.strategy.dynamic_position", "DynamicPositionStrategy"),
    ("rooster_trader.data.binance_feed", "fetch_recent"),
    ("rooster_trader.backtest.engine", "Backtester"),
]
for mod, cls in mods:
    try:
        m = __import__(mod, fromlist=[cls])
        check(f"导入 {mod}.{cls}", hasattr(m, cls), "成功")
    except Exception as e:
        check(f"导入 {mod}.{cls}", False, str(e)[:60])

# 5. 策略参数
print("\n[5/7] 策略参数检查")
symbols = [f"{c}/USD" for c in coins]
from rooster_trader.strategy.low_vol_parity import LowVolRiskParityStrategy
from rooster_trader.strategy.dynamic_position import DynamicPositionStrategy
lowvol = LowVolRiskParityStrategy(symbols=symbols, lookback_hours=168, top_n_pct=0.5,
                                  max_total_exposure=1.0)
check("波动率窗口", lowvol.lookback == 168, f"{lowvol.lookback//24}天")
check("选币比例", lowvol.top_n_pct == 0.5, "最低波50%")
check("总多头敞口", lowvol.max_total_exposure == 1.0, "100%")
strat = DynamicPositionStrategy(underlying_strategy=lowvol,
                                competition_start_date="2026-10-04", competition_days=14)
check("动态仓位", strat.competition_days == 14, "60%/100%/70%")

# 6. 信号冒烟
print("\n[6/7] 信号计算冒烟测试")
data = {}
for c in coins:
    data[f"{c}/USD"] = pd.read_csv(os.path.join(data_dir, f"{c}_1h.csv"),
                                   index_col=0, parse_dates=True)
common = None
for s, df in data.items():
    common = df.index if common is None else common.intersection(df.index)
data = {s: df.loc[common] for s, df in data.items()}
w = strat.on_bar(data)
active = {k: v for k, v in w.items() if v != 0}
longs = sum(v for v in w.values() if v > 0)
check("信号生成", longs > 0,
      f"多头总敞口{longs*100:.0f}%, 持仓{len(active)}币: {list(active.keys())[:8]}")

# 7. API连通性
print("\n[7/7] API连通性检查")
try:
    from rooster_trader.client import RoosterClient
    c = RoosterClient(cfg["TEST_API_KEY"], cfg["TEST_API_SECRET"])
    c.get_server_time()
    pairs = c.get_exchange_info()["TradePairs"]
    missing = [f"{x}/USD" for x in coins if f"{x}/USD" not in pairs]
    check("测试环境连通", True, "服务器响应正常")
    check("universe全部可交易", len(missing) == 0, f"缺失{missing if missing else '无'}")
except Exception as e:
    check("测试环境连通", False, str(e)[:80])

# 汇总
print("\n" + "=" * 70)
passed = sum(1 for _, ok, _ in results if ok)
total_checks = len(results)
print(f"检查结果: {passed}/{total_checks} 通过")
if passed == total_checks:
    print("✓ 全部检查通过，系统就绪")
else:
    print("✗ 存在未通过项，需修复:")
    for name, ok, detail in results:
        if not ok:
            print(f"    - {name}: {detail}")
print("=" * 70)
