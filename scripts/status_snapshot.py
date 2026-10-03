"""实时状态快照：读取config（自动判断测试/正式环境），打印组合市值、持仓与占比。"""
import os, sys, yaml

BASE = os.path.expanduser("~/roostoo-quant-hackathon")
sys.path.insert(0, BASE)
from rooster_trader.client import RoosterClient

cfg = yaml.safe_load(open(os.path.join(BASE, "config.yaml")))
if cfg.get("USE_TESTNET", True):
    cl = RoosterClient(cfg["TEST_API_KEY"], cfg["TEST_API_SECRET"])
    tag = "测试环境"
else:
    cl = RoosterClient(cfg["COMPETITION_API_KEY"], cfg["COMPETITION_API_SECRET"])
    tag = "正式比赛"

pv = cl.get_portfolio_value()
tk = cl.get_ticker().get("Data", {})
b = cl.get_balance().get("SpotWallet", {})

print(f"[{tag}] 组合市值: {pv:,.2f} USD")
print("-" * 52)
coin_val = 0.0
for coin, v in b.items():
    free = float(v.get("Free", 0))
    if coin == "USD":
        print(f"  现金 USD   {free:>12,.2f}   ({free/pv*100:5.1f}%)")
    elif free > 0:
        px = float(tk.get(f"{coin}/USD", {}).get("LastPrice", 0))
        val = free * px
        coin_val += val
        print(f"  {coin:<9} {free:>12.4f}   市值 {val:>9,.0f}  ({val/pv*100:5.1f}%)")
print("-" * 52)
print(f"  持仓总市值 {coin_val:,.0f}  仓位 {coin_val/pv*100:.1f}%   现金 {(pv-coin_val)/pv*100:.1f}%")
