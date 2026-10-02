"""
实盘主程序入口（金奖版 v5.1 防护修补版）
========================================
在 v5 时序动量基础上修补：
  1. 删除调仓块末尾 time.sleep(3600)（原会阻塞止损/风控1小时）
  2. USE_TESTNET 改从 config.yaml 读取（不再硬编码）
  3. risk_state.json 持久化日内风控基准（重启不丢失UTC基准）
  4. 首次启动即时建仓（现金占比>90%立即建，不再傻等%72）
  5. 持仓存在但 entries 缺失时自动补全开仓价（止损不裸奔）
  6. JSON 全部上下文管理器；读CSV统一UTC时区

最终策略：25币种30天时序动量（只做多、无趋势空仓）+ 动态仓位60/100/70
         + 单币10%止损 + 日内4%/8%回撤熔断
"""

import os
import sys
import json
import time
import yaml
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rooster_trader.client import RoosterClient
from rooster_trader.execution import ExecutionEngine
from rooster_trader.risk.manager import RiskManager, RiskConfig
from rooster_trader.strategy.trend import TrendFollowingStrategy
from rooster_trader.strategy.dynamic_position import DynamicPositionStrategy
from rooster_trader.data import binance_feed
from rooster_trader.utils.logger import setup_logger

logger = setup_logger("main")

# ---- 常量 ----
COMPETITION_START = "2026-10-04"
COMPETITION_DAYS = 14
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLD_START_DIR = os.path.join(BASE_DIR, "data", "history_binance")
LIVE_DIR = os.path.join(BASE_DIR, "data", "live")
ENTRY_FILE = os.path.join(LIVE_DIR, "entry_prices.json")
RISK_STATE_FILE = os.path.join(LIVE_DIR, "risk_state.json")
STOP_LOSS_PCT = 0.10
STOP_CHECK_INTERVAL = 600


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_universe() -> list:
    uf = os.path.join(COLD_START_DIR, "universe.txt")
    with open(uf, encoding="utf-8") as f:
        coins = [l.strip() for l in f if l.strip()]
    return [f"{c}/USD" for c in coins]


def _read_csv_utc(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def cold_start_data(symbols: list):
    """复制 Binance 120天数据到 live 目录作为序列起点"""
    os.makedirs(LIVE_DIR, exist_ok=True)
    for pair in symbols:
        coin = pair.split("/")[0]
        src = os.path.join(COLD_START_DIR, f"{coin}_1h.csv")
        dst = os.path.join(LIVE_DIR, f"{coin}_1h.csv")
        if os.path.exists(src):
            df = _read_csv_utc(src)
            df.to_csv(dst)
            logger.info("冷启动 %-8s %d根K线", pair, len(df))
        else:
            logger.warning("缺少冷启动数据: %s", pair)


def refresh_data(symbols: list) -> dict:
    """每小时用 Binance 公共K线增量更新，返回 {symbol: DataFrame}"""
    data = {}
    for pair in symbols:
        coin = pair.split("/")[0]
        try:
            data[pair] = binance_feed.update_live_csv(coin, LIVE_DIR, pull_bars=48)
        except Exception as e:
            logger.warning("刷新 %s 失败: %s，沿用本地", pair, e)
            fp = os.path.join(LIVE_DIR, f"{coin}_1h.csv")
            if os.path.exists(fp):
                data[pair] = _read_csv_utc(fp)
    return data


# ---- 开仓价持久化 ----
def load_entries() -> dict:
    if os.path.exists(ENTRY_FILE):
        try:
            with open(ENTRY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _atomic_save_json(filepath: str, data: dict):
    """原子写入：先写 .tmp 再 os.replace 覆盖，防止写入中断损坏JSON"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp = f"{filepath}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, filepath)


def save_entries(entries: dict):
    _atomic_save_json(ENTRY_FILE, entries)


# ---- 风控状态持久化 ----
def load_risk_state() -> dict:
    if os.path.exists(RISK_STATE_FILE):
        try:
            with open(RISK_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_risk_state(daily_start_val: float, last_day: int):
    _atomic_save_json(RISK_STATE_FILE, {"daily_start_value": daily_start_val,
                                        "last_calendar_day": last_day})


def filter_tradeable(client: RoosterClient, symbols: list, trend, data: dict):
    """行情过滤：只在Roostoo当前有报价币种中选币，返回(active_symbols, data)"""
    try:
        live_pairs = set(client.get_ticker().get("Data", {}).keys())
        active_symbols = [s for s in symbols if s in live_pairs]
        trend.symbols = active_symbols
        data = {s: df for s, df in data.items() if s in active_symbols}
        logger.info("有行情币种: %d/%d", len(active_symbols), len(symbols))
        return active_symbols, data
    except Exception as e:
        logger.warning("行情过滤失败，沿用完整universe: %s", e)
        return symbols, data


def record_entries_after_rebalance(client: RoosterClient, entries: dict):
    """再平衡后：新增现货记录开仓价，消失的删除"""
    bal = client.get_balance().get("SpotWallet", {})
    ticker = client.get_ticker().get("Data", {})
    held = set()
    for coin, info in bal.items():
        if coin == "USD":
            continue
        qty = float(info.get("Free", 0)) + float(info.get("Lock", 0))
        if qty > 0:
            held.add(coin)
            if coin not in entries:
                cell = ticker.get(f"{coin}/USD", {})
                px = float(cell.get("LastPrice", 0))
                if px > 0:
                    entries[coin] = px
    for coin in list(entries.keys()):
        if coin not in held:
            entries.pop(coin)
    save_entries(entries)


def check_intraday_stop(client: RoosterClient, executor: ExecutionEngine,
                        entries: dict) -> list:
    """单标的相对开仓价跌≥10%立即平仓；持仓无记录则补全开仓价"""
    stopped = []
    bal = client.get_balance().get("SpotWallet", {})
    ticker = client.get_ticker().get("Data", {})
    for coin, info in bal.items():
        if coin == "USD":
            continue
        qty = float(info.get("Free", 0))
        if qty <= 0:
            continue
        pair = f"{coin}/USD"
        if pair not in ticker:
            continue
        cur = float(ticker[pair]["LastPrice"])

        if coin not in entries:
            # 重启后补全：以当前价为开仓价（此后跌10%才止损）
            entries[coin] = cur
            save_entries(entries)
            continue

        entry = entries[coin]
        if cur / entry - 1 <= -STOP_LOSS_PCT:
            try:
                q = executor.adjust_quantity(pair, qty)
                if q > 0:
                    r = client.place_order(pair=pair, side="SELL", quantity=q,
                                           order_type="MARKET")
                    if r.get("Success"):
                        logger.warning("止损平仓 %s（%.4f→%.4f, %.1f%%）",
                                       coin, entry, cur, (cur/entry-1)*100)
                        entries.pop(coin, None)
                        stopped.append(coin)
                        time.sleep(0.5)
            except Exception as e:
                logger.error("止损失败 %s: %s", pair, e)
    save_entries(entries)
    return stopped


def main():
    logger.info("=" * 60)
    logger.info("APAC Quant Hackathon Bot 启动（v5.1 防护修补版）")
    logger.info("启动时间: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    config = load_config()

    # ---- 环境开关从 config 读取（开赛改 USE_TESTNET: false）----
    use_testnet = config.get("USE_TESTNET", True)
    if use_testnet:
        client = RoosterClient(config["TEST_API_KEY"], config["TEST_API_SECRET"])
        logger.info("使用【测试环境】API")
    else:
        client = RoosterClient(config["COMPETITION_API_KEY"], config["COMPETITION_API_SECRET"])
        logger.info("使用【正式比赛】API")

    try:
        client.get_server_time()
        logger.info("API 连通性验证成功")
    except Exception as e:
        logger.critical("API 连通失败: %s", e)
        sys.exit(1)

    symbols = load_universe()
    logger.info("交易universe: %d个币种", len(symbols))

    trend = TrendFollowingStrategy(
        symbols=symbols,
        lookback_hours=720,
        ma_hours=720,
        use_ma_filter=False,
        max_total_exposure=1.0,
        rebalance_hours=72,
    )
    strategy = DynamicPositionStrategy(
        underlying_strategy=trend,
        competition_start_date=COMPETITION_START,
        competition_days=COMPETITION_DAYS,
    )
    logger.info("策略: 25币种30天时序动量（只做多、无趋势空仓）+ 动态仓位")

    logger.info("加载冷启动历史数据...")
    cold_start_data(symbols)

    risk_mgr = RiskManager(RiskConfig(**config["RISK_CONFIG"]))
    executor = ExecutionEngine(client=client, min_trade_pct=0.005)
    entries = load_entries()

    # ---- 风控状态恢复（重启不丢失今日UTC基准）----
    today = datetime.now(timezone.utc).day
    risk_state = load_risk_state()
    if risk_state.get("last_calendar_day") == today:
        risk_mgr.daily_start_value = risk_state["daily_start_value"]
        logger.info("恢复今日风控基准市值: %.2f", risk_mgr.daily_start_value)
    else:
        pv0 = client.get_portfolio_value()
        risk_mgr.daily_start_value = pv0
        risk_mgr.peak_value = pv0
        save_risk_state(pv0, today)
        logger.info("初始化今日风控基准市值: %.2f", pv0)

    last_data_refresh = 0
    last_rebalance_hour = -1
    last_calendar_day = today
    last_stop_check = 0
    last_hourly_risk = 0

    # ---- 首次启动即时建仓（现金占比>90%立即建，无趋势则on_bar返回全0安全空仓）----
    bal = client.get_balance().get("SpotWallet", {})
    usd_free = float(bal.get("USD", {}).get("Free", 0))
    pv_now = client.get_portfolio_value()
    if pv_now > 0 and usd_free / pv_now > 0.90:
        logger.info("开局现金占比 %.0f%%，立即首次建仓...", usd_free / pv_now * 100)
        data = refresh_data(symbols)
        _, data = filter_tradeable(client, symbols, trend, data)
        target = strategy.on_bar(data)
        executor.rebalance(target)
        record_entries_after_rebalance(client, entries)
        last_rebalance_hour = int(time.time() / 3600)
        logger.info("首次建仓完成，进入常规72h轮询")

    # ---- 主循环 ----
    while True:
        try:
            now = datetime.now(timezone.utc)

            # 1. 每小时 Binance 增量
            if time.time() - last_data_refresh > 3600:
                data = refresh_data(symbols)
                last_data_refresh = time.time()
                logger.info("数据刷新完成，%d币种", len(data))

            # 2. UTC日期翻转：重置当日基准并持久化
            if now.day != last_calendar_day:
                pv_open = client.get_portfolio_value()
                risk_mgr.reset_daily(pv_open)
                last_calendar_day = now.day
                save_risk_state(pv_open, now.day)
                logger.info("新交易日，风控基准市值 %.2f", pv_open)

            # 3. 周期内止损（每10分钟）
            if time.time() - last_stop_check > STOP_CHECK_INTERVAL:
                check_intraday_stop(client, executor, entries)
                last_stop_check = time.time()

            # 4. 每小时日内风控
            if time.time() - last_hourly_risk > 3600:
                pv = client.get_portfolio_value()
                action = risk_mgr.check_daily_risk(pv)
                if action == "FLATTEN":
                    logger.critical("日内回撤触发硬止损，全平暂停1小时")
                    executor.flatten_all()
                    entries.clear()
                    save_entries(entries)
                    base = client.get_portfolio_value()
                    risk_mgr.reset_daily(base)
                    save_risk_state(base, now.day)
                    # 小步长等待60分钟，可随时响应中断（不一次性阻塞1小时）
                    for _ in range(60):
                        time.sleep(60)
                elif action == "REDUCE":
                    logger.warning("日内回撤达减仓阈值，预警")
                last_hourly_risk = time.time()

            # 5. 每72小时再平衡
            hours_since_epoch = int(time.time() / 3600)
            should_rebalance = (hours_since_epoch % 72 == 0) and \
                               (hours_since_epoch != last_rebalance_hour)

            if should_rebalance:
                logger.info("-" * 50)
                logger.info("触发再平衡 %s", now.strftime("%Y-%m-%d %H:%M"))
                data = refresh_data(symbols)
                _, data = filter_tradeable(client, symbols, trend, data)

                pos_info = strategy.get_current_position_info()
                logger.info("阶段: %s 第%d天, 仓位系数 %.0f%%",
                            pos_info["phase"], pos_info["days_elapsed"],
                            pos_info["base_position_ratio"] * 100)

                target = strategy.on_bar(data)
                active = {k: f"{v:.1%}" for k, v in target.items() if v != 0}
                logger.info("目标仓位: %s", active)

                result = executor.rebalance(target)
                logger.info("执行: 现货成功%d, 失败%d",
                            result["orders_executed"], result["orders_failed"])

                record_entries_after_rebalance(client, entries)
                logger.info("调仓后市值: %.2f", client.get_portfolio_value())
                last_rebalance_hour = hours_since_epoch
                # 不再 sleep(3600)：调度交回主循环，止损/风控保持运行

            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("收到中断，安全退出")
            break
        except Exception as e:
            logger.exception("主循环异常: %s", e)
            time.sleep(30)

    logger.info("Bot 已停止")


if __name__ == "__main__":
    main()
