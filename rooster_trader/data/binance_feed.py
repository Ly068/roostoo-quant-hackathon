"""
Binance 公开K线数据 Feed
=======================
数据源: https://data-api.binance.vision （Binance公共数据，无需Key）
提供：
  - fetch_recent(coin, limit)：最近N根1h K线
  - fetch_range(coin, start_ms, end_ms)：区间K线（自动分页）
  - update_live_csv(coin, live_dir, pull_bars)：增量更新本地CSV并返回DataFrame

价格/成交量完整真实，用于：
  - main.py 运行时增量刷新（替代有volume缺陷的 yfinance）
  - 冷启动历史数据由 scripts/download_binance.py 预下载
"""
import os
import time

import pandas as pd
import requests

BINANCE_BASE = "https://data-api.binance.vision"
INTERVAL = "1h"
HOUR_MS = 3600_000


def _request_klines(symbol, start_ms=None, limit=None):
    params = {"symbol": symbol, "interval": INTERVAL}
    if start_ms is not None:
        params["startTime"] = start_ms
    if limit is not None:
        params["limit"] = limit
    r = requests.get(BINANCE_BASE + "/api/v3/klines", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _rows_to_frame(rows):
    df = pd.DataFrame(rows, columns=[
        "openTime", "open", "high", "low", "close", "volume",
        "closeTime", "quoteVolume", "trades", "takerBase", "takerQuote", "ignore",
    ])
    df = df.drop_duplicates(subset="openTime")
    df["time"] = pd.to_datetime(df["openTime"], unit="ms", utc=True)
    df = df.set_index("time")[["open", "high", "low", "close", "volume"]]
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df


def fetch_recent(coin: str, limit: int = 48) -> pd.DataFrame:
    """获取某币种最近 limit 根1h K线"""
    rows = _request_klines(coin + "USDT", limit=limit)
    return _rows_to_frame(rows)


def fetch_range(coin: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """分页获取区间K线"""
    rows = []
    cur = start_ms
    while cur < end_ms:
        batch = _request_klines(coin + "USDT", start_ms=cur, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + HOUR_MS
        if len(batch) < 1000:
            break
        time.sleep(0.1)
    return _rows_to_frame(rows)


def update_live_csv(coin: str, live_dir: str, pull_bars: int = 48) -> pd.DataFrame:
    """
    增量更新 live 目录下 <coin>_1h.csv：
    拉取最近 pull_bars 根，与本地合并去重（保留最新），写回。
    返回合并后的完整 DataFrame。
    """
    fp = os.path.join(live_dir, f"{coin}_1h.csv")
    base = pd.read_csv(fp, index_col=0, parse_dates=True) if os.path.exists(fp) else None
    # CSV读回为naive，而new为UTC aware，统一为UTC避免concat比较报错
    if base is not None:
        base.index = pd.to_datetime(base.index, utc=True)
    new = fetch_recent(coin, pull_bars)
    if base is not None:
        merged = pd.concat([base, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = new
    os.makedirs(live_dir, exist_ok=True)
    merged.to_csv(fp)
    return merged
