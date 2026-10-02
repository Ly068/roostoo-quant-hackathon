"""
低波动异象 + 风险平价策略（Low-Volatility Anomaly / Risk Parity）
=============================================================
学术依据：低波动异象（Low Volatility Anomaly / BAB, Frazzini-Pedersen 2014）——
经风险调整后，低波动资产往往优于高波动资产；高波动山寨币的"暴涨"常以更大回撤收场。

流程：
  1. Parkinson 极差波动率（用 high/low，比收盘价标准差更高效），并做：
     - ffill 补缺失、H/L 与 O/C 对齐（防 H=L 死水、插针错乱）
     - Winsorization：单小时极差 clip 到 15% 内（防单根插针污染方差）
  2. 趋势过滤：收盘价 > lookback 均线（不接单边下跌的死水币）
  3. 选波动率最低的 top_n_pct，按波动率倒数加权（Risk Parity）
  4. 缓存 latest_volatilities 供动态止损
"""
import numpy as np
import pandas as pd
from typing import Dict

from rooster_trader.strategy.base import BaseStrategy


class LowVolRiskParityStrategy(BaseStrategy):
    def __init__(
        self,
        symbols: list,
        lookback_hours: int = 72,
        top_n_pct: float = 0.5,
        max_total_exposure: float = 1.0,
    ):
        super().__init__(name="LowVolParity", symbols=symbols)
        self.lookback = lookback_hours
        self.top_n_pct = top_n_pct
        self.max_total_exposure = max_total_exposure
        self.latest_volatilities: Dict[str, float] = {}

    def warmup_bars(self) -> int:
        return self.lookback + 5

    def _parkinson(self, df: pd.DataFrame) -> float:
        """清洗假数据并计算 Parkinson 每小时极差波动率"""
        if df is None or len(df) < self.lookback:
            return np.inf
        w = df.tail(self.lookback).copy().ffill()

        # H/L 与 open/close 对齐，防 H=L（死水）或插针导致高低价错乱
        H = w[["high", "open", "close"]].max(axis=1)
        L = w[["low", "open", "close"]].min(axis=1)
        hl = H / np.maximum(L, 1e-9)

        # Winsorization：单小时极差限制在 1.0001~1.15，防单根插针污染
        hl = np.clip(hl, 1.0001, 1.15)

        # Parkinson: sqrt(1/(4 ln2) * mean(ln(H/L)^2))，0.36067=1/(4ln2)
        return float(np.sqrt(0.36067 * (np.log(hl) ** 2).mean()))

    def on_bar(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        vols = {}
        for pair in self.symbols:
            df = data.get(pair)
            if df is None or len(df) < self.lookback + 2:
                continue
            vol = self._parkinson(df)
            ma = df["close"].rolling(self.lookback).mean().iloc[-1]
            if np.isfinite(vol) and df["close"].iloc[-1] > ma:
                vols[pair] = vol

        self.latest_volatilities = vols
        weights = {s: 0.0 for s in self.symbols}

        if vols:
            ordered = sorted(vols, key=vols.get)          # 波动率升序
            k = max(1, int(len(ordered) * self.top_n_pct))
            selected = ordered[:k]
            inv = {p: 1.0 / vols[p] for p in selected}   # 波动率倒数
            total = sum(inv.values())
            for p in selected:
                weights[p] = inv[p] / total * self.max_total_exposure

        return weights
