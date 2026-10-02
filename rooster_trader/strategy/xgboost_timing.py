"""
XGBoost 机器学习择时策略（优化版 v2）
======================================
优化内容：
1. 减少过拟合：max_depth=3, n_estimators=100, 更强正则化
2. 禁用弱币种：XRP/USD样本外准确率过低，跳过
3. 增加跨币种特征：BTC动量/RSI/ATR/相关性（5个新特征，共18个）
4. 支持加载预训练模型：比赛时直接加载，不用现场训练
5. 严格防未来数据泄漏

核心逻辑：18个技术特征 → XGBoost分类器 → 预测未来12小时方向 → 信号过滤+仓位调整
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib

from rooster_trader.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models"
)

# 优化后的XGBoost参数（减少过拟合）
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "min_child_weight": 5,
    "random_state": 42,
    "eval_metric": "logloss",
    "verbosity": 0,
}

# 禁用ML预测的币种（样本外准确率<50%）
DISABLED_SYMBOLS = ["XRP/USD"]

# 跨币种参考币种
CROSS_REF_SYMBOL = "BTC/USD"


class XGBoostTimingStrategy(BaseStrategy):
    """
    XGBoost机器学习择时策略（优化版）。

    参数:
        symbols: 交易对列表
        forecast_hours: 预测未来多少小时的方向（默认12）
        feature_window: 特征计算的最大窗口（默认50）
        train_min_samples: 最小训练样本数（默认500）
        retrain_hours: 每隔多少小时重新训练（默认168=每周）
        long_threshold: 做多概率阈值（默认0.55）
        short_threshold: 做空概率阈值（默认0.55）
        max_positions: 最大持仓数（默认2）
        position_size: 单个仓位权重（默认0.15）
        load_pretrained: 是否加载预训练模型（默认True）
    """

    def __init__(
        self,
        symbols: list,
        forecast_hours: int = 12,
        feature_window: int = 50,
        train_min_samples: int = 500,
        retrain_hours: int = 168,
        long_threshold: float = 0.55,
        short_threshold: float = 0.55,
        max_positions: int = 2,
        position_size: float = 0.15,
        load_pretrained: bool = True,
    ):
        super().__init__(name="XGBoostTiming", symbols=symbols)
        self.forecast_hours = forecast_hours
        self.feature_window = feature_window
        self.train_min_samples = train_min_samples
        self.retrain_hours = retrain_hours
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.max_positions = max_positions
        self.position_size = position_size

        # 启用的币种（排除禁用的）
        self.enabled_symbols = [s for s in symbols if s not in DISABLED_SYMBOLS]
        self.disabled_symbols = [s for s in symbols if s in DISABLED_SYMBOLS]
        if self.disabled_symbols:
            logger.info("[XGBoost] 禁用ML预测的币种: %s（样本外准确率过低）", self.disabled_symbols)

        # 每个币种一个模型
        self.models: Dict[str, xgb.XGBClassifier] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.last_train_time: Dict[str, int] = {}
        self.feature_names: List[str] = []

        os.makedirs(MODEL_DIR, exist_ok=True)

        # 加载预训练模型
        if load_pretrained:
            self._load_pretrained_models()

    def _load_pretrained_models(self):
        """加载预训练模型（比赛时直接用，不用现场训练）"""
        meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
        if not os.path.exists(meta_path):
            logger.info("[XGBoost] 未找到预训练模型元数据，将在运行时训练")
            return

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])
            loaded = 0

            for symbol in self.enabled_symbols:
                coin = symbol.split("/")[0]
                model_path = os.path.join(MODEL_DIR, f"{coin}_xgb_model.json")
                scaler_path = os.path.join(MODEL_DIR, f"{coin}_scaler.pkl")

                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    model = xgb.XGBClassifier(**XGB_PARAMS)
                    model.load_model(model_path)
                    scaler = joblib.load(scaler_path)
                    self.models[symbol] = model
                    self.scalers[symbol] = scaler
                    self.last_train_time[symbol] = 999999  # 标记为已加载，不重新训练
                    loaded += 1

            logger.info("[XGBoost] 预训练模型加载完成: %d/%d 个币种", loaded, len(self.enabled_symbols))
        except Exception as e:
            logger.warning("[XGBoost] 加载预训练模型失败: %s，将在运行时训练", e)

    def _calculate_features(self, df: pd.DataFrame, btc_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        计算18个技术特征（13个单币种 + 5个跨币种）。
        【防泄漏】所有rolling计算只用t时刻及之前的数据。
        """
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # 1. RSI (14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # 2. MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # 3. Bollinger Bands (20, 2)
        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df["bb_position"] = (close - bb_sma) / (bb_std + 1e-10)
        df["bb_width"] = (bb_std * 2) / (bb_sma + 1e-10)

        # 4. ATR (14)
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / (close + 1e-10)

        # 5. 多周期动量
        df["mom_6h"] = close.pct_change(6)
        df["mom_24h"] = close.pct_change(24)
        df["mom_72h"] = close.pct_change(72)

        # 6. 成交量变化率（clip防止无穷大）
        df["vol_change"] = volume.pct_change(1).clip(-10, 10)
        df["vol_ratio"] = (volume / (volume.rolling(20).mean() + 1e-10)).clip(0, 100)

        # 7. 波动率
        df["volatility_24h"] = close.pct_change().rolling(24).std()

        # ---- 跨币种特征（5个）----
        if btc_df is not None:
            btc_aligned = btc_df.reindex(df.index)
            btc_close = btc_aligned["close"]

            df["btc_mom_6h"] = btc_close.pct_change(6)
            df["btc_mom_24h"] = btc_close.pct_change(24)

            btc_delta = btc_close.diff()
            btc_gain = btc_delta.where(btc_delta > 0, 0).rolling(14).mean()
            btc_loss = (-btc_delta.where(btc_delta < 0, 0)).rolling(14).mean()
            btc_rs = btc_gain / (btc_loss + 1e-10)
            df["btc_rsi"] = 100 - (100 / (1 + btc_rs))

            btc_hl = btc_aligned["high"] - btc_aligned["low"]
            btc_hc = np.abs(btc_aligned["high"] - btc_close.shift(1))
            btc_lc = np.abs(btc_aligned["low"] - btc_close.shift(1))
            btc_tr = pd.concat([btc_hl, btc_hc, btc_lc], axis=1).max(axis=1)
            btc_atr = btc_tr.rolling(14).mean()
            df["btc_atr_pct"] = btc_atr / (btc_close + 1e-10)

            coin_ret = close.pct_change()
            btc_ret = btc_close.pct_change()
            df["corr_btc_24h"] = coin_ret.rolling(24).corr(btc_ret)

        self.feature_names = [
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_position", "bb_width", "atr_pct",
            "mom_6h", "mom_24h", "mom_72h",
            "vol_change", "vol_ratio", "volatility_24h",
            "btc_mom_6h", "btc_mom_24h", "btc_rsi", "btc_atr_pct", "corr_btc_24h",
        ]

        return df

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """创建标签：未来forecast_hours收益率 > 0 为1，否则为0"""
        future_return = df["close"].shift(-self.forecast_hours) / df["close"] - 1
        return (future_return > 0).astype(int)

    def _prepare_dataset(self, df: pd.DataFrame, btc_df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据集"""
        df = self._calculate_features(df, btc_df)
        df["label"] = self._create_labels(df)
        df = df.dropna(subset=self.feature_names + ["label"])

        X = df[self.feature_names].values
        y = df["label"].values
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        return X, y

    def _train_model(self, symbol: str, df: pd.DataFrame, btc_df: pd.DataFrame = None) -> bool:
        """训练单个币种的XGBoost模型（使用优化后的参数）"""
        if len(df) < self.train_min_samples + self.feature_window:
            # 数据不足时只debug日志，不warning（避免刷屏）
            logger.debug("[XGBoost] %s 数据不足(%d)，跳过训练", symbol, len(df))
            return False

        X, y = self._prepare_dataset(df, btc_df)

        if len(np.unique(y)) < 2:
            logger.warning("[XGBoost] %s 标签单一，跳过训练", symbol)
            return False

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_scaled, y)

        self.models[symbol] = model
        self.scalers[symbol] = scaler
        self.last_train_time[symbol] = int(len(df))

        coin = symbol.split("/")[0]
        model_path = os.path.join(MODEL_DIR, f"{coin}_xgb_model.json")
        scaler_path = os.path.join(MODEL_DIR, f"{coin}_scaler.pkl")
        model.save_model(model_path)
        joblib.dump(scaler, scaler_path)

        train_acc = model.score(X_scaled, y)
        logger.info("[XGBoost] %s 训练完成, 样本%d, 训练准确率%.1f%%",
                    symbol, len(y), train_acc * 100)
        return True

    def _predict(self, symbol: str, df: pd.DataFrame, btc_df: pd.DataFrame = None) -> Optional[float]:
        """预测当前K线的做多概率"""
        if symbol not in self.models:
            return None

        df = self._calculate_features(df, btc_df)
        latest = df.iloc[-1]

        if any(pd.isna(latest[f]) for f in self.feature_names):
            return None

        X = latest[self.feature_names].values.reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        X_scaled = self.scalers[symbol].transform(X)
        prob = self.models[symbol].predict_proba(X_scaled)[0]

        return float(prob[1])

    def on_bar(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        每根K线调用，返回目标仓位权重。
        只对enabled_symbols进行预测，禁用的币种返回0。
        """
        target_weights = {s: 0.0 for s in self.symbols}
        predictions = []

        btc_df = data.get(CROSS_REF_SYMBOL)

        for symbol in self.enabled_symbols:
            if symbol not in data or len(data[symbol]) < self.feature_window + 10:
                continue

            df = data[symbol]

            # 检查是否需要训练/重新训练（已加载预训练模型的跳过）
            need_train = (
                symbol not in self.models
                or (self.last_train_time.get(symbol, 0) < 999999
                    and (len(df) - self.last_train_time.get(symbol, 0)) > self.retrain_hours)
            )

            if need_train and len(df) >= self.train_min_samples:
                self._train_model(symbol, df, btc_df)

            # 预测
            prob_long = self._predict(symbol, df, btc_df)
            if prob_long is None:
                continue

            if prob_long >= self.long_threshold:
                predictions.append({"symbol": symbol, "side": "LONG", "strength": prob_long})
            elif prob_long <= (1 - self.short_threshold):
                predictions.append({"symbol": symbol, "side": "SHORT", "strength": 1 - prob_long})

        predictions.sort(key=lambda x: x["strength"], reverse=True)
        selected = predictions[: self.max_positions]

        for pred in selected:
            symbol = pred["symbol"]
            weight = self.position_size if pred["side"] == "LONG" else -self.position_size
            target_weights[symbol] = weight
            logger.info("[XGBoost] %s %s (概率%.1f%%)",
                        pred["side"], symbol, pred["strength"] * 100)

        return target_weights

    def get_signal_filter(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        作为信号过滤器使用：返回每个币种的ML预测方向强度。
        正值=看多，负值=看空，绝对值=置信度。
        禁用的币种返回0（不过滤）。
        """
        filters = {}
        btc_df = data.get(CROSS_REF_SYMBOL)

        for symbol in self.symbols:
            if symbol in self.disabled_symbols:
                filters[symbol] = 0.0  # 禁用的币种不过滤
                continue
            if symbol not in self.models or symbol not in data:
                filters[symbol] = 0.0
                continue
            prob_long = self._predict(symbol, data[symbol], btc_df)
            if prob_long is None:
                filters[symbol] = 0.0
            else:
                filters[symbol] = (prob_long - 0.5) * 2
        return filters
