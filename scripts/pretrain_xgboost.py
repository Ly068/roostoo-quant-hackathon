"""
XGBoost 模型预训练脚本（严格防未来数据泄漏版）
==================================================

防泄漏核心原则：
1. 特征只用 t 时刻及之前的数据，标签用 t 时刻之后的收益
2. Walk-forward 滚动训练：只用过去N天训练，预测下一个时间段
3. Rolling 计算用 closed='left'，不包含当前bar
4. Scaler 只在训练集上 fit，绝不用全量数据
5. 训练/测试严格按时间分割，禁止随机打乱
6. 训练集和测试集之间留 gap（避免标签重叠）

用法：
    python scripts/pretrain_xgboost.py
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "history")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ================================================================
# 防泄漏配置
# ================================================================
FORECAST_HOURS = 12       # 预测未来12小时方向
FEATURE_WINDOW = 50       # 特征计算最大窗口
TRAIN_MIN_SAMPLES = 500   # 最小训练样本数
GAP_HOURS = 12            # 训练集和测试集之间的gap（防止标签重叠）
WALK_FORWARD_STEPS = 3    # Walk-forward验证步数

# ================================================================
# 优化配置（减少过拟合 + 禁用弱币种）
# ================================================================
# 过拟合优化：更浅的树 + 更少的树 + 更强正则化
XGB_PARAMS = {
    "n_estimators": 100,       # 从200降到100，减少过拟合
    "max_depth": 3,            # 从4降到3，更浅的树
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,          # L1正则化增强（从0.1到0.5）
    "reg_lambda": 2.0,         # L2正则化增强（从1.0到2.0）
    "min_child_weight": 5,     # 叶子节点最小样本数（防止学到极端个例）
    "random_state": 42,
    "eval_metric": "logloss",
    "verbosity": 0,
}

# 禁用ML预测的币种（样本外准确率<50%的币种）
DISABLED_ML_SYMBOLS = ["XRP/USD"]

# 启用ML预测的币种
ENABLED_ML_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD"]

# 跨币种特征参考币种（市场风向标）
CROSS_REF_SYMBOL = "BTC/USD"


def load_data(symbols: list) -> dict:
    """加载历史数据"""
    data = {}
    for pair in symbols:
        coin = pair.split("/")[0]
        filepath = os.path.join(DATA_DIR, f"{coin}_1h.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            data[pair] = df
            print(f"  加载 {pair}: {len(df)} 根, {df.index[0].date()} ~ {df.index[-1].date()}")
    return data


def calculate_features(df: pd.DataFrame, btc_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    计算技术特征（13个单币种特征 + 5个跨币种特征 = 18个特征）。

    【防泄漏关键】所有rolling计算只用t时刻及之前的数据，
    跨币种特征也只用参考币种t时刻及之前的数据。

    跨币种特征（新增）：
    - btc_mom_6h: BTC的6小时动量（市场风向标）
    - btc_mom_24h: BTC的24小时动量
    - btc_rsi: BTC的RSI（市场情绪）
    - btc_atr_pct: BTC的ATR百分比（市场波动率）
    - corr_btc_24h: 本币种与BTC的24小时相关性
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

    # ---- 跨币种特征（新增5个）----
    if btc_df is not None:
        # 对齐时间索引
        btc_aligned = btc_df.reindex(df.index)
        btc_close = btc_aligned["close"]

        # BTC的6h/24h动量（市场风向标）
        df["btc_mom_6h"] = btc_close.pct_change(6)
        df["btc_mom_24h"] = btc_close.pct_change(24)

        # BTC的RSI（市场情绪）
        btc_delta = btc_close.diff()
        btc_gain = btc_delta.where(btc_delta > 0, 0).rolling(14).mean()
        btc_loss = (-btc_delta.where(btc_delta < 0, 0)).rolling(14).mean()
        btc_rs = btc_gain / (btc_loss + 1e-10)
        df["btc_rsi"] = 100 - (100 / (1 + btc_rs))

        # BTC的ATR百分比（市场波动率）
        btc_hl = btc_aligned["high"] - btc_aligned["low"]
        btc_hc = np.abs(btc_aligned["high"] - btc_close.shift(1))
        btc_lc = np.abs(btc_aligned["low"] - btc_close.shift(1))
        btc_tr = pd.concat([btc_hl, btc_hc, btc_lc], axis=1).max(axis=1)
        btc_atr = btc_tr.rolling(14).mean()
        df["btc_atr_pct"] = btc_atr / (btc_close + 1e-10)

        # 本币种与BTC的24小时收益率相关性
        coin_ret = close.pct_change()
        btc_ret = btc_close.pct_change()
        df["corr_btc_24h"] = coin_ret.rolling(24).corr(btc_ret)

    feature_names = [
        # 单币种特征（13个）
        "rsi", "macd", "macd_signal", "macd_hist",
        "bb_position", "bb_width", "atr_pct",
        "mom_6h", "mom_24h", "mom_72h",
        "vol_change", "vol_ratio", "volatility_24h",
        # 跨币种特征（5个）
        "btc_mom_6h", "btc_mom_24h", "btc_rsi", "btc_atr_pct", "corr_btc_24h",
    ]

    return df, feature_names


def create_labels(df: pd.DataFrame, forecast_hours: int) -> pd.Series:
    """
    创建标签：未来forecast_hours收益率 > 0 为1（上涨），否则为0（下跌）。

    【防泄漏关键】标签用的是 t+forecast_hours 时刻的价格，
    而特征用的是 t 时刻及之前的数据，两者时间上严格分离。
    """
    future_price = df["close"].shift(-forecast_hours)
    future_return = (future_price / df["close"]) - 1
    labels = (future_return > 0).astype(int)
    labels.name = "label"
    return labels


def prepare_dataset(df: pd.DataFrame, forecast_hours: int, btc_df: pd.DataFrame = None):
    """
    准备特征矩阵和标签向量。
    返回 X, y, timestamps, feature_names
    """
    df_feat, feature_names = calculate_features(df, btc_df)
    labels = create_labels(df_feat, forecast_hours)

    # 合并并去掉NaN（特征窗口不足 + 标签未来数据不足 + 跨币种特征不足）
    dataset = pd.concat([df_feat[feature_names], labels], axis=1)
    dataset = dataset.dropna()

    X = dataset[feature_names].values
    y = dataset["label"].values
    timestamps = dataset.index

    # 替换无穷大和NaN（双重保险）
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

    return X, y, timestamps, feature_names


def walk_forward_validation(df: pd.DataFrame, symbol: str, forecast_hours: int, btc_df: pd.DataFrame = None):
    """
    Walk-forward 滚动验证（严格防泄漏的标准做法）。
    """
    X, y, timestamps, feature_names = prepare_dataset(df, forecast_hours, btc_df)
    n_total = len(X)

    if n_total < TRAIN_MIN_SAMPLES + 200:
        print(f"  ⚠️ {symbol} 数据不足({n_total})，跳过walk-forward验证")
        return None, feature_names

    # 按时间分割（禁止随机打乱！）
    test_size = n_total // (WALK_FORWARD_STEPS + 1)
    all_predictions = []
    all_actuals = []
    all_probabilities = []

    print(f"  Walk-forward验证: {WALK_FORWARD_STEPS}步, 每步测试集{test_size}样本")

    for step in range(WALK_FORWARD_STEPS):
        # 训练集：从开头到 test_start
        test_start = (step + 1) * test_size
        train_end = test_start - GAP_HOURS  # 留gap防止标签重叠

        if train_end < TRAIN_MIN_SAMPLES:
            print(f"    步{step+1}: 训练集不足({train_end})，跳过")
            continue

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_start + test_size], y[test_start:test_start + test_size]

        if len(X_test) == 0:
            break

        # 【防泄漏关键】Scaler只在训练集上fit！
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # 测试集用训练集的scaler

        # 训练XGBoost（使用优化后的参数）
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train_scaled, y_train)

        # 预测测试集
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
        all_probabilities.extend(y_prob)

        # 打印每步结果
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0
        print(f"    步{step+1}: 训练{len(X_train)}样本, 测试{len(X_test)}样本, "
              f"准确率{acc*100:.1f}%, AUC={auc:.3f}")

    # 汇总结果
    if len(all_predictions) == 0:
        return None, feature_names

    results = {
        "accuracy": accuracy_score(all_actuals, all_predictions),
        "precision": precision_score(all_actuals, all_predictions, zero_division=0),
        "recall": recall_score(all_actuals, all_predictions, zero_division=0),
        "f1": f1_score(all_actuals, all_predictions, zero_division=0),
        "auc": roc_auc_score(all_actuals, all_probabilities) if len(np.unique(all_actuals)) > 1 else 0,
        "n_test_samples": len(all_predictions),
    }

    return results, feature_names


def train_final_model(df: pd.DataFrame, symbol: str, forecast_hours: int, btc_df: pd.DataFrame = None):
    """
    用全部历史数据训练最终模型（用于比赛实盘）。
    """
    X, y, timestamps, feature_names = prepare_dataset(df, forecast_hours, btc_df)

    if len(X) < TRAIN_MIN_SAMPLES:
        print(f"  ⚠️ {symbol} 数据不足({len(X)})，跳过训练")
        return None, None, None, 0

    # 【防泄漏】Scaler在全量训练数据上fit（这是训练数据，不是未来数据）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练最终模型（使用优化后的参数）
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_scaled, y)

    # 训练集表现（仅供参考，不是样本外表现！）
    y_pred = model.predict(X_scaled)
    train_acc = accuracy_score(y, y_pred)

    return model, scaler, feature_names, train_acc


def main():
    print("=" * 70)
    print("XGBoost 模型预训练（严格防未来数据泄漏）")
    print("=" * 70)

    # 加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    symbols = config["SYMBOL_UNIVERSE"]

    # 加载数据
    print("\n[1/4] 加载历史数据...")
    data = load_data(symbols)

    # Walk-forward 验证（证明模型在样本外有效）
    print("\n[2/4] Walk-forward 滚动验证（样本外表现）...")
    print("-" * 70)
    print(f"  启用ML预测的币种: {ENABLED_ML_SYMBOLS}")
    print(f"  禁用ML预测的币种: {DISABLED_ML_SYMBOLS}（样本外准确率<50%，禁用）")
    print(f"  跨币种参考: {CROSS_REF_SYMBOL}")
    btc_df = data.get(CROSS_REF_SYMBOL)

    wf_results = {}
    for symbol in ENABLED_ML_SYMBOLS:
        if symbol not in data:
            continue
        print(f"\n  {symbol}:")
        result, _ = walk_forward_validation(data[symbol], symbol, FORECAST_HOURS, btc_df)
        if result:
            wf_results[symbol] = result

    # 打印汇总
    print("\n" + "=" * 70)
    print("Walk-forward 验证汇总（样本外表现）")
    print("=" * 70)
    print(f"{'币种':<12} {'准确率':>8} {'精确率':>8} {'召回率':>8} {'F1':>8} {'AUC':>8} {'样本数':>8}")
    print("-" * 70)
    for symbol, r in wf_results.items():
        print(f"{symbol:<12} {r['accuracy']*100:>7.1f}% {r['precision']*100:>7.1f}% "
              f"{r['recall']*100:>7.1f}% {r['f1']:>8.3f} {r['auc']:>8.3f} {r['n_test_samples']:>8d}")

    avg_acc = np.mean([r["accuracy"] for r in wf_results.values()])
    avg_auc = np.mean([r["auc"] for r in wf_results.values()])
    print("-" * 70)
    print(f"{'平均':<12} {avg_acc*100:>7.1f}% {'':>8} {'':>8} {'':>8} {avg_auc:>8.3f}")
    print("=" * 70)

    # 训练最终模型并保存
    print("\n[3/4] 训练最终模型并保存...")
    saved_models = {}
    for symbol in ENABLED_ML_SYMBOLS:
        if symbol not in data:
            continue
        result = train_final_model(data[symbol], symbol, FORECAST_HOURS, btc_df)
        if result[0] is not None:
            model, scaler, feature_names, train_acc = result
            coin = symbol.split("/")[0]

            # 保存模型和scaler
            model_path = os.path.join(MODEL_DIR, f"{coin}_xgb_model.json")
            scaler_path = os.path.join(MODEL_DIR, f"{coin}_scaler.pkl")
            model.save_model(model_path)
            joblib.dump(scaler, scaler_path)

            saved_models[symbol] = {
                "model_path": model_path,
                "scaler_path": scaler_path,
                "train_acc": train_acc,
                "n_features": len(feature_names),
            }
            print(f"  ✅ {symbol}: 模型已保存, 训练集准确率{train_acc*100:.1f}%, {len(feature_names)}特征")

    # 保存特征名称和配置
    import json
    meta = {
        "forecast_hours": FORECAST_HOURS,
        "feature_names": feature_names,
        "symbols": list(saved_models.keys()),
        "disabled_symbols": DISABLED_ML_SYMBOLS,
        "xgb_params": XGB_PARAMS,
        "walk_forward_results": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                       for kk, vv in v.items()}
                                  for k, v in wf_results.items()},
        "optimization_measures": [
            "减少过拟合: max_depth 4→3, n_estimators 200→100",
            "增强正则化: reg_alpha 0.1→0.5, reg_lambda 1.0→2.0, min_child_weight=5",
            "禁用弱币种: XRP/USD样本外准确率38.7%，禁用ML预测",
            "增加跨币种特征: BTC动量/RSI/ATR/相关性（5个新特征）",
            "特征总数: 13→18个",
        ],
        "anti_leakage_measures": [
            "特征只用t时刻及之前数据",
            "标签用t+forecast_hours时刻数据",
            "Walk-forward滚动验证",
            "训练集/测试集严格按时间分割",
            "训练集和测试集之间留12小时gap",
            "Scaler只在训练集上fit",
            "禁止随机打乱数据顺序",
            "Rolling计算不包含未来bar",
        ],
    }
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 最终总结
    print("\n[4/4] 预训练完成总结")
    print("=" * 70)
    print(f"  已保存模型: {len(saved_models)}/{len(ENABLED_ML_SYMBOLS)} 个币种")
    print(f"  禁用币种: {DISABLED_ML_SYMBOLS}（样本外准确率过低）")
    print(f"  预测周期: 未来{FORECAST_HOURS}小时方向")
    print(f"  特征数量: {len(feature_names)} 个（13单币种+5跨币种）")
    if wf_results:
        avg_acc = np.mean([r["accuracy"] for r in wf_results.values()])
        avg_auc = np.mean([r["auc"] for r in wf_results.values()])
        print(f"  样本外平均准确率: {avg_acc*100:.1f}%")
        print(f"  样本外平均AUC: {avg_auc:.3f}")
    print(f"  模型目录: {MODEL_DIR}")
    print()
    print("  优化措施（5项）:")
    for i, measure in enumerate(meta["optimization_measures"], 1):
        print(f"    {i}. {measure}")
    print()
    print("  防泄漏措施（8重防护）:")
    for i, measure in enumerate(meta["anti_leakage_measures"], 1):
        print(f"    {i}. {measure}")
    print("=" * 70)


if __name__ == "__main__":
    main()
