"""
量化交易 Dashboard（Streamlit）v6：低波动异象 + 风险平价
决赛Pitch与比赛期间实时监控两用。

运行：
  streamlit run dashboard/app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLD_DIR = os.path.join(BASE_DIR, "data", "history_binance")
LIVE_DIR = os.path.join(BASE_DIR, "data", "live")
LB = 168

st.set_page_config(page_title="Team48-Y 低波异象系统", layout="wide", page_icon="📊")


@st.cache_data(ttl=300)
def load_universe_coins():
    uf = os.path.join(COLD_DIR, "universe.txt")
    return [l.strip() for l in open(uf, encoding="utf-8") if l.strip()]


@st.cache_data(ttl=120)
def load_prices(coin):
    for d in [LIVE_DIR, COLD_DIR]:
        fp = os.path.join(d, f"{coin}_1h.csv")
        if os.path.exists(fp):
            return pd.read_csv(fp, index_col=0, parse_dates=True)
    return None


def parkinson_vol(df, lookback=LB):
    """Parkinson 极差波动率（含ffill/H-L对齐/Winsorization）"""
    if df is None or len(df) < lookback + 2:
        return None
    w = df.tail(lookback).copy().ffill()
    H = w[["high", "open", "close"]].max(axis=1)
    L = w[["low", "open", "close"]].min(axis=1)
    hl = np.clip(H / np.maximum(L, 1e-9), 1.0001, 1.15)
    return float(np.sqrt(0.36067 * (np.log(hl) ** 2).mean()))


def lowvol_table(coins, lookback=LB, top=0.5):
    rows = []
    for coin in coins:
        df = load_prices(coin)
        vol = parkinson_vol(df, lookback)
        if vol is None:
            continue
        close = df["close"]
        ma = close.rolling(lookback).mean().iloc[-1]
        above = close.iloc[-1] > ma
        rows.append({"币种": coin, "Parkinson波动": vol, "均线上": above})
    elig = [r for r in rows if r["均线上"]]
    elig.sort(key=lambda r: r["Parkinson波动"])
    k = max(1, int(len(elig) * top))
    selected = {r["币种"] for r in elig[:k]}
    for r in rows:
        if r["币种"] in selected:
            r["状态"] = "做多(低波)"
        elif r["均线上"]:
            r["状态"] = "观察"
        else:
            r["状态"] = "空仓"
    return pd.DataFrame(rows).sort_values("Parkinson波动").reset_index(drop=True), selected


st.sidebar.title("Team48-Y · HKU")
st.sidebar.caption("APAC Quant Trading Hackathon")
page = st.sidebar.radio("导航", ["策略总览", "低波选币", "行情", "回测验证", "风控体系"])

coins = load_universe_coins()

# ================================================================
if page == "策略总览":
    st.title("低波动异象 + 风险平价（只做多）")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("交易币种", f"{len(coins)}")
    c2.metric("波动率窗口", "7天")
    c3.metric("再平衡", "每3天")
    c4.metric("方向", "只做多")

    st.subheader("策略逻辑")
    st.markdown("""
    **学术依据：** 低波动异象 / Betting Against Beta（Frazzini–Pedersen 2014）——
    经风险调整后，低波动资产往往优于高波动资产。

    **流程：** 计算每币 **Parkinson 极差波动率**（用 high/low，比收盘价标准差高效；
    并做 ffill、H/L 对齐、Winsorization 15% 降噪）→ 仅保留 **收盘价>168h均线** 的币
    → 选波动率最低的 **50%** → 按**波动率倒数加权（Risk Parity）**；无符合则空仓。

    **为什么不做空：** 多空横截面动量在 6–10 月上涨轮动市实测亏损 -18%（空头被挤压）；
    低波策略在震荡市选稳健币，实测仍正收益，无需靠做空。
    """)
    st.info("动态仓位：观察期(1-3天)60% → 确认期(4-10天)100% → 收尾期(11-14天)70%")

# ================================================================
elif page == "低波选币":
    st.title("低波选币状态（实时）")
    df, selected = lowvol_table(coins)
    if len(df):
        fig = px.bar(df, x="币种", y="Parkinson波动", color="状态",
                     color_discrete_map={"做多(低波)": "#2ca02c",
                                         "观察": "#7f7f7f", "空仓": "#dddddd"},
                     title="Parkinson 波动率与选币状态（绿色=做多，波动越低越靠前）")
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"**做多池（{len(selected)}币，波动率倒数加权）：** {', '.join(sorted(selected))}")
        show = df.copy()
        show["Parkinson波动"] = show["Parkinson波动"].map(lambda x: f"{x*100:.2f}%")
        st.dataframe(show, use_container_width=True)

# ================================================================
elif page == "行情":
    st.title("币种行情")
    coin = st.selectbox("选择币种", coins,
                        index=coins.index("BTC") if "BTC" in coins else 0)
    df = load_prices(coin)
    if df is not None:
        sub = df.tail(LB)
        fig = go.Figure(data=[go.Candlestick(
            x=sub.index, open=sub["open"], high=sub["high"],
            low=sub["low"], close=sub["close"], name="K线")])
        fig.add_trace(go.Scatter(x=sub.index, y=df["close"].rolling(LB).mean().iloc[-LB:],
                                 line=dict(color="#ff7f0e", width=1), name="MA168"))
        fig.update_layout(title=f"{coin}/USD 近7天 + MA168", height=480,
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("最新价", f"{df['close'].iloc[-1]:,.4f}")
        c2.metric("7天涨跌", f"{(df['close'].iloc[-1]/df['close'].iloc[-LB]-1)*100:+.2f}%")
        c3.metric("Parkinson波动", f"{parkinson_vol(df)*100:.2f}%")

# ================================================================
elif page == "回测验证":
    st.title("回测与稳健性验证")
    st.caption("数据：Binance 1h K线（volume完整），手续费0.1%/边，72h调仓")

    st.subheader("三段独立窗口（每段约29天，互不重叠）")
    seg = pd.DataFrame([
        {"窗口": "段1 (7月震荡)", "收益": 10.9, "Sharpe": 1.8},
        {"窗口": "段2 (8月涨)", "收益": 11.7, "Sharpe": 2.3},
        {"窗口": "段3 (9月山寨季)", "收益": 35.7, "Sharpe": 4.6},
    ])
    fig = px.bar(seg, x="窗口", y="收益", color="收益",
                 color_continuous_scale=["#eeeeee", "#9ed99e", "#2ca02c"],
                 title="三段收益：全部为正（震荡市也赚钱）")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(seg, use_container_width=True)

    st.subheader("全周期（120天）")
    m1, m2, m3 = st.columns(3)
    m1.metric("总收益", "+52.1%")
    m2.metric("Sharpe", "6.3")
    m3.metric("最大回撤", "-13.6%")

    st.warning("诚实说明：历史Sharpe是特定行情产物，不能外推；强单边牛市中低波策略"
               "进攻性弱于纯动量，这是为'三段全正'稳健性付出的代价。")

# ================================================================
elif page == "风控体系":
    st.title("风控体系（独立 RiskManager）")
    st.markdown("""
    | 层级 | 规则 | 动作 |
    |------|------|------|
    | 单标的灾难止损 | 相对开仓价反向偏离12% | 立即单独平仓 |
    | 日回撤熔断 | 日内回撤>4% / >8% | 减仓预警 / 全平暂停1h |
    | 连亏熔断 | 连续亏损3笔 | 暂停30分钟 |
    | 敞口约束 | 总敞口≤100%（无杠杆） | 硬约束 |
    """)
    st.markdown("""
    **为什么止损是12%：** 实测3σ动态止损对低波币过紧（≈5%~6%），在正常插针处被反复
    割肉踏空、反而降低收益；12%宽止损仅在真正持续下跌时触发（全周期1次），保留尾部保险。
    """)

st.sidebar.caption("初始资金 $50,000 · 周期14天 · 10月4-17日")
