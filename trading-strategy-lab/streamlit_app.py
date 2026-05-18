
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import contextlib
import html
import io
import itertools
import logging
import re
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller


logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


STOCK_POOL_CSV = """code,name,industry
1101,台泥,水泥工業
1102,亞泥,水泥工業
1210,大成,食品工業
1216,統一,食品工業
1301,台塑,塑膠工業
1303,南亞,塑膠工業
1312,國喬,塑膠工業
1314,中石化,塑膠工業
1319,東陽,汽車工業
1326,台化,塑膠工業
1402,遠東新,紡織纖維
1440,南紡,紡織纖維
1476,儒鴻,紡織纖維
1477,聚陽,紡織纖維
1503,士電,電機機械
1504,東元,電機機械
1513,中興電,電機機械
1536,和大,汽車工業
1560,中砂,電機機械
1565,精華,生技醫療業
1590,亞德客-KY,電機機械
1605,華新,電器電纜
1608,華榮,電器電纜
1609,大亞,電器電纜
1717,長興,化學工業
1718,中纖,化學工業
1722,台肥,化學工業
1795,美時,生技醫療業
1802,台玻,玻璃陶瓷
1904,正隆,造紙工業
1905,華紙,造紙工業
1907,永豐餘,造紙工業
1909,榮成,造紙工業
2002,中鋼,鋼鐵工業
2006,東和鋼鐵,鋼鐵工業
2014,中鴻,鋼鐵工業
2027,大成鋼,鋼鐵工業
2049,上銀,電機機械
2059,川湖,電子零組件業
2105,正新,橡膠工業
2201,裕隆,汽車工業
2231,為升,汽車工業
2301,光寶科,光電業
2303,聯電,半導體業
2308,台達電,電子零組件業
2312,金寶,其他電子業
2313,華通,電子零組件業
2317,鴻海,其他電子業
2323,中環,光電業
2324,仁寶,電腦及週邊設備業
2327,國巨,電子零組件業
2328,廣宇,電子零組件業
2329,華泰,半導體業
2330,台積電,半導體業
2331,精英,電腦及週邊設備業
2332,友訊,通信網路業
2337,旺宏,半導體業
2338,光罩,半導體業
2340,台亞,半導體業
2344,華邦電,半導體業
2345,智邦,通信網路業
2347,聯強,電子通路業
2352,佳世達,電腦及週邊設備業
2353,宏碁,電腦及週邊設備業
2354,鴻準,其他電子業
2355,敬鵬,電子零組件業
2356,英業達,電腦及週邊設備業
2357,華碩,電腦及週邊設備業
2360,致茂,其他電子業
2367,燿華,電子零組件業
2368,金像電,電子零組件業
2371,大同,電機機械
2376,技嘉,電腦及週邊設備業
2377,微星,電腦及週邊設備業
2379,瑞昱,半導體業
2382,廣達,電腦及週邊設備業
2383,台光電,電子零組件業
2385,群光,電子零組件業
2388,威盛,半導體業
2392,正崴,電子零組件業
2393,億光,光電業
2395,研華,電腦及週邊設備業
2401,凌陽,半導體業
2404,漢唐,其他電子業
2408,南亞科,半導體業
2409,友達,光電業
2412,中華電,通信網路業
2421,建準,電子零組件業
2439,美律,通信網路業
2441,超豐,半導體業
2449,京元電子,半導體業
2454,聯發科,半導體業
2455,全新,通信網路業
2457,飛宏,電子零組件業
2458,義隆,半導體業
2474,可成,其他電子業
2481,強茂,半導體業
2485,兆赫,通信網路業
2486,一詮,光電業
2489,瑞軒,光電業
2492,華新科,電子零組件業
2498,宏達電,通信網路業
2515,中工,建材營造業
2520,冠德,建材營造業
2542,興富發,建材營造業
2548,華固,建材營造業
2603,長榮,航運業
2605,新興,航運業
2606,裕民,航運業
2609,陽明,航運業
2610,華航,航運業
2615,萬海,航運業
2618,長榮航,航運業
2633,台灣高鐵,航運業
2634,漢翔,航運業
2801,彰銀,金融保險業
2834,臺企銀,金融保險業
2880,華南金,金融保險業
2881,富邦金,金融保險業
2882,國泰金,金融保險業
2883,凱基金,金融保險業
2884,玉山金,金融保險業
2885,元大金,金融保險業
2886,兆豐金,金融保險業
2887,台新新光金,金融保險業
2890,永豐金,金融保險業
2891,中信金,金融保險業
2892,第一金,金融保險業
2913,農林,貿易百貨業
2915,潤泰全,其他業
3005,神基,電腦及週邊設備業
3006,晶豪科,半導體業
3008,大立光,光電業
3017,奇鋐,電腦及週邊設備業
3019,亞光,光電業
3034,聯詠,半導體業
3035,智原,半導體業
3036,文曄,電子通路業
3037,欣興,電子零組件業
3042,晶技,電子零組件業
3044,健鼎,電子零組件業
3045,台灣大,通信網路業
3078,僑威,電子零組件業
3081,聯亞,通信網路業
3105,穩懋,半導體業
3152,璟德,通信網路業
3189,景碩,半導體業
3211,順達,電腦及週邊設備業
3227,原相,半導體業
3231,緯創,電腦及週邊設備業
3260,威剛,半導體業
3264,欣銓,半導體業
3293,鈊象,文化創意業
3324,雙鴻,電腦及週邊設備業
3374,精材,半導體業
3376,新日興,電子零組件業
3380,明泰,通信網路業
3406,玉晶光,光電業
3443,創意,半導體業
3481,群創,光電業
3529,力旺,半導體業
3532,台勝科,半導體業
3533,嘉澤,電子零組件業
3552,同致,汽車工業
3653,健策,電子零組件業
3661,世芯-KY,半導體業
3665,貿聯-KY,其他電子業
3673,TPK-KY,光電業
3680,家登,其他電子業
3691,碩禾,光電業
3702,大聯大,電子通路業
3706,神達,電腦及週邊設備業
3711,日月光投控,半導體業
3714,富采,光電業
4123,晟德,生技醫療業
4128,中天,生技醫療業
4162,智擎,生技醫療業
4736,泰博,生技醫療業
4743,合一,生技醫療業
4904,遠傳,通信網路業
4919,新唐,半導體業
4938,和碩,電腦及週邊設備業
4958,臻鼎-KY,電子零組件業
5009,榮剛,鋼鐵工業
5269,祥碩,半導體業
5274,信驊,半導體業
5347,世界,半導體業
5371,中光電,光電業
5388,中磊,通信網路業
5425,台半,半導體業
5457,宣德,電子零組件業
5483,中美晶,半導體業
5534,長虹,建材營造業
5871,中租-KY,其他業
5876,上海商銀,金融保險業
5880,合庫金,金融保險業
5904,寶雅,貿易百貨業
6005,群益證,金融保險業
6116,彩晶,光電業
6121,新普,電腦及週邊設備業
6139,亞翔,其他電子業
6147,頎邦,半導體業
6153,嘉聯益,電子零組件業
6173,信昌電,電子零組件業
6176,瑞儀,光電業
6182,合晶,半導體業
6188,廣明,光電業
6213,聯茂,電子零組件業
6223,旺矽,半導體業
6239,力成,半導體業
6245,立端,電腦及週邊設備業
6257,矽格,半導體業
6269,台郡,電子零組件業
6271,同欣電,半導體業
6274,台燿,電子零組件業
6278,台表科,電子零組件業
6279,胡連,電子零組件業
6282,康舒,電子零組件業
6285,啟碁,通信網路業
6290,良維,電子零組件業
6414,樺漢,電腦及週邊設備業
6443,元晶,光電業
6472,保瑞,生技醫療業
6488,環球晶,半導體業
6505,台塑化,油電燃氣業
6510,精測,半導體業
6526,達發,半導體業
6547,高端疫苗,生技醫療業
6669,緯穎,電腦及週邊設備業
6757,台灣虎航,航運業
6770,力積電,半導體業
8039,台虹,電子零組件業
8044,網家,數位雲端
8046,南電,電子零組件業
8069,元太,光電業
8086,宏捷科,半導體業
8112,至上,電子通路業
8150,南茂,半導體業
8163,達方,電子零組件業
8299,群聯,半導體業
8358,金居,電子零組件業
8436,大江,生技醫療業
9904,寶成,運動休閒
9914,美利達,運動休閒
9938,百和,其他業
9939,宏全,其他業
9945,潤泰新,建材營造業
9958,世紀鋼,鋼鐵工業
"""

single_stock_futures = pd.read_csv(io.StringIO(STOCK_POOL_CSV), dtype=str).to_dict("records")
KNOWN_NAMES = {item["code"]: item["name"] for item in single_stock_futures}

PERIOD = "3y"
INTERVAL = "1d"
MIN_OBS = 60
TOP_N = 10
ADF_P_THRESHOLD = 0.05
R2_THRESHOLD = 0.50
MIN_HALF_LIFE = 2
MAX_HALF_LIFE = 60
MAX_TREND_STRENGTH = 1.0
MIN_CROSSINGS_PER_YEAR = 4
MAX_OPEN_POSITIONS_PER_DIRECTION = 3
MA_WINDOW = 5


@dataclass(frozen=True)
class Config:
    lookback: int
    entry_z: float
    exit_z: float
    stop_z: float
    capital: float
    broker_fee: float
    sell_tax: float
    integer_shares: bool


st.set_page_config(page_title="Trading Strategy Lab", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.7rem; }
    div[data-testid="stMetric"] {
      background: #fff; border: 1px solid #e6e8ef; border-radius: 8px;
      padding: 14px 16px; box-shadow: 0 1px 2px rgba(20, 29, 47, 0.04);
    }
    .strategy-card {
      background: #ffffff; border: 1px solid #e6e8ef; border-radius: 12px;
      padding: 24px 26px; box-shadow: 0 1px 2px rgba(20, 29, 47, 0.04);
      margin-top: 16px; margin-bottom: 12px;
    }
    .strategy-card h3 { margin-top: 0; margin-bottom: 8px; }
    .strategy-card p { color: #607080; margin-bottom: 0; line-height: 1.7; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    if "selected_strategy" not in st.session_state:
        st.session_state["selected_strategy"] = None
    if "a_code_input" not in st.session_state:
        st.session_state["a_code_input"] = ""
    if "b_code_input" not in st.session_state:
        st.session_state["b_code_input"] = ""
    if "s2_a_code_input" not in st.session_state:
        st.session_state["s2_a_code_input"] = ""
    if "s2_b_code_input" not in st.session_state:
        st.session_state["s2_b_code_input"] = ""

    if st.session_state["selected_strategy"] is None:
        render_strategy_selector()
        return

    st.title("Trading Strategy Lab")

    if st.session_state["selected_strategy"] == "strategy_1":
        st.caption("策略1：Pair Trading 單一 pair 回測，使用 rolling OLS hedge ratio。")
        settings = sidebar_strategy1_settings()
        render_strategy1_backtest(settings)
        return

    if st.session_state["selected_strategy"] == "strategy_2":
        st.caption("策略2：MisbahAN-style Cointegration Pair Screening + 台股分類股票池 + 單一 pair 回測。")
        settings = sidebar_strategy2_settings()
        render_strategy2(settings)
        return

    st.session_state["selected_strategy"] = None
    st.rerun()


def render_strategy_selector() -> None:
    st.title("Trading Strategy Lab")
    st.caption("請先選擇要使用的交易策略。")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="strategy-card">
              <h3>策略1：Pair Trading 回測</h3>
              <p>
                手動指定兩檔台股，使用 rolling OLS 估計 spread 與 z-score，
                並用 OLS beta 作為雙邊部位權重。
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("進入策略1", type="primary", use_container_width=True):
            st.session_state["selected_strategy"] = "strategy_1"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="strategy-card">
              <h3>策略2：Cointegration Pairs 台股實證</h3>
              <p>
                保留你的台股股票池與細產業分類，選股邏輯改成更接近 MisbahAN：
                OLS R²、residual ADF test、Durbin-Watson、condition number，再選單一 pair 回測。
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("進入策略2", use_container_width=True):
            st.session_state["selected_strategy"] = "strategy_2"
            st.rerun()

    st.info("策略2會下載同一細產業內多檔台股資料，第一次執行會比策略1久。")


def sidebar_strategy1_settings() -> dict[str, object]:
    if st.sidebar.button("返回策略選擇"):
        st.session_state["selected_strategy"] = None
        st.rerun()

    st.sidebar.caption("目前策略：策略1 Pair Trading")
    st.sidebar.divider()

    st.sidebar.header("回測設定")
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        a_code = st.text_input("A code", key="a_code_input").strip().upper()
    with col_b:
        b_code = st.text_input("B code", key="b_code_input").strip().upper()

    today = pd.Timestamp.today().date()
    start_default = (pd.Timestamp(today) - pd.DateOffset(years=2)).date()
    start = st.sidebar.date_input("Backtest start", value=start_default, key="s1_start")
    end = st.sidebar.date_input("Backtest end", value=today, key="s1_end")
    benchmark = st.sidebar.text_input("Benchmark", value="^TWII", key="s1_benchmark")

    a_name = resolve_stock_name(a_code) if a_code else ""
    b_name = resolve_stock_name(b_code) if b_code else ""
    if a_code or b_code:
        st.sidebar.caption(
            f"辨識結果：{format_stock_label(a_code, a_name)} / {format_stock_label(b_code, b_name)}"
        )

    st.sidebar.divider()
    st.sidebar.header("策略參數")
    lookback = st.sidebar.slider("Rolling OLS lookback", 40, 260, 120, 10, key="s1_lookback")
    entry_z = st.sidebar.slider("Entry z-score", 0.5, 4.0, 2.0, 0.1, key="s1_entry_z")
    exit_z = st.sidebar.slider("Exit z-score", -1.0, 1.0, 0.0, 0.1, key="s1_exit_z")
    stop_z = st.sidebar.slider("Stop z-score", 2.0, 6.0, 3.5, 0.1, key="s1_stop_z")
    capital = st.sidebar.number_input("Capital per pair", min_value=10_000, value=100_000, step=10_000, key="s1_capital")
    integer_shares = st.sidebar.toggle("整股交易", value=True, key="s1_integer_shares")

    st.sidebar.divider()
    st.sidebar.header("交易成本")
    broker_fee = st.sidebar.number_input("Broker fee", min_value=0.0, value=0.001425, step=0.0001, format="%.6f", key="s1_broker_fee")
    sell_tax = st.sidebar.number_input("Sell tax", min_value=0.0, value=0.001425, step=0.0001, format="%.6f", key="s1_sell_tax")

    return {
        "a_code": a_code,
        "a_name": a_name,
        "b_code": b_code,
        "b_name": b_name,
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "benchmark": benchmark,
        "config": Config(
            lookback=int(lookback),
            entry_z=float(entry_z),
            exit_z=float(exit_z),
            stop_z=float(stop_z),
            capital=float(capital),
            broker_fee=float(broker_fee),
            sell_tax=float(sell_tax),
            integer_shares=bool(integer_shares),
        ),
    }


def sidebar_strategy2_settings() -> dict[str, object]:
    if st.sidebar.button("返回策略選擇"):
        st.session_state["selected_strategy"] = None
        st.rerun()

    st.sidebar.caption("目前策略：策略2 MisbahAN-style Cointegration 台股實證")
    st.sidebar.divider()

    st.sidebar.header("單一 Pair 回測標的")
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        a_code = st.text_input("A code", key="s2_a_code_input", placeholder="例如：2330").strip().upper()
    with col_b:
        b_code = st.text_input("B code", key="s2_b_code_input", placeholder="例如：2303").strip().upper()

    a_name = resolve_stock_name(a_code) if a_code else ""
    b_name = resolve_stock_name(b_code) if b_code else ""
    if a_code or b_code:
        st.sidebar.caption(
            f"辨識結果：{format_stock_label(a_code, a_name)} / {format_stock_label(b_code, b_name)}"
        )

    today = pd.Timestamp.today().date()
    start_default = (pd.Timestamp(today) - pd.DateOffset(years=3)).date()
    start = st.sidebar.date_input("Full sample start", value=start_default, key="s2_start")
    end = st.sidebar.date_input("Full sample end", value=today, key="s2_end")
    benchmark = st.sidebar.text_input("Benchmark", value="^TWII", key="s2_benchmark")

    st.sidebar.divider()
    st.sidebar.header("Formation / Test 設定")
    formation_ratio = st.sidebar.slider(
        "Formation ratio",
        min_value=0.50,
        max_value=0.85,
        value=0.70,
        step=0.05,
        help="前段資料用來檢查 cointegration；後段資料用策略1格式回測。",
        key="s2_formation_ratio",
    )
    adf_pvalue = st.sidebar.selectbox("ADF p-value threshold", [0.01, 0.05, 0.10], index=1, key="s2_adf_pvalue")

    st.sidebar.divider()
    st.sidebar.header("MisbahAN-style Pair Screening")
    industry_groups = build_industry_groups()
    industry_options = sorted(industry_groups.keys())
    default_industry = "半導體業" if "半導體業" in industry_options else industry_options[0]

    screen_industry = st.sidebar.selectbox(
        "篩選細產業",
        industry_options,
        index=industry_options.index(default_industry),
        key="s2_screen_industry",
    )
    screen_period = st.sidebar.selectbox("篩選資料期間", ["1y", "2y", "3y", "5y", "10y"], index=2, key="s2_screen_period")
    corr_threshold = st.sidebar.slider(
        "Correlation threshold",
        min_value=0.00,
        max_value=0.98,
        value=0.00,
        step=0.01,
        key="s2_corr_threshold",
        help="設為 0 表示不使用 correlation 預篩；MisbahAN 原始邏輯主要用 OLS R² 篩選。",
    )
    r2_threshold = st.sidebar.slider(
        "OLS R-squared threshold",
        min_value=0.00,
        max_value=0.99,
        value=0.50,
        step=0.01,
        key="s2_r2_threshold",
        help="對齊 MisbahAN 的 stockPairs = df[df['r_squared'] > 0.5] 篩選邏輯。",
    )
    top_n = st.sidebar.slider("顯示最佳 pair 數量", 1, 20, 10, 1, key="s2_top_n")

    st.sidebar.divider()
    st.sidebar.header("策略參數")
    lookback = st.sidebar.slider("Rolling OLS lookback", 40, 260, 120, 10, key="s2_lookback")
    entry_z = st.sidebar.slider("Entry z-score", 0.5, 4.0, 2.0, 0.1, key="s2_entry_z")
    exit_z = st.sidebar.slider("Exit z-score", -1.0, 1.0, 0.0, 0.1, key="s2_exit_z")
    stop_z = st.sidebar.slider("Stop z-score", 2.0, 6.0, 3.5, 0.1, key="s2_stop_z")
    capital = st.sidebar.number_input("Capital per pair", min_value=10_000, value=100_000, step=10_000, key="s2_capital")
    integer_shares = st.sidebar.toggle("整股交易", value=True, key="s2_integer_shares")

    st.sidebar.divider()
    st.sidebar.header("交易成本")
    broker_fee = st.sidebar.number_input("Broker fee", min_value=0.0, value=0.001425, step=0.0001, format="%.6f", key="s2_broker_fee")
    sell_tax = st.sidebar.number_input("Sell tax", min_value=0.0, value=0.001425, step=0.0001, format="%.6f", key="s2_sell_tax")

    return {
        "a_code": a_code,
        "a_name": a_name,
        "b_code": b_code,
        "b_name": b_name,
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "benchmark": benchmark,
        "formation_ratio": float(formation_ratio),
        "adf_pvalue": float(adf_pvalue),
        "screen_industry": screen_industry,
        "screen_period": screen_period,
        "corr_threshold": float(corr_threshold),
        "r2_threshold": float(r2_threshold),
        "top_n": int(top_n),
        "config": Config(
            lookback=int(lookback),
            entry_z=float(entry_z),
            exit_z=float(exit_z),
            stop_z=float(stop_z),
            capital=float(capital),
            broker_fee=float(broker_fee),
            sell_tax=float(sell_tax),
            integer_shares=bool(integer_shares),
        ),
    }


def render_strategy1_backtest(settings: dict[str, object]) -> None:
    a_code = str(settings["a_code"])
    b_code = str(settings["b_code"])
    a_name = str(settings["a_name"])
    b_name = str(settings["b_name"])
    start = pd.Timestamp(settings["start"])
    end = pd.Timestamp(settings["end"])
    config: Config = settings["config"]  # type: ignore[assignment]

    left, right = st.columns([0.52, 0.48])

    with left:
        st.subheader("回測標的")
        if a_code and b_code:
            st.markdown(f"### {format_stock_label(a_code, a_name)} / {format_stock_label(b_code, b_name)}")
            st.write("訊號使用 t 日 Close；交易使用 t+1 日 Open。")
        else:
            st.info("請先在左側輸入 A code / B code，或從右側推薦 pair 套用到回測。")
        run = st.button("Run backtest", type="primary", use_container_width=True)

    with right:
        render_strategy1_pair_screening_panel()

    if not run:
        st.info("調整左側設定後按 Run backtest。")
        return

    validate_pair_inputs(a_code, b_code, start, end)

    try:
        with st.spinner("下載價格並執行回測..."):
            warmup = max(config.lookback * 3, 365)
            download_start = start - pd.DateOffset(days=warmup)
            open_df, close_df = download_ohlc([a_code, b_code], download_start, end)
            benchmark = download_benchmark(str(settings["benchmark"]), download_start, end)
            result = run_backtest(open_df, close_df, benchmark, a_code, b_code, start, end, config)
    except Exception as exc:
        st.error(f"回測失敗：{exc}")
        return

    show_summary(result["summary"])
    show_latest_trading_signal(result, a_code, b_code, a_name, b_name, config)
    show_charts(result, a_code, b_code, str(settings["benchmark"]), config)
    show_tables(result)


def render_strategy2(settings: dict[str, object]) -> None:
    a_code = str(settings["a_code"])
    b_code = str(settings["b_code"])
    a_name = str(settings["a_name"])
    b_name = str(settings["b_name"])
    start = pd.Timestamp(settings["start"])
    end = pd.Timestamp(settings["end"])
    formation_ratio = float(settings["formation_ratio"])
    adf_pvalue = float(settings["adf_pvalue"])
    corr_threshold = float(settings["corr_threshold"])
    r2_threshold = float(settings["r2_threshold"])
    config: Config = settings["config"]  # type: ignore[assignment]

    st.subheader("策略2：MisbahAN-style Cointegration Pairs 台股實證")

    left, right = st.columns([0.55, 0.45])

    with left:
        st.markdown("#### 單一 Pair 回測")
        if a_code and b_code:
            st.markdown(f"### {format_stock_label(a_code, a_name)} / {format_stock_label(b_code, b_name)}")
            st.write(
                "策略2先用 formation period 做 MisbahAN-style cointegration diagnostics，"
                "再用 test period 做單一 pair 回測，顯示方式與策略1一致。"
            )
        else:
            st.info("策略2沒有預設 pair。請在左側輸入 A/B code，或從右側最佳 pair 篩選結果套用。")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Formation ratio", f"{formation_ratio:.0%}")
        m2.metric("ADF threshold", f"{adf_pvalue:.2f}")
        m3.metric("Corr threshold", f"{corr_threshold:.2f}")
        m4.metric("R² threshold", f"{r2_threshold:.2f}")

        run = st.button("Run 策略2單一 Pair 回測", type="primary", use_container_width=True)

    with right:
        render_strategy2_best_pairs_panel(settings)

    if not run:
        st.info("輸入兩檔股票代號，或從右側最佳 pair 結果套用後，按 Run 開始回測。")
        return

    validate_pair_inputs(a_code, b_code, start, end)

    try:
        with st.spinner("下載價格、執行 cointegration diagnostics 與回測..."):
            warmup = max(config.lookback * 3, 365)
            download_start = start - pd.DateOffset(days=warmup)
            open_df, close_df = download_ohlc([a_code, b_code], download_start, end)
            benchmark = download_benchmark(str(settings["benchmark"]), download_start, end)

            test_start = determine_strategy2_test_start(close_df, a_code, b_code, start, end, formation_ratio, config.lookback)
            diagnostics = analyze_strategy2_cointegration_pair(
                close_df=close_df,
                a_code=a_code,
                b_code=b_code,
                start=start,
                test_start=test_start,
                adf_pvalue=adf_pvalue,
                r2_threshold=r2_threshold,
            )
            result = run_backtest(open_df, close_df, benchmark, a_code, b_code, test_start, end, config)

    except Exception as exc:
        st.error(f"策略2執行失敗：{exc}")
        return

    show_strategy2_diagnostics(diagnostics, a_code, b_code, a_name, b_name)
    show_summary(result["summary"])
    show_latest_trading_signal(result, a_code, b_code, a_name, b_name, config)
    show_charts(result, a_code, b_code, str(settings["benchmark"]), config)
    show_tables(result)


def validate_pair_inputs(a_code: str, b_code: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
    if not a_code or not b_code:
        st.error("請先輸入 A code 和 B code。")
        st.stop()
    if a_code == b_code:
        st.error("A code 和 B code 不能相同。")
        st.stop()
    if start >= end:
        st.error("開始日期必須早於結束日期。")
        st.stop()


def render_strategy1_pair_screening_panel() -> None:
    st.subheader("細產業 Pair 推薦")
    industry_groups = build_industry_groups()
    industry_options = sorted(industry_groups.keys())

    industry = st.selectbox(
        "選擇細產業",
        industry_options,
        index=industry_options.index("半導體業") if "半導體業" in industry_options else 0,
        key="s1_screen_industry",
    )
    stocks = tuple(industry_groups[industry])
    st.caption("策略1右側推薦使用相同台股分類，但篩選邏輯也已統一為 MisbahAN-style OLS R² + residual ADF。")

    if st.button("篩選最佳 10 組 Pair", use_container_width=True, key="s1_screen_button"):
        with st.spinner("下載價格並篩選 pair..."):
            try:
                best_pairs = screen_best_pairs_by_industry(
                    industry=industry,
                    stocks=stocks,
                    corr_threshold=0.0,
                    r2_threshold=R2_THRESHOLD,
                    top_n=TOP_N,
                    period=PERIOD,
                    adf_pvalue=ADF_P_THRESHOLD,
                )
            except Exception as exc:
                st.error(f"篩選失敗：{exc}")
                return
        st.session_state["screened_industry"] = industry
        st.session_state["screened_pairs"] = best_pairs.to_dict("records")

    records = st.session_state.get("screened_pairs", [])
    if not records or st.session_state.get("screened_industry") != industry:
        st.info("請選擇細產業後按「篩選最佳 10 組 Pair」。")
        return

    best_pairs_df = pd.DataFrame(records)
    if best_pairs_df.empty:
        st.warning("這個細產業目前沒有符合篩選條件的 pair。")
        return

    show_best_pairs_table(best_pairs_df)

    pair_labels = build_pair_labels(best_pairs_df)
    selected_label = st.selectbox("套用推薦 pair 到回測", pair_labels, key="s1_best_pair_select")
    selected = best_pairs_df.iloc[pair_labels.index(selected_label)]

    if st.button("套用選取 pair", type="primary", use_container_width=True, key="s1_apply_pair"):
        st.session_state["a_code_input"] = str(selected["stock_A_code"])
        st.session_state["b_code_input"] = str(selected["stock_B_code"])
        st.rerun()


def render_strategy2_best_pairs_panel(settings: dict[str, object]) -> None:
    st.markdown("#### 最佳 Pair 篩選結果")

    industry = str(settings["screen_industry"])
    period = str(settings["screen_period"])
    corr_threshold = float(settings["corr_threshold"])
    r2_threshold = float(settings["r2_threshold"])
    adf_pvalue = float(settings["adf_pvalue"])
    top_n = int(settings["top_n"])

    industry_groups = build_industry_groups()
    stocks = tuple(industry_groups.get(industry, []))

    st.caption(
        f"細產業：{industry}｜股票數：{len(stocks)}｜資料期間：{period}｜"
        f"Correlation threshold：{corr_threshold:.2f}｜"
        f"OLS R² threshold：{r2_threshold:.2f}｜Top N：{top_n}"
    )

    run_screen = st.button("篩選策略2最佳 Pair", use_container_width=True, key="s2_screen_button")

    if run_screen:
        with st.spinner("下載價格並執行 MisbahAN-style pair screening..."):
            try:
                best_pairs = screen_best_pairs_by_industry(
                    industry=industry,
                    stocks=stocks,
                    corr_threshold=corr_threshold,
                    r2_threshold=r2_threshold,
                    top_n=top_n,
                    period=period,
                    adf_pvalue=adf_pvalue,
                )
            except Exception as exc:
                st.error(f"篩選失敗：{exc}")
                return

        st.session_state["s2_screened_industry"] = industry
        st.session_state["s2_screened_period"] = period
        st.session_state["s2_screened_corr_threshold"] = corr_threshold
        st.session_state["s2_screened_r2_threshold"] = r2_threshold
        st.session_state["s2_screened_adf_pvalue"] = adf_pvalue
        st.session_state["s2_screened_top_n"] = top_n
        st.session_state["s2_screened_pairs"] = best_pairs.to_dict("records")

    records = st.session_state.get("s2_screened_pairs", [])
    same_setting = (
        st.session_state.get("s2_screened_industry") == industry
        and st.session_state.get("s2_screened_period") == period
        and float(st.session_state.get("s2_screened_corr_threshold", -1)) == corr_threshold
        and float(st.session_state.get("s2_screened_r2_threshold", -1)) == r2_threshold
        and float(st.session_state.get("s2_screened_adf_pvalue", -1)) == adf_pvalue
        and int(st.session_state.get("s2_screened_top_n", -1)) == top_n
    )

    if not records or not same_setting:
        st.info("按「篩選策略2最佳 Pair」後，這裡會列出最接近 MisbahAN 選股邏輯排序後的最佳幾對。")
        return

    best_pairs_df = pd.DataFrame(records)
    if best_pairs_df.empty:
        st.warning("這個設定下沒有找到 pair。可降低 R² threshold、降低 correlation threshold、放寬 ADF threshold，或延長篩選資料期間。")
        return

    show_best_pairs_table(best_pairs_df)

    pair_labels = build_pair_labels(best_pairs_df)
    selected_label = st.selectbox("套用最佳 pair 到策略2回測", pair_labels, key="s2_selected_best_pair")
    selected = best_pairs_df.iloc[pair_labels.index(selected_label)]

    if st.button("套用選取 pair 到策略2", type="primary", use_container_width=True, key="s2_apply_pair"):
        st.session_state["s2_a_code_input"] = str(selected["stock_A_code"])
        st.session_state["s2_b_code_input"] = str(selected["stock_B_code"])
        st.rerun()


def show_best_pairs_table(best_pairs_df: pd.DataFrame) -> None:
    display_cols = [
        "rank",
        "stock_A_code", "stock_A_name",
        "stock_B_code", "stock_B_name",
        "correlation",
        "r_squared",
        "beta_hedge_ratio",
        "adf_pvalue",
        "adf_stat",
        "crit_1pct",
        "crit_5pct",
        "crit_10pct",
        "durbin_watson",
        "condition_number",
        "half_life",
        "trend_strength",
        "crossings_per_year",
        "score",
        "suitable",
    ]
    show_df = best_pairs_df[[c for c in display_cols if c in best_pairs_df.columns]].copy()
    numeric_cols = [
        "correlation",
        "r_squared",
        "beta_hedge_ratio",
        "adf_pvalue",
        "adf_stat",
        "crit_1pct",
        "crit_5pct",
        "crit_10pct",
        "durbin_watson",
        "condition_number",
        "half_life",
        "trend_strength",
        "crossings_per_year",
    ]
    for c in numeric_cols:
        if c in show_df.columns:
            show_df[c] = pd.to_numeric(show_df[c], errors="coerce").round(4)
    st.dataframe(show_df, use_container_width=True, hide_index=True)


def build_pair_labels(best_pairs_df: pd.DataFrame) -> list[str]:
    return [
        f"{int(row['rank'])}. {row['stock_A_code']} {row['stock_A_name']} / "
        f"{row['stock_B_code']} {row['stock_B_name']} | "
        f"R²={float(row.get('r_squared', np.nan)):.3f} | "
        f"ADF p={float(row.get('adf_pvalue', np.nan)):.4f} | "
        f"score={int(row.get('score', 0))}"
        for _, row in best_pairs_df.iterrows()
    ]


def build_industry_groups() -> dict[str, list[tuple[str, str]]]:
    groups: dict[str, list[tuple[str, str]]] = {}
    for item in single_stock_futures:
        industry = str(item["industry"])
        code = str(item["code"])
        name = str(item["name"])
        groups.setdefault(industry, []).append((code, name))
    return groups


@st.cache_data(ttl=3600, show_spinner=False)
def screen_best_pairs_by_industry(
    industry: str,
    stocks: tuple[tuple[str, str], ...],
    corr_threshold: float = 0.0,
    r2_threshold: float = R2_THRESHOLD,
    top_n: int = TOP_N,
    period: str = PERIOD,
    adf_pvalue: float = ADF_P_THRESHOLD,
) -> pd.DataFrame:
    """
    MisbahAN-style pair selection adapted to the user's Taiwan stock universe.

    1. Use selected Taiwan industry group as universe.
    2. For every pair, run OLS on raw close prices:
       price_A = alpha + beta * price_B + residual.
    3. Record alpha, beta, R-squared, Durbin-Watson, condition number.
    4. Run ADF on residuals with maxlag=0, matching MisbahAN's implementation.
    5. Select pairs using R² threshold and ADF p-value threshold.
    6. Rank pairs; no pair portfolio aggregation is performed.
    """

    if len(stocks) <= 1:
        return pd.DataFrame()

    price_df = download_screening_prices(stocks, period, INTERVAL)
    group_price = get_group_price(price_df, stocks)

    if group_price.shape[1] <= 1:
        return pd.DataFrame()

    group_price = group_price.dropna(axis=1, thresh=MIN_OBS)
    group_price = group_price.ffill().dropna()
    group_price = group_price.loc[:, (group_price > 0).all(axis=0)]

    if group_price.shape[1] <= 1:
        return pd.DataFrame()

    log_returns = np.log(group_price / group_price.shift(1)).dropna()
    screening_records: list[dict[str, object]] = []

    for col_a, col_b in itertools.combinations(group_price.columns, 2):
        pair_price = group_price[[col_a, col_b]].dropna().copy()
        if len(pair_price) < MIN_OBS:
            continue

        pair_returns = log_returns[[col_a, col_b]].dropna() if col_a in log_returns and col_b in log_returns else pd.DataFrame()
        correlation = (
            float(pair_returns[col_a].corr(pair_returns[col_b]))
            if not pair_returns.empty and len(pair_returns) >= 2
            else np.nan
        )

        if corr_threshold > 0 and (pd.isna(correlation) or correlation < corr_threshold):
            continue

        y = pd.to_numeric(pair_price[col_a], errors="coerce")
        x = pd.to_numeric(pair_price[col_b], errors="coerce")
        valid = y.notna() & x.notna() & (y > 0) & (x > 0)
        y = y.loc[valid]
        x = x.loc[valid]

        if len(y) < MIN_OBS:
            continue

        try:
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            alpha = float(model.params.iloc[0])
            beta = float(model.params.iloc[1])
            residuals = pd.Series(model.resid, index=y.index)
            r_squared = float(model.rsquared)
            dw = float(durbin_watson(residuals))
            condition_number = float(model.condition_number)
        except Exception:
            continue

        adf_stat, adf_p_val, crit_1, crit_5, crit_10 = adf_test(
            residuals,
            maxlag=0,
            autolag=None,
            return_critical=True,
        )
        half_life = estimate_half_life(residuals)
        trend_slope, trend_pvalue, trend_strength = trend_strength_test(residuals)
        crossings, crossings_per_year = count_crossings(residuals)

        code_a, name_a = col_a.split("_", 1)
        code_b, name_b = col_b.split("_", 1)

        record = {
            "group": industry,
            "stock_A_code": code_a,
            "stock_A_name": name_a,
            "stock_B_code": code_b,
            "stock_B_name": name_b,
            "correlation": correlation,
            "alpha": alpha,
            "beta_hedge_ratio": beta,
            "r_squared": r_squared,
            "durbin_watson": dw,
            "condition_number": condition_number,
            "spread_method": "misbahan_ols_raw_close_residual",
            "spread_mean": float(residuals.mean()),
            "spread_std": float(residuals.std()),
            "spread_min": float(residuals.min()),
            "spread_max": float(residuals.max()),
            "adf_stat": float(adf_stat),
            "adf_pvalue": float(adf_p_val),
            "crit_1pct": float(crit_1),
            "crit_5pct": float(crit_5),
            "crit_10pct": float(crit_10),
            "adf_pass": bool(pd.notna(adf_p_val) and adf_p_val < adf_pvalue),
            "r2_pass": bool(pd.notna(r_squared) and r_squared > r2_threshold),
            "half_life": float(half_life) if pd.notna(half_life) else np.nan,
            "half_life_reasonable": bool(pd.notna(half_life) and MIN_HALF_LIFE <= half_life <= MAX_HALF_LIFE),
            "trend_slope": float(trend_slope),
            "trend_pvalue": float(trend_pvalue),
            "trend_strength": float(trend_strength),
            "no_obvious_trend": bool(pd.notna(trend_strength) and trend_strength < MAX_TREND_STRENGTH),
            "mean_crossings": int(crossings),
            "crossings_per_year": float(crossings_per_year),
            "enough_crossings": bool(pd.notna(crossings_per_year) and crossings_per_year >= MIN_CROSSINGS_PER_YEAR),
            "start_date": residuals.index.min(),
            "end_date": residuals.index.max(),
            "observations": int(len(residuals)),
        }
        screening_records.append(record)

    screening_df = pd.DataFrame(screening_records)
    if screening_df.empty:
        return pd.DataFrame()

    screening_df["score"] = screening_df.apply(score_pair, axis=1)

    # Keep MisbahAN-style selection primary: R² + residual ADF.
    # Half-life / trend / crossings remain diagnostic fields, not hard filters.
    screening_df["suitable"] = screening_df["r2_pass"] & screening_df["adf_pass"]

    screening_df = screening_df.sort_values(
        ["suitable", "score", "adf_pvalue", "r_squared", "condition_number"],
        ascending=[False, False, True, False, True],
    ).reset_index(drop=True)

    selected = screening_df[screening_df["suitable"]].copy()
    if selected.empty:
        selected = screening_df[(screening_df["r2_pass"]) | (screening_df["adf_pass"])].copy()
    if selected.empty:
        selected = screening_df.copy()

    selected = selected.head(top_n).reset_index(drop=True)
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    return selected


def determine_strategy2_test_start(
    close_df: pd.DataFrame,
    a_code: str,
    b_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    formation_ratio: float,
    lookback: int,
) -> pd.Timestamp:
    pair_close = close_df.loc[(close_df.index >= start) & (close_df.index <= end), [a_code, b_code]].dropna()
    if len(pair_close) <= lookback + 10:
        raise ValueError("資料長度不足，請延長回測期間或降低 Rolling OLS lookback。")

    split_idx = int(len(pair_close) * formation_ratio)
    split_idx = max(lookback, min(split_idx, len(pair_close) - 5))
    if split_idx <= 5 or split_idx >= len(pair_close) - 1:
        raise ValueError("formation/test 切分後資料不足，請調整 Formation ratio 或日期區間。")

    return pd.Timestamp(pair_close.index[split_idx])


def analyze_strategy2_cointegration_pair(
    close_df: pd.DataFrame,
    a_code: str,
    b_code: str,
    start: pd.Timestamp,
    test_start: pd.Timestamp,
    adf_pvalue: float,
    r2_threshold: float,
) -> dict[str, object]:
    formation_close = close_df.loc[(close_df.index >= start) & (close_df.index < test_start), [a_code, b_code]].dropna()
    if len(formation_close) < MIN_OBS:
        raise ValueError("formation period 資料不足，無法進行 cointegration diagnostics。")

    y = formation_close[a_code]
    x = formation_close[b_code]
    model = sm.OLS(y, sm.add_constant(x)).fit()
    residuals = pd.Series(model.resid, index=formation_close.index)

    returns = np.log(formation_close / formation_close.shift(1)).dropna()
    correlation = float(returns[a_code].corr(returns[b_code])) if len(returns) else np.nan

    alpha = float(model.params.iloc[0])
    beta = float(model.params.iloc[1])
    r_squared = float(model.rsquared)
    dw = float(durbin_watson(residuals))
    condition_number = float(model.condition_number)

    adf_stat, adf_p, crit_1, crit_5, crit_10 = adf_test(
        residuals,
        maxlag=0,
        autolag=None,
        return_critical=True,
    )
    half_life = estimate_half_life(residuals)
    trend_slope, trend_pvalue, trend_strength = trend_strength_test(residuals)
    crossings, crossings_per_year = count_crossings(residuals)

    row = pd.Series(
        {
            "adf_pvalue": adf_p,
            "r_squared": r_squared,
            "durbin_watson": dw,
            "condition_number": condition_number,
            "half_life": half_life,
            "trend_strength": trend_strength,
            "crossings_per_year": crossings_per_year,
            "correlation": correlation,
        }
    )

    adf_pass = bool(pd.notna(adf_p) and adf_p < adf_pvalue)
    r2_pass = bool(pd.notna(r_squared) and r_squared > r2_threshold)

    return {
        "formation_start": formation_close.index.min(),
        "formation_end": formation_close.index.max(),
        "test_start": test_start,
        "observations": int(len(formation_close)),
        "correlation": correlation,
        "alpha": alpha,
        "beta_hedge_ratio": beta,
        "r_squared": r_squared,
        "durbin_watson": dw,
        "condition_number": condition_number,
        "spread_mean": float(residuals.mean()),
        "spread_std": float(residuals.std(ddof=1)),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "crit_1pct": float(crit_1),
        "crit_5pct": float(crit_5),
        "crit_10pct": float(crit_10),
        "adf_pass": adf_pass,
        "r2_pass": r2_pass,
        "half_life": float(half_life) if pd.notna(half_life) else np.nan,
        "trend_slope": float(trend_slope),
        "trend_pvalue": float(trend_pvalue),
        "trend_strength": float(trend_strength),
        "mean_crossings": int(crossings),
        "crossings_per_year": float(crossings_per_year),
        "score": score_pair(row),
        "suitable": bool(adf_pass and r2_pass),
    }


def show_strategy2_diagnostics(
    diagnostics: dict[str, object],
    a_code: str,
    b_code: str,
    a_name: str,
    b_name: str,
) -> None:
    st.subheader("Formation Period Cointegration Diagnostics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test start", pd.Timestamp(diagnostics["test_start"]).strftime("%Y-%m-%d"))
    c2.metric("R²", num(float(diagnostics["r_squared"])))
    c3.metric("ADF p-value", num(float(diagnostics["adf_pvalue"])))
    c4.metric("Suitable", "Yes" if diagnostics["suitable"] else "No")

    diagnostics_df = pd.DataFrame(
        [
            {
                "pair": f"{format_stock_label(a_code, a_name)} / {format_stock_label(b_code, b_name)}",
                "formation_start": pd.Timestamp(diagnostics["formation_start"]).strftime("%Y-%m-%d"),
                "formation_end": pd.Timestamp(diagnostics["formation_end"]).strftime("%Y-%m-%d"),
                "observations": diagnostics["observations"],
                "correlation": diagnostics["correlation"],
                "alpha": diagnostics["alpha"],
                "beta_hedge_ratio": diagnostics["beta_hedge_ratio"],
                "r_squared": diagnostics["r_squared"],
                "durbin_watson": diagnostics["durbin_watson"],
                "condition_number": diagnostics["condition_number"],
                "adf_stat": diagnostics["adf_stat"],
                "adf_pvalue": diagnostics["adf_pvalue"],
                "crit_1pct": diagnostics["crit_1pct"],
                "crit_5pct": diagnostics["crit_5pct"],
                "crit_10pct": diagnostics["crit_10pct"],
                "adf_pass": diagnostics["adf_pass"],
                "r2_pass": diagnostics["r2_pass"],
                "half_life": diagnostics["half_life"],
                "trend_strength": diagnostics["trend_strength"],
                "crossings_per_year": diagnostics["crossings_per_year"],
                "score": diagnostics["score"],
                "suitable": diagnostics["suitable"],
            }
        ]
    )

    for col in [
        "correlation",
        "alpha",
        "beta_hedge_ratio",
        "r_squared",
        "durbin_watson",
        "condition_number",
        "adf_stat",
        "adf_pvalue",
        "crit_1pct",
        "crit_5pct",
        "crit_10pct",
        "half_life",
        "trend_strength",
        "crossings_per_year",
    ]:
        diagnostics_df[col] = pd.to_numeric(diagnostics_df[col], errors="coerce").round(4)

    st.dataframe(diagnostics_df, use_container_width=True, hide_index=True)


@st.cache_data(ttl=3600, show_spinner=False)
def download_screening_prices(stocks: tuple[tuple[str, str], ...], period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    cache_dir = Path(__file__).resolve().parent / ".cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))

    price_dict: dict[str, pd.Series] = {}

    for code, name in stocks:
        series = pd.Series(dtype=float)
        for ticker in candidate_yahoo_tickers(code):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.redirect_stderr(io.StringIO()):
                        raw = yf.download(
                            tickers=ticker,
                            period=period,
                            interval=interval,
                            auto_adjust=True,
                            progress=False,
                            threads=False,
                        )

                series = extract_close_series_from_single_download(raw)
                series = pd.to_numeric(series, errors="coerce").dropna()
                series = series[series > 0]

                if len(series) >= MIN_OBS:
                    break
            except Exception:
                series = pd.Series(dtype=float)

        if len(series) >= MIN_OBS:
            price_dict[f"{code}_{name}"] = series

    price_df = pd.DataFrame(price_dict).sort_index()
    if not price_df.empty:
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None).normalize()
    return price_df.ffill().dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def download_ohlc(codes: list[str], start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    cache_dir = Path(__file__).resolve().parent / ".cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))

    clean_codes = sorted({c.strip().upper() for c in codes if c.strip()})
    open_series: dict[str, pd.Series] = {}
    close_series: dict[str, pd.Series] = {}

    for code in clean_codes:
        downloaded = pd.DataFrame()
        for ticker in candidate_yahoo_tickers(code):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.redirect_stderr(io.StringIO()):
                        raw = yf.download(
                            tickers=ticker,
                            start=start.strftime("%Y-%m-%d"),
                            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                            interval="1d",
                            auto_adjust=True,
                            progress=False,
                            threads=False,
                        )

                downloaded = extract_ohlc_from_single_download(raw)
                downloaded = downloaded.apply(pd.to_numeric, errors="coerce").dropna()
                downloaded = downloaded[(downloaded["Open"] > 0) & (downloaded["Close"] > 0)]

                if not downloaded.empty:
                    break
            except Exception:
                downloaded = pd.DataFrame()

        if downloaded.empty:
            raise ValueError(
                f"Yahoo Finance returned no price data for {code}. "
                f"Tried {', '.join(candidate_yahoo_tickers(code))}."
            )

        downloaded.index = pd.to_datetime(downloaded.index).tz_localize(None).normalize()
        open_series[code] = downloaded["Open"]
        close_series[code] = downloaded["Close"]

    open_df = pd.DataFrame(open_series).sort_index()
    close_df = pd.DataFrame(close_series).sort_index()

    valid = open_df.notna().all(axis=1) & close_df.notna().all(axis=1)
    open_df = normalize_index(open_df.loc[valid].sort_index())
    close_df = normalize_index(close_df.loc[valid].sort_index())

    if open_df.empty or close_df.empty:
        raise ValueError("Downloaded data has no complete Open/Close rows.")

    return open_df, close_df


@st.cache_data(ttl=3600, show_spinner=False)
def download_benchmark(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    import yfinance as yf

    try:
        raw = yf.download(
            tickers=ticker,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return pd.Series(dtype=float, name=ticker)

    if raw.empty:
        return pd.Series(dtype=float, name=ticker)

    close = extract_close_series_from_single_download(raw)
    close.name = ticker
    return normalize_index(pd.to_numeric(close, errors="coerce").dropna())


def candidate_yahoo_tickers(code: str) -> list[str]:
    code = str(code).strip().upper()
    if not code:
        return []
    if "." in code or code.startswith("^") or "-" in code or "=" in code:
        return [code]
    return [f"{code}.TW", f"{code}.TWO"]


def extract_close_series_from_single_download(raw: pd.DataFrame) -> pd.Series:
    if raw is None or raw.empty:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
            return close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close

        if "Close" in raw.columns.get_level_values(1):
            for col in raw.columns:
                if col[1] == "Close":
                    return raw[col]

        return pd.Series(dtype=float)

    if "Close" not in raw.columns:
        return pd.Series(dtype=float)

    close = raw["Close"]
    return close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close


def extract_ohlc_from_single_download(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        data: dict[str, pd.Series] = {}
        for field in ["Open", "Close"]:
            if field in raw.columns.get_level_values(0):
                obj = raw[field]
                data[field] = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
            elif field in raw.columns.get_level_values(1):
                for col in raw.columns:
                    if col[1] == field:
                        data[field] = raw[col]
                        break
        if "Open" in data and "Close" in data:
            return pd.DataFrame(data)
        return pd.DataFrame()

    if "Open" not in raw.columns or "Close" not in raw.columns:
        return pd.DataFrame()

    return raw[["Open", "Close"]].copy()


def get_group_price(price_df: pd.DataFrame, stock_list: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    cols = []
    for code, name in stock_list:
        col = f"{code}_{name}"
        if col in price_df.columns:
            cols.append(col)

    if not cols:
        return pd.DataFrame()

    group_price = price_df[cols].copy()
    group_price = group_price.dropna(axis=1, how="all")
    group_price = group_price.ffill().dropna()
    group_price = group_price.loc[:, (group_price > 0).all(axis=0)]
    return group_price


def normalize_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out


def run_backtest(
    open_df: pd.DataFrame,
    close_df: pd.DataFrame,
    benchmark: pd.Series,
    a_code: str,
    b_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: Config,
) -> dict[str, object]:
    full_close = close_df.loc[close_df.index <= end].copy()
    full_open = open_df.loc[open_df.index <= end].copy()

    if len(full_close) <= config.lookback + 5:
        raise ValueError("The selected date range is too short for this lookback.")

    signals = rolling_spread_signal(full_close, a_code, b_code, config.lookback)
    latest_signal = latest_spread_signal(full_close, a_code, b_code, config.lookback)
    signals = signals.loc[(signals.index >= start) & (signals.index <= end)].copy()

    trades = backtest_pair(signals, full_open, full_close, a_code, b_code, config)
    equity = build_equity(trades, full_close, start, config.capital)

    equity["daily_pnl"] = equity["cumulative_pnl"].diff().fillna(equity["cumulative_pnl"])
    equity["daily_return"] = equity["daily_pnl"] / config.capital
    equity["drawdown"] = equity["equity"] / equity["equity"].cummax() - 1

    comparison = return_comparison(equity, benchmark, config.capital)
    summary = summarize(equity, trades, config.capital)
    price_history = full_close.loc[(full_close.index >= start) & (full_close.index <= end), [a_code, b_code]]
    weight_cols = ["entry_date", "a_weight", "b_weight", "a_sign", "b_sign", "direction", "weight_method"]
    weights = trades[weight_cols].copy() if not trades.empty else pd.DataFrame(columns=weight_cols)

    return {
        "signals": signals,
        "latest_signal": latest_signal,
        "trades": trades,
        "equity": equity,
        "comparison": comparison,
        "summary": summary,
        "price_history": price_history,
        "weights": weights,
    }


def rolling_spread_signal(close_df: pd.DataFrame, a_code: str, b_code: str, lookback: int) -> pd.DataFrame:
    logp = np.log(close_df[[a_code, b_code]].dropna())
    rows: list[dict[str, object]] = []

    for i in range(lookback, len(logp) - 1):
        signal_date = logp.index[i]
        exec_date = logp.index[i + 1]
        train = logp.iloc[i - lookback : i]

        alpha, beta = ols(train[a_code].to_numpy(), train[b_code].to_numpy())
        train_spread = train[a_code].to_numpy() - (alpha + beta * train[b_code].to_numpy())
        spread_mean = float(train_spread.mean())
        spread_std = float(train_spread.std(ddof=1))

        if not np.isfinite(spread_std) or spread_std == 0:
            continue

        spread = float(logp.iloc[i][a_code] - alpha - beta * logp.iloc[i][b_code])
        zscore = (spread - spread_mean) / spread_std

        rows.append(
            {
                "signal_date": signal_date,
                "exec_date": exec_date,
                "alpha": alpha,
                "beta": beta,
                "spread": spread,
                "spread_mean": spread_mean,
                "spread_std": spread_std,
                "zscore": zscore,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["exec_date", "alpha", "beta", "spread", "spread_mean", "spread_std", "zscore", "ma5"]).rename_axis("signal_date")

    out = pd.DataFrame(rows).set_index("signal_date")
    out["ma5"] = out["spread"].rolling(MA_WINDOW).mean()
    return out


def latest_spread_signal(close_df: pd.DataFrame, a_code: str, b_code: str, lookback: int) -> dict[str, object]:
    logp = np.log(close_df[[a_code, b_code]].dropna())
    if len(logp) <= lookback:
        return {}

    rows: list[dict[str, object]] = []

    for i in range(lookback, len(logp)):
        signal_date = logp.index[i]
        train = logp.iloc[i - lookback : i]

        alpha, beta = ols(train[a_code].to_numpy(), train[b_code].to_numpy())
        train_spread = train[a_code].to_numpy() - (alpha + beta * train[b_code].to_numpy())
        spread_mean = float(train_spread.mean())
        spread_std = float(train_spread.std(ddof=1))

        if not np.isfinite(spread_std) or spread_std == 0:
            continue

        spread = float(logp.iloc[i][a_code] - alpha - beta * logp.iloc[i][b_code])
        zscore = (spread - spread_mean) / spread_std

        rows.append(
            {
                "signal_date": signal_date,
                "alpha": float(alpha),
                "beta": float(beta),
                "spread": spread,
                "spread_mean": spread_mean,
                "spread_std": spread_std,
                "zscore": float(zscore),
            }
        )

    if not rows:
        return {}

    latest_df = pd.DataFrame(rows).set_index("signal_date")
    latest_df["ma5"] = latest_df["spread"].rolling(MA_WINDOW).mean()

    latest_date = latest_df.index[-1]
    row = latest_df.iloc[-1]

    return {
        "signal_date": latest_date,
        "alpha": float(row["alpha"]),
        "beta": float(row["beta"]),
        "spread": float(row["spread"]),
        "spread_mean": float(row["spread_mean"]),
        "spread_std": float(row["spread_std"]),
        "ma5": float(row["ma5"]) if pd.notna(row["ma5"]) else np.nan,
        "zscore": float(row["zscore"]),
        "a_close": float(close_df.loc[latest_date, a_code]),
        "b_close": float(close_df.loc[latest_date, b_code]),
    }


def backtest_pair(
    signals: pd.DataFrame,
    open_df: pd.DataFrame,
    close_df: pd.DataFrame,
    a_code: str,
    b_code: str,
    config: Config,
) -> pd.DataFrame:
    trades: list[dict[str, object]] = []
    open_positions: list[dict[str, object]] = []

    for signal_date, row in signals.iterrows():
        exec_date = row["exec_date"]
        if exec_date not in open_df.index:
            continue

        a_price = float(open_df.loc[exec_date, a_code])
        b_price = float(open_df.loc[exec_date, b_code])
        z = float(row["zscore"])
        spread = float(row["spread"])
        ma5 = float(row["ma5"]) if "ma5" in row and pd.notna(row["ma5"]) else np.nan
        beta = float(row["beta"])

        if not np.isfinite(beta) or beta <= 0:
            continue

        still_open: list[dict[str, object]] = []
        for position in open_positions:
            stop_loss = should_stop_loss(str(position["direction"]), z, config.stop_z)
            normal_exit = should_exit(str(position["direction"]), z, config.exit_z)

            if stop_loss or normal_exit:
                exit_cost = transaction_cost(
                    float(position["a_shares"]) * a_price,
                    "sell" if position["a_sign"] == 1 else "buy",
                    config,
                ) + transaction_cost(
                    float(position["b_shares"]) * b_price,
                    "sell" if position["b_sign"] == 1 else "buy",
                    config,
                )

                pnl = trade_pnl(position, a_price, b_price) - exit_cost
                position.update(
                    {
                        "exit_signal_date": signal_date,
                        "exit_date": exec_date,
                        "exit_zscore": z,
                        "exit_spread": spread,
                        "exit_ma5": ma5,
                        "a_exit_price": a_price,
                        "b_exit_price": b_price,
                        "exit_cost": exit_cost,
                        "total_cost": float(position["entry_cost"]) + exit_cost,
                        "holding_days": (pd.Timestamp(exec_date) - pd.Timestamp(position["entry_date"])).days,
                        "pnl": pnl,
                        "return_on_gross_exposure": pnl / float(position["gross_exposure"]),
                        "exit_reason": "z_stop_loss" if stop_loss else "normal_exit",
                        "status": "closed",
                    }
                )
                trades.append(position)
            else:
                still_open.append(position)

        open_positions = still_open

        direction, a_sign, b_sign = entry_direction(z, spread, ma5, config.entry_z)
        if direction is None:
            continue

        if is_stop_zone(z, config.stop_z):
            continue

        same_direction_open_count = sum(1 for position in open_positions if str(position["direction"]) == direction)
        if same_direction_open_count >= MAX_OPEN_POSITIONS_PER_DIRECTION:
            continue

        weights = choose_weights(beta)
        a_notional = config.capital * float(weights["a_weight"])
        b_notional = config.capital * float(weights["b_weight"])
        a_shares = int(a_notional // a_price) if config.integer_shares else a_notional / a_price
        b_shares = int(b_notional // b_price) if config.integer_shares else b_notional / b_price

        if a_shares <= 0 or b_shares <= 0:
            continue

        a_entry_notional = float(a_shares * a_price)
        b_entry_notional = float(b_shares * b_price)
        entry_cost = transaction_cost(a_entry_notional, "buy" if a_sign == 1 else "sell", config) + transaction_cost(
            b_entry_notional,
            "buy" if b_sign == 1 else "sell",
            config,
        )

        open_positions.append(
            {
                "a_code": a_code,
                "b_code": b_code,
                "direction": direction,
                "entry_signal_date": signal_date,
                "entry_date": exec_date,
                "entry_zscore": z,
                "entry_spread": spread,
                "entry_ma5": ma5,
                "entry_beta": beta,
                "a_sign": a_sign,
                "b_sign": b_sign,
                "a_entry_price": a_price,
                "b_entry_price": b_price,
                "a_shares": a_shares,
                "b_shares": b_shares,
                "a_entry_notional": a_entry_notional,
                "b_entry_notional": b_entry_notional,
                "gross_exposure": a_entry_notional + b_entry_notional,
                "entry_cost": entry_cost,
                "entry_layer": same_direction_open_count + 1,
                **weights,
            }
        )

    if open_positions:
        last_date = close_df.index[-1]
        for position in open_positions:
            pnl = trade_pnl(position, float(close_df.loc[last_date, a_code]), float(close_df.loc[last_date, b_code]))
            position.update(
                {
                    "exit_signal_date": pd.NaT,
                    "exit_date": pd.NaT,
                    "exit_zscore": np.nan,
                    "a_exit_price": np.nan,
                    "b_exit_price": np.nan,
                    "exit_cost": 0.0,
                    "total_cost": float(position["entry_cost"]),
                    "holding_days": (last_date - pd.Timestamp(position["entry_date"])).days,
                    "pnl": pnl,
                    "return_on_gross_exposure": pnl / float(position["gross_exposure"]),
                    "exit_reason": "still_open",
                    "status": "open",
                }
            )
            trades.append(position)

    return pd.DataFrame(trades)


def build_equity(trades: pd.DataFrame, close_df: pd.DataFrame, start: pd.Timestamp, capital: float) -> pd.DataFrame:
    dates = close_df.loc[close_df.index >= start].index
    pnl_curve = pd.Series(0.0, index=dates, name="cumulative_pnl")

    if trades.empty:
        return pd.DataFrame({"cumulative_pnl": pnl_curve, "equity": capital + pnl_curve})

    for _, tr in trades.iterrows():
        entry_date = pd.Timestamp(tr["entry_date"])
        a_code = str(tr["a_code"])
        b_code = str(tr["b_code"])

        if tr["status"] == "closed":
            exit_date = pd.Timestamp(tr["exit_date"])
            active = dates[(dates >= entry_date) & (dates < exit_date)]
            for day in active:
                pnl_curve.loc[day] += trade_pnl(tr, float(close_df.loc[day, a_code]), float(close_df.loc[day, b_code]))
            pnl_curve.loc[dates >= exit_date] += float(tr["pnl"])
        else:
            active = dates[dates >= entry_date]
            for day in active:
                pnl_curve.loc[day] += trade_pnl(tr, float(close_df.loc[day, a_code]), float(close_df.loc[day, b_code]))

    return pd.DataFrame({"cumulative_pnl": pnl_curve, "equity": capital + pnl_curve})


def return_comparison(equity: pd.DataFrame, benchmark: pd.Series, capital: float) -> pd.DataFrame:
    strategy = equity["equity"] / capital - 1
    if benchmark.empty:
        return pd.DataFrame({"strategy_cumulative_return": strategy})

    bench = benchmark.reindex(strategy.index).ffill().dropna()
    strategy = strategy.reindex(bench.index).ffill().fillna(0)
    return pd.DataFrame(
        {
            "strategy_cumulative_return": strategy,
            "benchmark_cumulative_return": bench / bench.iloc[0] - 1,
        }
    )


def summarize(equity: pd.DataFrame, trades: pd.DataFrame, capital: float) -> dict[str, float]:
    returns = equity["daily_return"].replace([np.inf, -np.inf], np.nan).dropna()
    ending = float(equity["equity"].iloc[-1])
    closed = trades[trades["status"].eq("closed")] if not trades.empty and "status" in trades.columns else pd.DataFrame()
    wins = closed[closed["pnl"] > 0] if not closed.empty and "pnl" in closed.columns else pd.DataFrame()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)

    return {
        "total_return": ending / capital - 1,
        "ending_equity": ending,
        "cagr": (ending / capital) ** (1 / years) - 1,
        "volatility": returns.std(ddof=1) * np.sqrt(252) if len(returns) else np.nan,
        "sharpe": sharpe(returns),
        "max_drawdown": float(equity["drawdown"].min()),
        "trade_count": float(len(trades)),
        "win_rate": float(len(wins) / len(closed)) if len(closed) else np.nan,
        "total_pnl": ending - capital,
    }


def show_summary(summary: dict[str, float]) -> None:
    metrics = [
        ("Total Return", pct(summary["total_return"])),
        ("Sharpe", num(summary["sharpe"])),
        ("Max DD", pct(summary["max_drawdown"])),
        ("CAGR", pct(summary["cagr"])),
        ("Win Rate", pct(summary["win_rate"])),
        ("Trades", f"{int(summary['trade_count'])}"),
    ]
    for i in range(0, len(metrics), 3):
        cols = st.columns(3)
        for col, (label, value) in zip(cols, metrics[i : i + 3]):
            col.metric(label, value)


def show_latest_trading_signal(
    result: dict[str, object],
    a_code: str,
    b_code: str,
    a_name: str,
    b_name: str,
    config: Config,
) -> None:
    latest_signal: dict[str, object] = result.get("latest_signal", {})  # type: ignore[assignment]
    trades: pd.DataFrame = result["trades"]  # type: ignore[assignment]

    st.subheader("Latest Trading Signal")

    if not latest_signal:
        st.info("目前資料不足，無法產生最新交易訊號。")
        return

    latest_date = pd.Timestamp(latest_signal["signal_date"])
    latest_z = float(latest_signal["zscore"])
    latest_spread = float(latest_signal["spread"])
    latest_ma5 = float(latest_signal["ma5"]) if "ma5" in latest_signal and pd.notna(latest_signal["ma5"]) else np.nan
    latest_beta = float(latest_signal["beta"])
    a_close = float(latest_signal["a_close"])
    b_close = float(latest_signal["b_close"])

    open_positions = trades[trades["status"].eq("open")].copy() if not trades.empty and "status" in trades.columns else pd.DataFrame()
    direction, a_sign, b_sign = entry_direction(latest_z, latest_spread, latest_ma5, config.entry_z)
    in_stop_zone = is_stop_zone(latest_z, config.stop_z)

    if direction is None:
        signal_text = "無新進場訊號"
        layer_text = "無"
        a_action = "無"
        b_action = "無"
        a_signed_weight = np.nan
        b_signed_weight = np.nan
        a_suggested_shares = 0
        b_suggested_shares = 0
    elif in_stop_zone:
        signal_text = f"有 {direction} 訊號，但 z-score 已進入停損區，不開新倉"
        layer_text = "不開倉"
        a_action = "無"
        b_action = "無"
        a_signed_weight = np.nan
        b_signed_weight = np.nan
        a_suggested_shares = 0
        b_suggested_shares = 0
    else:
        same_direction_open_count = 0
        if not open_positions.empty:
            same_direction_open_count = int((open_positions["direction"].astype(str) == direction).sum())

        if same_direction_open_count >= MAX_OPEN_POSITIONS_PER_DIRECTION:
            signal_text = f"有 {direction} 訊號，但同方向已滿 {MAX_OPEN_POSITIONS_PER_DIRECTION} 層，不再加碼"
            layer_text = f"已滿 {MAX_OPEN_POSITIONS_PER_DIRECTION} 層"
            a_action = "無"
            b_action = "無"
            a_signed_weight = np.nan
            b_signed_weight = np.nan
            a_suggested_shares = 0
            b_suggested_shares = 0
        else:
            weights = choose_weights(latest_beta)
            a_signed_weight = float(weights["a_weight"]) * a_sign
            b_signed_weight = float(weights["b_weight"]) * b_sign
            a_notional = config.capital * float(weights["a_weight"])
            b_notional = config.capital * float(weights["b_weight"])
            a_suggested_shares = int(a_notional // a_close) if config.integer_shares else a_notional / a_close
            b_suggested_shares = int(b_notional // b_close) if config.integer_shares else b_notional / b_close
            layer_text = f"第 {same_direction_open_count + 1} 層 / 最多 {MAX_OPEN_POSITIONS_PER_DIRECTION} 層"
            signal_text = f"新進場訊號：{direction}"
            a_action = action_text(a_sign)
            b_action = action_text(b_sign)

    if open_positions.empty:
        stop_loss_text = "無持倉"
        exit_text = "無持倉"
    else:
        stop_loss_count = int(open_positions["direction"].astype(str).apply(lambda d: should_stop_loss(d, latest_z, config.stop_z)).sum())
        exit_count = int(open_positions["direction"].astype(str).apply(lambda d: should_exit(d, latest_z, config.exit_z)).sum())
        stop_loss_text = "是" if stop_loss_count > 0 else "否"
        exit_text = "是" if exit_count > 0 else "否"

    render_latest_cards(
        [
            ("最新日期", latest_date.strftime("%Y-%m-%d")),
            ("最新 z-score", f"{latest_z:.2f}"),
            ("是否停損", stop_loss_text),
            ("是否出場", exit_text),
        ]
    )

    signal_df = pd.DataFrame(
        [
            {
                "今日訊號": signal_text,
                "A 標的": format_stock_label(a_code, a_name),
                "A 買賣方向": a_action,
                "A signed weight": signed_pct(a_signed_weight),
                "A 建議股數": format_shares(a_suggested_shares),
                "B 標的": format_stock_label(b_code, b_name),
                "B 買賣方向": b_action,
                "B signed weight": signed_pct(b_signed_weight),
                "B 建議股數": format_shares(b_suggested_shares),
                "目前第幾層": layer_text,
                "估算價格基準": "最新 Close；實際交易仍以隔天 Open 為準",
            }
        ]
    )
    st.dataframe(signal_df, use_container_width=True, hide_index=True)

    st.markdown("#### 目前理論持倉狀況")
    if open_positions.empty:
        st.info("截至最新資料，目前沒有未平倉部位。")
        return

    open_positions["a_signed_shares"] = open_positions["a_shares"].astype(float) * open_positions["a_sign"].astype(float)
    open_positions["b_signed_shares"] = open_positions["b_shares"].astype(float) * open_positions["b_sign"].astype(float)
    open_positions["a_signed_weight"] = open_positions["a_weight"].astype(float) * open_positions["a_sign"].astype(float)
    open_positions["b_signed_weight"] = open_positions["b_weight"].astype(float) * open_positions["b_sign"].astype(float)
    open_positions["current_zscore"] = latest_z
    open_positions["current_action"] = open_positions["direction"].astype(str).apply(
        lambda d: "停損出場" if should_stop_loss(d, latest_z, config.stop_z) else ("正常出場" if should_exit(d, latest_z, config.exit_z) else "續抱")
    )

    total_a_signed_shares = float(open_positions["a_signed_shares"].sum())
    total_b_signed_shares = float(open_positions["b_signed_shares"].sum())
    total_open_pnl = float(open_positions["pnl"].sum()) if "pnl" in open_positions.columns else np.nan

    render_latest_cards(
        [
            ("未平倉筆數", f"{len(open_positions)}"),
            (f"{a_code} 淨股數", format_shares(total_a_signed_shares)),
            (f"{b_code} 淨股數", format_shares(total_b_signed_shares)),
            ("未實現 P&L", num(total_open_pnl)),
        ]
    )

    display_cols = [
        "direction",
        "entry_layer",
        "entry_date",
        "entry_zscore",
        "current_zscore",
        "a_signed_shares",
        "b_signed_shares",
        "a_signed_weight",
        "b_signed_weight",
        "pnl",
        "holding_days",
        "current_action",
    ]
    position_df = open_positions[[c for c in display_cols if c in open_positions.columns]].copy()
    for col in ["entry_zscore", "current_zscore", "a_signed_weight", "b_signed_weight", "pnl"]:
        if col in position_df.columns:
            position_df[col] = position_df[col].astype(float).round(4)
    st.dataframe(position_df, use_container_width=True, hide_index=True)


def show_charts(result: dict[str, object], a_code: str, b_code: str, benchmark_name: str, config: Config) -> None:
    price_history: pd.DataFrame = result["price_history"]  # type: ignore[assignment]
    signals: pd.DataFrame = result["signals"]  # type: ignore[assignment]
    comparison: pd.DataFrame = result["comparison"]  # type: ignore[assignment]
    equity: pd.DataFrame = result["equity"]  # type: ignore[assignment]
    trades: pd.DataFrame = result["trades"]  # type: ignore[assignment]
    weights: pd.DataFrame = result["weights"]  # type: ignore[assignment]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_history.index, y=price_history[a_code], name=a_code, mode="lines"))
    fig.add_trace(go.Scatter(x=price_history.index, y=price_history[b_code], name=b_code, mode="lines"))
    fig.update_layout(title="Historical Close Price", template="plotly_white", height=520, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    fig = spread_chart(signals, config)
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comparison.index, y=comparison["strategy_cumulative_return"], name="Strategy", mode="lines"))
    if "benchmark_cumulative_return" in comparison:
        fig.add_trace(go.Scatter(x=comparison.index, y=comparison["benchmark_cumulative_return"], name=benchmark_name, mode="lines"))
    fig.update_layout(title="Cumulative Return vs Benchmark", template="plotly_white", height=520, yaxis_tickformat=".1%", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity["drawdown"], name="Drawdown", mode="lines", fill="tozeroy"))
    fig.update_layout(title="Drawdown", template="plotly_white", height=520, yaxis_tickformat=".1%", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    fig = trade_chart(trades)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    fig = weight_chart(weights)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def spread_chart(signals: pd.DataFrame, config: Config) -> go.Figure:
    fig = go.Figure()
    if signals.empty:
        fig.update_layout(title="Spread Signal", template="plotly_white", height=370)
        return fig

    upper = signals["spread_mean"] + config.entry_z * signals["spread_std"]
    lower = signals["spread_mean"] - config.entry_z * signals["spread_std"]
    ma5 = signals["ma5"] if "ma5" in signals.columns else pd.Series(index=signals.index, dtype=float)
    long_signals = signals[(signals["zscore"] <= -config.entry_z) & (signals["spread"] < ma5)]
    short_signals = signals[(signals["zscore"] >= config.entry_z) & (signals["spread"] > ma5)]

    fig.add_trace(go.Scatter(x=signals.index, y=signals["spread"], name="spread", mode="lines"))
    fig.add_trace(go.Scatter(x=signals.index, y=signals["spread_mean"], name="rolling mean", mode="lines"))
    if "ma5" in signals.columns:
        fig.add_trace(go.Scatter(x=signals.index, y=signals["ma5"], name="5MA", mode="lines"))
    fig.add_trace(go.Scatter(x=signals.index, y=upper, name="+entry band", mode="lines", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=signals.index, y=lower, name="-entry band", mode="lines", line=dict(dash="dash")))

    if not short_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=short_signals.index,
                y=short_signals["spread"],
                name="short signal",
                mode="markers",
                marker=dict(symbol="triangle-down", size=11),
            )
        )

    if not long_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=long_signals.index,
                y=long_signals["spread"],
                name="long signal",
                mode="markers",
                marker=dict(symbol="triangle-up", size=11),
            )
        )

    fig.update_layout(title="Spread Signal", template="plotly_white", height=370, hovermode="x unified")
    return fig


def trade_chart(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades.empty:
        fig.update_layout(title="Trade P&L", template="plotly_white", height=320)
        return fig

    labels = pd.to_datetime(trades["entry_date"]).dt.strftime("%Y-%m-%d") + " " + trades["direction"].astype(str)
    colors = np.where(trades["pnl"] >= 0, "#0f766e", "#c2410c")
    fig.add_trace(go.Bar(x=labels, y=trades["pnl"], marker_color=colors, name="P&L"))
    fig.update_layout(title="Trade P&L", template="plotly_white", height=320)
    return fig


def weight_chart(weights: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if weights.empty:
        fig.update_layout(title="Signed Entry Weights", template="plotly_white", height=320)
        return fig

    signed_weights = weights.copy()
    signed_weights["a_signed_weight"] = signed_weights["a_weight"] * signed_weights["a_sign"]
    signed_weights["b_signed_weight"] = signed_weights["b_weight"] * signed_weights["b_sign"]

    fig.add_trace(
        go.Scatter(
            x=signed_weights["entry_date"],
            y=signed_weights["a_signed_weight"],
            name="A signed weight",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=signed_weights["entry_date"],
            y=signed_weights["b_signed_weight"],
            name="B signed weight",
            mode="lines+markers",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#6b7280")
    fig.update_layout(title="Signed Entry Weights", template="plotly_white", height=320, yaxis_tickformat=".0%", hovermode="x unified")
    return fig


def show_tables(result: dict[str, object]) -> None:
    trades: pd.DataFrame = result["trades"]  # type: ignore[assignment]
    signals: pd.DataFrame = result["signals"]  # type: ignore[assignment]

    st.subheader("交易明細")
    if trades.empty:
        st.warning("這段期間沒有觸發交易。")
    else:
        cols = [
            "direction",
            "entry_layer",
            "entry_date",
            "exit_date",
            "entry_zscore",
            "exit_zscore",
            "a_weight",
            "b_weight",
            "gross_exposure",
            "pnl",
            "return_on_gross_exposure",
            "holding_days",
            "exit_reason",
            "status",
        ]
        st.dataframe(trades[[c for c in cols if c in trades.columns]], use_container_width=True)
        st.download_button("Download trades CSV", trades.to_csv(index=False).encode("utf-8-sig"), "pair_trades.csv", "text/csv")

    with st.expander("Signal data"):
        st.dataframe(signals.tail(300), use_container_width=True)


def render_latest_cards(items: list[tuple[str, str]]) -> None:
    cards = ""
    for label, value in items:
        safe_label = html.escape(str(label))
        safe_value = html.escape(str(value))
        cards += (
            '<div class="latest-signal-card">'
            f'<div class="latest-signal-label">{safe_label}</div>'
            f'<div class="latest-signal-value">{safe_value}</div>'
            "</div>"
        )

    st.markdown(
        f"""
        <style>
        .latest-signal-grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 12px;
          margin: 0.4rem 0 1rem 0;
        }}
        .latest-signal-card {{
          background: #ffffff;
          border: 1px solid #e6e8ef;
          border-radius: 8px;
          padding: 14px 16px;
          box-shadow: 0 1px 2px rgba(20, 29, 47, 0.04);
          min-height: 92px;
        }}
        .latest-signal-label {{
          color: #344054;
          font-size: 0.95rem;
          font-weight: 600;
          line-height: 1.35;
          margin-bottom: 8px;
        }}
        .latest-signal-value {{
          color: #17202a;
          font-size: clamp(1.25rem, 2.4vw, 2.05rem);
          font-weight: 500;
          line-height: 1.15;
          white-space: normal;
          overflow-wrap: anywhere;
          word-break: break-word;
        }}
        </style>
        <div class="latest-signal-grid">
          {cards}
        </div>
        """,
        unsafe_allow_html=True,
    )


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(alpha), float(beta)


def adf_test(
    spread: pd.Series,
    maxlag: int | None = None,
    autolag: str | None = "AIC",
    return_critical: bool = False,
):
    s = pd.Series(spread).dropna()
    try:
        if maxlag is not None:
            result = adfuller(s, maxlag=maxlag, autolag=None)
        else:
            result = adfuller(s, autolag=autolag)

        adf_stat = float(result[0])
        pvalue = float(result[1])
        critical_values = result[4]

        if return_critical:
            return (
                adf_stat,
                pvalue,
                float(critical_values.get("1%", np.nan)),
                float(critical_values.get("5%", np.nan)),
                float(critical_values.get("10%", np.nan)),
            )

        return adf_stat, pvalue

    except Exception:
        if return_critical:
            return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        return float("nan"), float("nan")


def estimate_half_life(spread: pd.Series) -> float:
    s = pd.Series(spread).dropna()
    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()
    idx = lagged.index.intersection(delta.index)
    y = delta.loc[idx]
    x = lagged.loc[idx]

    if len(y) < 2:
        return float("nan")

    try:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta = model.params.iloc[1]
    except Exception:
        return float("nan")

    if beta >= 0:
        return float("nan")

    return float(-np.log(2) / beta)


def trend_strength_test(spread: pd.Series) -> tuple[float, float, float]:
    s = pd.Series(spread).dropna()
    if len(s) < 2:
        return float("nan"), float("nan"), float("inf")

    t = np.arange(len(s))
    try:
        X = sm.add_constant(t)
        model = sm.OLS(s.values, X).fit()
        slope = model.params[1]
        pvalue = model.pvalues[1]
    except Exception:
        return float("nan"), float("nan"), float("inf")

    total_change = abs(slope) * len(s)
    std = s.std()
    strength = np.inf if std == 0 else total_change / std
    return float(slope), float(pvalue), float(strength)


def count_crossings(spread: pd.Series) -> tuple[int, float]:
    s = pd.Series(spread).dropna()
    centered = s - s.mean()
    signs = np.sign(centered)
    crossings = int(np.sum(signs.shift(1) * signs < 0))
    years = len(s) / 252
    return crossings, float(crossings / years if years > 0 else np.nan)


def score_pair(row: pd.Series) -> int:
    score = 0

    r_squared = row.get("r_squared", np.nan)
    if pd.notna(r_squared):
        if r_squared >= 0.80:
            score += 30
        elif r_squared >= 0.65:
            score += 20
        elif r_squared >= 0.50:
            score += 12

    adf_pvalue = row.get("adf_pvalue", np.nan)
    if pd.notna(adf_pvalue):
        if adf_pvalue < 0.01:
            score += 35
        elif adf_pvalue < 0.05:
            score += 25
        elif adf_pvalue < 0.10:
            score += 10

    dw = row.get("durbin_watson", np.nan)
    if pd.notna(dw):
        # DW near 2 is generally preferred for less autocorrelation in residuals.
        if 1.5 <= dw <= 2.5:
            score += 10
        elif 1.0 <= dw <= 3.0:
            score += 5

    condition_number = row.get("condition_number", np.nan)
    if pd.notna(condition_number):
        if condition_number < 100:
            score += 8
        elif condition_number < 1000:
            score += 4

    hl = row.get("half_life", np.nan)
    if pd.notna(hl) and MIN_HALF_LIFE <= hl <= MAX_HALF_LIFE:
        score += 10
        if 5 <= hl <= 30:
            score += 5

    trend_strength = row.get("trend_strength", np.nan)
    if pd.notna(trend_strength):
        if trend_strength < 0.5:
            score += 8
        elif trend_strength < MAX_TREND_STRENGTH:
            score += 4

    crossings_per_year = row.get("crossings_per_year", np.nan)
    if pd.notna(crossings_per_year):
        if crossings_per_year >= 12:
            score += 5
        elif crossings_per_year >= MIN_CROSSINGS_PER_YEAR:
            score += 2

    correlation = row.get("correlation", np.nan)
    if pd.notna(correlation):
        if correlation >= 0.95:
            score += 5
        elif correlation >= 0.90:
            score += 3
        elif correlation >= 0.80:
            score += 1

    return score


def entry_direction(z: float, spread: float, ma5: float, entry_z: float) -> tuple[str | None, int, int]:
    if not np.isfinite(ma5):
        return None, 0, 0
    if z <= -entry_z and spread < ma5:
        return "long_spread", 1, -1
    if z >= entry_z and spread > ma5:
        return "short_spread", -1, 1
    return None, 0, 0


def should_exit(direction: str, z: float, exit_z: float) -> bool:
    return (direction == "long_spread" and z >= exit_z) or (direction == "short_spread" and z <= -exit_z)


def should_stop_loss(direction: str, z: float, stop_z: float) -> bool:
    return (direction == "long_spread" and z <= -stop_z) or (direction == "short_spread" and z >= stop_z)


def is_stop_zone(z: float, stop_z: float) -> bool:
    return abs(z) >= stop_z


def choose_weights(beta: float) -> dict[str, object]:
    beta_abs = abs(float(beta))
    if not np.isfinite(beta_abs) or beta_abs <= 0:
        beta_abs = 1.0

    a_weight = 1 / (1 + beta_abs)
    return {
        "a_weight": float(a_weight),
        "b_weight": float(1 - a_weight),
        "weight_method": "ols_beta",
    }


def transaction_cost(notional: float, side: str, config: Config) -> float:
    cost = abs(notional) * config.broker_fee
    if side == "sell":
        cost += abs(notional) * config.sell_tax
    return float(cost)


def trade_pnl(trade: pd.Series | dict[str, object], a_price: float, b_price: float) -> float:
    pnl_a = float(trade["a_sign"]) * float(trade["a_shares"]) * (a_price - float(trade["a_entry_price"]))
    pnl_b = float(trade["b_sign"]) * float(trade["b_shares"]) * (b_price - float(trade["b_entry_price"]))
    return float(pnl_a + pnl_b - float(trade["entry_cost"]))


def sharpe(returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return float("nan")
    std = clean.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return float("nan")
    return float(np.sqrt(252) * clean.mean() / std)


def format_stock_label(code: str, name: str) -> str:
    code = str(code).strip()
    name = str(name).strip()
    if code and name:
        return f"{code} {name}"
    return code or "尚未選擇"


def resolve_stock_name(code: str) -> str:
    code = str(code).strip().upper()
    if not code:
        return ""

    if code in KNOWN_NAMES:
        return KNOWN_NAMES[code]

    twse_name = fetch_twse_stock_name(code)
    if twse_name:
        return twse_name

    yahoo_name = fetch_yahoo_tw_stock_name(code)
    if yahoo_name:
        return yahoo_name

    return ""


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_twse_name_map() -> dict[str, str]:
    url = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return {}

    name_map: dict[str, str] = {}
    for row in data:
        code = str(row.get("Code", "")).strip()
        name = str(row.get("Name", "")).strip()
        if code and name:
            name_map[code] = name

    return name_map


def fetch_twse_stock_name(code: str) -> str:
    return fetch_twse_name_map().get(code, "")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_yahoo_tw_stock_name(code: str) -> str:
    for ticker in candidate_yahoo_tickers(code):
        url = f"https://tw.stock.yahoo.com/quote/{ticker}"
        try:
            response = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
        except Exception:
            continue

        text = response.text
        match = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue

        title = html.unescape(match.group(1))
        title = re.sub(r"\s+", " ", title).strip()

        if " (" in title:
            candidate = title.split(" (", 1)[0].strip()
            if candidate and candidate != code:
                return candidate

    return ""


def action_text(sign: int | float) -> str:
    if sign > 0:
        return "買進 / 做多"
    if sign < 0:
        return "賣出 / 放空"
    return "無"


def signed_pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "無"
    return f"{value:+.2%}"


def format_shares(value: float | int) -> str:
    if value is None or not np.isfinite(float(value)):
        return "無"
    if abs(float(value) - int(float(value))) < 1e-9:
        return f"{int(value):,}"
    return f"{float(value):,.2f}"


def pct(value: float) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.2%}"


def num(value: float) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.2f}"


if __name__ == "__main__":
    main()
