from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import html
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


PRESET_PAIRS = [
    ("3006", "晶豪科", "4967", "十銓"),
    ("2344", "華邦電", "2451", "創見"),
    ("2369", "菱生", "8150", "南茂"),
    ("3189", "景碩", "6271", "同欣電"),
    ("2329", "華泰", "2449", "京元電子"),
    ("2836", "高雄銀", "2838", "聯邦銀"),
    ("2886", "兆豐金", "2891", "中信金"),
    ("2880", "華南金", "2891", "中信金"),
    ("2801", "彰銀", "2838", "聯邦銀"),
    ("2886", "兆豐金", "2892", "第一金"),
]

KNOWN_NAMES = {
    "2886": "兆豐金", "2891": "中信金", "2880": "華南金", "2892": "第一金",
    "2836": "高雄銀", "2838": "聯邦銀", "2801": "彰銀", "2329": "華泰",
    "2449": "京元電子", "2369": "菱生", "8150": "南茂", "3189": "景碩",
    "6271": "同欣電", "3006": "晶豪科", "4967": "十銓", "2344": "華邦電", "2451": "創見",
}


@dataclass(frozen=True)
class Config:
    lookback: int
    entry_z: float
    exit_z: float
    capital: float
    broker_fee: float
    sell_tax: float
    integer_shares: bool
    weight_mode: str
    opt_lookback: int
    min_weight: float
    max_weight: float
    weight_step: float


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
      background: #ffffff;
      border: 1px solid #e6e8ef;
      border-radius: 12px;
      padding: 24px 26px;
      box-shadow: 0 1px 2px rgba(20, 29, 47, 0.04);
      margin-top: 16px;
      margin-bottom: 12px;
    }
    .strategy-card h3 {
      margin-top: 0;
      margin-bottom: 8px;
    }
    .strategy-card p {
      color: #607080;
      margin-bottom: 0;
      line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    if "selected_strategy" not in st.session_state:
        st.session_state["selected_strategy"] = None

    if st.session_state["selected_strategy"] is None:
        render_strategy_selector()
        return

    st.title("Trading Strategy Lab")
    st.caption("Pair trading backtest with rolling hedge ratio and Sharpe-based leg weights.")
    settings = sidebar_settings()
    render_backtest(settings)


def render_strategy_selector() -> None:
    st.title("Trading Strategy Lab")
    st.caption("請先選擇要使用的交易策略。")

    st.markdown(
        """
        <div class="strategy-card">
          <h3>策略1：Pair Trading 回測</h3>
          <p>
            使用 rolling OLS 估計 spread 與 z-score，並可選擇 OLS hedge ratio
            或 rolling max-Sharpe grid 作為雙邊部位權重。
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("進入策略1", type="primary", use_container_width=True):
            st.session_state["selected_strategy"] = "strategy_1"
            st.rerun()

    st.info("目前已開放策略1；後續可以在這個入口頁繼續新增策略2、策略3。")


def sidebar_settings() -> dict[str, object]:
    if st.sidebar.button("返回策略選擇"):
        st.session_state["selected_strategy"] = None
        st.rerun()

    st.sidebar.caption("目前策略：策略1 Pair Trading")
    st.sidebar.divider()

    st.sidebar.header("回測設定")
    mode = st.sidebar.radio("標的來源", ["Notebook preset", "自訂 pair"], horizontal=True)

    labels = [f"{a} {an} / {b} {bn}" for a, an, b, bn in PRESET_PAIRS]

    if mode == "Notebook preset":
        label = st.sidebar.selectbox("Pair", labels, index=6)
        a_code, a_name, b_code, b_name = PRESET_PAIRS[labels.index(label)]
    else:
        col_a, col_b = st.sidebar.columns(2)
        with col_a:
            a_code = st.text_input("A code", value="2886").strip().upper()
        with col_b:
            b_code = st.text_input("B code", value="2891").strip().upper()
        a_name = ""
        b_name = ""

    today = pd.Timestamp.today().date()
    start_default = (pd.Timestamp(today) - pd.DateOffset(years=2)).date()
    start = st.sidebar.date_input("Backtest start", value=start_default)
    end = st.sidebar.date_input("Backtest end", value=today)
    suffix = st.sidebar.text_input("Yahoo suffix", value=".TW")
    benchmark = st.sidebar.text_input("Benchmark", value="^TWII")

    if mode == "自訂 pair":
        a_name = resolve_stock_name(a_code, suffix)
        b_name = resolve_stock_name(b_code, suffix)
        st.sidebar.caption(
            f"辨識結果：{format_stock_label(a_code, a_name)} / {format_stock_label(b_code, b_name)}"
        )

    st.sidebar.divider()
    st.sidebar.header("策略參數")
    lookback = st.sidebar.slider("Rolling OLS lookback", 40, 260, 120, 10)
    entry_z = st.sidebar.slider("Entry z-score", 0.5, 4.0, 2.0, 0.1)
    exit_z = st.sidebar.slider("Exit z-score", -1.0, 1.0, 0.0, 0.1)
    capital = st.sidebar.number_input("Capital per pair", min_value=10_000, value=100_000, step=10_000)
    integer_shares = st.sidebar.toggle("整股交易", value=True)

    st.sidebar.divider()
    st.sidebar.header("權重最佳化")
    weight_label = st.sidebar.selectbox("部位方法", ["Rolling max-Sharpe grid", "OLS hedge ratio"])
    opt_lookback = st.sidebar.slider("Optimization lookback", 60, 360, 180, 20)
    min_w, max_w = st.sidebar.slider("單邊 gross weight 範圍", 0.10, 0.90, (0.20, 0.80), 0.05)
    weight_step = st.sidebar.select_slider("Grid step", options=[0.01, 0.025, 0.05, 0.10], value=0.05)

    st.sidebar.divider()
    st.sidebar.header("交易成本")
    broker_fee = st.sidebar.number_input("Broker fee", min_value=0.0, value=0.001425, step=0.0001, format="%.6f")
    sell_tax = st.sidebar.number_input("Sell tax", min_value=0.0, value=0.001425, step=0.0001, format="%.6f")

    return {
        "a_code": a_code,
        "a_name": a_name,
        "b_code": b_code,
        "b_name": b_name,
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "suffix": suffix,
        "benchmark": benchmark,
        "config": Config(
            lookback=lookback,
            entry_z=entry_z,
            exit_z=exit_z,
            capital=float(capital),
            broker_fee=float(broker_fee),
            sell_tax=float(sell_tax),
            integer_shares=integer_shares,
            weight_mode="max_sharpe" if weight_label == "Rolling max-Sharpe grid" else "ols",
            opt_lookback=opt_lookback,
            min_weight=float(min_w),
            max_weight=float(max_w),
            weight_step=float(weight_step),
        ),
    }


def format_stock_label(code: str, name: str) -> str:
    code = str(code).strip()
    name = str(name).strip()
    return f"{code} {name}" if name else code


def resolve_stock_name(code: str, suffix: str = ".TW") -> str:
    code = str(code).strip().upper()
    if not code:
        return ""

    if code in KNOWN_NAMES:
        return KNOWN_NAMES[code]

    twse_name = fetch_twse_stock_name(code)
    if twse_name:
        return twse_name

    yahoo_name = fetch_yahoo_tw_stock_name(code, suffix)
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
    name_map = fetch_twse_name_map()
    return name_map.get(code, "")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_yahoo_tw_stock_name(code: str, suffix: str = ".TW") -> str:
    ticker = to_yahoo_ticker(code, suffix)
    url = f"https://tw.stock.yahoo.com/quote/{ticker}"

    try:
        response = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
    except Exception:
        return ""

    text = response.text
    match = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""

    title = html.unescape(match.group(1))
    title = re.sub(r"\s+", " ", title).strip()

    # Yahoo Taiwan title usually looks like:
    # 兆豐金 (2886.TW) 走勢圖 - Yahoo奇摩股市
    if " (" in title:
        candidate = title.split(" (", 1)[0].strip()
        if candidate and candidate != code:
            return candidate

    return ""


def render_backtest(settings: dict[str, object]) -> None:
    a_code = str(settings["a_code"])
    b_code = str(settings["b_code"])
    a_name = str(settings["a_name"])
    b_name = str(settings["b_name"])
    start = pd.Timestamp(settings["start"])
    end = pd.Timestamp(settings["end"])
    config: Config = settings["config"]  # type: ignore[assignment]

    left, right = st.columns([0.68, 0.32])
    with left:
        st.subheader(f"{format_stock_label(a_code, a_name)} / {format_stock_label(b_code, b_name)}")
        st.write("訊號使用 t 日 Close；交易使用 t+1 日 Open。")
    with right:
        run = st.button("Run backtest", type="primary", use_container_width=True)

    if not run:
        st.info("調整左側設定後按 Run backtest。")
        return
    if start >= end:
        st.error("Backtest start 必須早於 end。")
        return

    try:
        with st.spinner("下載價格並執行回測..."):
            warmup = max(config.lookback * 3, config.opt_lookback * 2, 365)
            download_start = start - pd.DateOffset(days=warmup)
            open_df, close_df = download_ohlc([a_code, b_code], download_start, end, str(settings["suffix"]))
            benchmark = download_benchmark(str(settings["benchmark"]), download_start, end)
            result = run_backtest(open_df, close_df, benchmark, a_code, b_code, start, end, config)
    except Exception as exc:
        st.error(f"回測失敗：{exc}")
        return

    show_summary(result["summary"])
    show_charts(result, a_code, b_code, str(settings["benchmark"]), config)
    show_tables(result)


@st.cache_data(ttl=3600, show_spinner=False)
def download_ohlc(codes: list[str], start: pd.Timestamp, end: pd.Timestamp, suffix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    cache_dir = Path(__file__).resolve().parent / ".cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))

    clean_codes = sorted({c.strip().upper() for c in codes if c.strip()})
    tickers = [to_yahoo_ticker(code, suffix) for code in clean_codes]
    raw = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise ValueError("Yahoo Finance returned no price data.")

    open_df = extract_price_field(raw, "Open", clean_codes, suffix)
    close_df = extract_price_field(raw, "Close", clean_codes, suffix)
    valid = open_df.notna().all(axis=1) & close_df.notna().all(axis=1)
    open_df = normalize_index(open_df.loc[valid].sort_index())
    close_df = normalize_index(close_df.loc[valid].sort_index())
    if open_df.empty or close_df.empty:
        raise ValueError("Downloaded data has no complete Open/Close rows.")
    return open_df, close_df


@st.cache_data(ttl=3600, show_spinner=False)
def download_benchmark(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    import yfinance as yf

    raw = yf.download(
        tickers=ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = ticker
    return normalize_index(close.dropna())


def to_yahoo_ticker(code: str, suffix: str) -> str:
    if "." in code or code.startswith("^") or "-" in code or "=" in code:
        return code
    return f"{code}{suffix}"


def extract_price_field(raw: pd.DataFrame, field: str, codes: list[str], suffix: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            out = raw[field].copy()
        else:
            out = pd.concat(
                {ticker: raw[(ticker, field)] for ticker in raw.columns.get_level_values(0).unique() if (ticker, field) in raw.columns},
                axis=1,
            )
    else:
        out = raw[[field]].copy()
        out.columns = [codes[0]]

    out = out.rename(columns={to_yahoo_ticker(code, suffix): code for code in codes})
    missing = [code for code in codes if code not in out.columns]
    if missing:
        raise ValueError(f"Missing price fields for: {', '.join(missing)}")
    return out[codes].apply(pd.to_numeric, errors="coerce")


def normalize_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out


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
        rows.append({"signal_date": signal_date, "exec_date": exec_date, "alpha": alpha, "beta": beta, "spread": spread, "spread_mean": spread_mean, "spread_std": spread_std, "zscore": zscore})
    if not rows:
        return pd.DataFrame(columns=["exec_date", "alpha", "beta", "spread", "spread_mean", "spread_std", "zscore"]).rename_axis("signal_date")
    return pd.DataFrame(rows).set_index("signal_date")


def run_backtest(open_df: pd.DataFrame, close_df: pd.DataFrame, benchmark: pd.Series, a_code: str, b_code: str, start: pd.Timestamp, end: pd.Timestamp, config: Config) -> dict[str, object]:
    full_close = close_df.loc[close_df.index <= end].copy()
    full_open = open_df.loc[open_df.index <= end].copy()
    if len(full_close) <= config.lookback + 5:
        raise ValueError("The selected date range is too short for this lookback.")
    signals = rolling_spread_signal(full_close, a_code, b_code, config.lookback)
    signals = signals.loc[(signals.index >= start) & (signals.index <= end)].copy()
    trades = backtest_pair(signals, full_open, full_close, a_code, b_code, config)
    equity = build_equity(trades, full_close, start, config.capital)
    equity["daily_pnl"] = equity["cumulative_pnl"].diff().fillna(equity["cumulative_pnl"])
    equity["daily_return"] = equity["daily_pnl"] / config.capital
    equity["drawdown"] = equity["equity"] / equity["equity"].cummax() - 1
    comparison = return_comparison(equity, benchmark, config.capital)
    summary = summarize(equity, trades, config.capital)
    price_history = full_close.loc[(full_close.index >= start) & (full_close.index <= end), [a_code, b_code]]
    weight_cols = ["entry_date", "a_weight", "b_weight", "optimizer_sharpe", "weight_method"]
    weights = trades[weight_cols].copy() if not trades.empty else pd.DataFrame(columns=weight_cols)
    return {"signals": signals, "trades": trades, "equity": equity, "comparison": comparison, "summary": summary, "price_history": price_history, "weights": weights}


def backtest_pair(signals: pd.DataFrame, open_df: pd.DataFrame, close_df: pd.DataFrame, a_code: str, b_code: str, config: Config) -> pd.DataFrame:
    trades: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for signal_date, row in signals.iterrows():
        exec_date = row["exec_date"]
        if exec_date not in open_df.index:
            continue
        a_price = float(open_df.loc[exec_date, a_code])
        b_price = float(open_df.loc[exec_date, b_code])
        z = float(row["zscore"])
        beta = float(row["beta"])
        if not np.isfinite(beta) or beta <= 0:
            continue
        if current is None:
            direction, a_sign, b_sign = entry_direction(z, config.entry_z)
            if direction is None:
                continue
            weights = choose_weights(signals, close_df, signal_date, beta, config, a_code, b_code)
            a_notional = config.capital * float(weights["a_weight"])
            b_notional = config.capital * float(weights["b_weight"])
            a_shares = int(a_notional // a_price) if config.integer_shares else a_notional / a_price
            b_shares = int(b_notional // b_price) if config.integer_shares else b_notional / b_price
            if a_shares <= 0 or b_shares <= 0:
                continue
            a_entry_notional = float(a_shares * a_price)
            b_entry_notional = float(b_shares * b_price)
            entry_cost = transaction_cost(a_entry_notional, "buy" if a_sign == 1 else "sell", config) + transaction_cost(b_entry_notional, "buy" if b_sign == 1 else "sell", config)
            current = {"a_code": a_code, "b_code": b_code, "direction": direction, "entry_signal_date": signal_date, "entry_date": exec_date, "entry_zscore": z, "entry_beta": beta, "a_sign": a_sign, "b_sign": b_sign, "a_entry_price": a_price, "b_entry_price": b_price, "a_shares": a_shares, "b_shares": b_shares, "a_entry_notional": a_entry_notional, "b_entry_notional": b_entry_notional, "gross_exposure": a_entry_notional + b_entry_notional, "entry_cost": entry_cost, **weights}
        else:
            if not should_exit(str(current["direction"]), z, config.exit_z):
                continue
            exit_cost = transaction_cost(float(current["a_shares"]) * a_price, "sell" if current["a_sign"] == 1 else "buy", config) + transaction_cost(float(current["b_shares"]) * b_price, "sell" if current["b_sign"] == 1 else "buy", config)
            pnl = trade_pnl(current, a_price, b_price) - exit_cost
            current.update({"exit_signal_date": signal_date, "exit_date": exec_date, "exit_zscore": z, "a_exit_price": a_price, "b_exit_price": b_price, "exit_cost": exit_cost, "total_cost": float(current["entry_cost"]) + exit_cost, "holding_days": (pd.Timestamp(exec_date) - pd.Timestamp(current["entry_date"])).days, "pnl": pnl, "return_on_gross_exposure": pnl / float(current["gross_exposure"]), "status": "closed"})
            trades.append(current)
            current = None
    if current is not None:
        last_date = close_df.index[-1]
        pnl = trade_pnl(current, float(close_df.loc[last_date, a_code]), float(close_df.loc[last_date, b_code]))
        current.update({"exit_signal_date": pd.NaT, "exit_date": pd.NaT, "exit_zscore": np.nan, "a_exit_price": np.nan, "b_exit_price": np.nan, "exit_cost": 0.0, "total_cost": float(current["entry_cost"]), "holding_days": (last_date - pd.Timestamp(current["entry_date"])).days, "pnl": pnl, "return_on_gross_exposure": pnl / float(current["gross_exposure"]), "status": "open"})
        trades.append(current)
    return pd.DataFrame(trades)


def choose_weights(signals: pd.DataFrame, close_df: pd.DataFrame, signal_date: pd.Timestamp, beta: float, config: Config, a_code: str, b_code: str) -> dict[str, object]:
    ols_a = 1 / (1 + abs(beta))
    if config.weight_mode != "max_sharpe":
        return {"a_weight": float(ols_a), "b_weight": float(1 - ols_a), "optimizer_sharpe": np.nan, "weight_method": "ols_beta"}
    hist = signals.loc[signals.index < signal_date].tail(config.opt_lookback)
    returns = close_df[[a_code, b_code]].pct_change().reindex(hist.index).dropna()
    if len(returns) < 40:
        return {"a_weight": float(ols_a), "b_weight": float(1 - ols_a), "optimizer_sharpe": np.nan, "weight_method": "ols_fallback"}
    z = hist["zscore"].reindex(returns.index)
    positions = signal_positions(z, config.entry_z, config.exit_z).shift(1).fillna(0)
    best_w = None
    best_s = -np.inf
    for a_weight in np.arange(config.min_weight, config.max_weight + config.weight_step / 2, config.weight_step):
        b_weight = 1 - a_weight
        strategy_returns = positions * (a_weight * returns[a_code] - b_weight * returns[b_code])
        s = sharpe(strategy_returns)
        if np.isfinite(s) and s > best_s:
            best_s = s
            best_w = float(a_weight)
    if best_w is None:
        return {"a_weight": float(ols_a), "b_weight": float(1 - ols_a), "optimizer_sharpe": np.nan, "weight_method": "ols_fallback"}
    return {"a_weight": best_w, "b_weight": float(1 - best_w), "optimizer_sharpe": float(best_s), "weight_method": "rolling_max_sharpe_grid"}


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
    return pd.DataFrame({"strategy_cumulative_return": strategy, "benchmark_cumulative_return": bench / bench.iloc[0] - 1})


def summarize(equity: pd.DataFrame, trades: pd.DataFrame, capital: float) -> dict[str, float]:
    returns = equity["daily_return"].replace([np.inf, -np.inf], np.nan).dropna()
    ending = float(equity["equity"].iloc[-1])
    closed = trades[trades["status"].eq("closed")] if not trades.empty else trades
    wins = closed[closed["pnl"] > 0] if not closed.empty else closed
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    return {"total_return": ending / capital - 1, "ending_equity": ending, "cagr": (ending / capital) ** (1 / years) - 1, "volatility": returns.std(ddof=1) * np.sqrt(252) if len(returns) else np.nan, "sharpe": sharpe(returns), "max_drawdown": float(equity["drawdown"].min()), "trade_count": float(len(trades)), "win_rate": float(len(wins) / len(closed)) if len(closed) else np.nan, "total_pnl": ending - capital}


def show_summary(summary: dict[str, float]) -> None:
    metrics = [("Total Return", pct(summary["total_return"])), ("Sharpe", num(summary["sharpe"])), ("Max DD", pct(summary["max_drawdown"])), ("CAGR", pct(summary["cagr"])), ("Win Rate", pct(summary["win_rate"])), ("Trades", f"{int(summary['trade_count'])}")]
    for i in range(0, len(metrics), 3):
        cols = st.columns(3)
        for col, (label, value) in zip(cols, metrics[i : i + 3]):
            col.metric(label, value)


def show_charts(result: dict[str, object], a_code: str, b_code: str, benchmark_name: str, config: Config) -> None:
    price_history: pd.DataFrame = result["price_history"]  # type: ignore[assignment]
    signals: pd.DataFrame = result["signals"]  # type: ignore[assignment]
    comparison: pd.DataFrame = result["comparison"]  # type: ignore[assignment]
    equity: pd.DataFrame = result["equity"]  # type: ignore[assignment]
    trades: pd.DataFrame = result["trades"]  # type: ignore[assignment]
    weights: pd.DataFrame = result["weights"]  # type: ignore[assignment]
    left, right = st.columns([0.58, 0.42])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_history.index, y=price_history[a_code], name=a_code, mode="lines"))
        fig.add_trace(go.Scatter(x=price_history.index, y=price_history[b_code], name=b_code, mode="lines"))
        fig.update_layout(title="Historical Close Price", template="plotly_white", height=370, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.plotly_chart(spread_chart(signals, config), use_container_width=True)
    left, right = st.columns([0.58, 0.42])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comparison.index, y=comparison["strategy_cumulative_return"], name="Strategy", mode="lines"))
        if "benchmark_cumulative_return" in comparison:
            fig.add_trace(go.Scatter(x=comparison.index, y=comparison["benchmark_cumulative_return"], name=benchmark_name, mode="lines"))
        fig.update_layout(title="Cumulative Return vs Benchmark", template="plotly_white", height=350, yaxis_tickformat=".1%", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity.index, y=equity["drawdown"], name="Drawdown", mode="lines", fill="tozeroy"))
        fig.update_layout(title="Drawdown", template="plotly_white", height=350, yaxis_tickformat=".1%", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(trade_chart(trades), use_container_width=True)
    with right:
        st.plotly_chart(weight_chart(weights), use_container_width=True)


def spread_chart(signals: pd.DataFrame, config: Config) -> go.Figure:
    fig = go.Figure()
    if signals.empty:
        fig.update_layout(title="Spread Signal", template="plotly_white", height=370)
        return fig
    upper = signals["spread_mean"] + config.entry_z * signals["spread_std"]
    lower = signals["spread_mean"] - config.entry_z * signals["spread_std"]
    fig.add_trace(go.Scatter(x=signals.index, y=signals["spread"], name="spread", mode="lines"))
    fig.add_trace(go.Scatter(x=signals.index, y=signals["spread_mean"], name="rolling mean", mode="lines"))
    fig.add_trace(go.Scatter(x=signals.index, y=upper, name="+entry band", mode="lines", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=signals.index, y=lower, name="-entry band", mode="lines", line=dict(dash="dash")))
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
        fig.update_layout(title="Entry Weights", template="plotly_white", height=320)
        return fig
    fig.add_trace(go.Scatter(x=weights["entry_date"], y=weights["a_weight"], name="A weight", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=weights["entry_date"], y=weights["b_weight"], name="B weight", mode="lines+markers"))
    fig.update_layout(title="Entry Weights", template="plotly_white", height=320, yaxis_tickformat=".0%", hovermode="x unified")
    return fig


def show_tables(result: dict[str, object]) -> None:
    trades: pd.DataFrame = result["trades"]  # type: ignore[assignment]
    signals: pd.DataFrame = result["signals"]  # type: ignore[assignment]
    st.subheader("交易明細")
    if trades.empty:
        st.warning("這段期間沒有觸發交易。")
    else:
        cols = ["direction", "entry_date", "exit_date", "entry_zscore", "exit_zscore", "a_weight", "b_weight", "optimizer_sharpe", "gross_exposure", "pnl", "return_on_gross_exposure", "holding_days", "status"]
        st.dataframe(trades[[c for c in cols if c in trades.columns]], use_container_width=True)
        st.download_button("Download trades CSV", trades.to_csv(index=False).encode("utf-8-sig"), "pair_trades.csv", "text/csv")
    with st.expander("Signal data"):
        st.dataframe(signals.tail(300), use_container_width=True)


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(x)), x])
    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(alpha), float(beta)


def entry_direction(z: float, entry_z: float) -> tuple[str | None, int, int]:
    if z <= -entry_z:
        return "long_spread", 1, -1
    if z >= entry_z:
        return "short_spread", -1, 1
    return None, 0, 0


def should_exit(direction: str, z: float, exit_z: float) -> bool:
    return (direction == "long_spread" and z >= exit_z) or (direction == "short_spread" and z <= -exit_z)


def signal_positions(zscores: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    pos = 0.0
    rows = []
    for z in zscores:
        if pos == 0:
            if z <= -entry_z:
                pos = 1.0
            elif z >= entry_z:
                pos = -1.0
        elif pos > 0 and z >= exit_z:
            pos = 0.0
        elif pos < 0 and z <= -exit_z:
            pos = 0.0
        rows.append(pos)
    return pd.Series(rows, index=zscores.index, dtype=float)


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


def pct(value: float) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.2%}"


def num(value: float) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.2f}"


if __name__ == "__main__":
    main()
