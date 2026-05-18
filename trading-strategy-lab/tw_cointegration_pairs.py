from __future__ import annotations

"""
Taiwan stock cointegration pairs trading research script.

This is a clean-room implementation of the common cointegration pairs workflow:
1. keep the Taiwan stock universe already defined in streamlit_app.py;
2. download Taiwan stock OHLC data from Yahoo Finance via yfinance;
3. screen same-industry pairs by correlation, OLS residual stationarity, half-life,
   mean crossings, and residual trend strength;
4. build spread = log(Y) - (alpha + beta * log(X));
5. trade rolling z-score mean reversion signals out-of-sample;
6. export pair screening results, trades, daily equity curves, and portfolio summary.

Run from this folder:

    python tw_cointegration_pairs.py --industry 半導體業 --top-n 10

Example with a smaller hand-picked universe:

    python tw_cointegration_pairs.py --codes 2330 2303 2454 3034 3711 --top-n 5

Notes:
- This script is for empirical research only, not investment advice.
- Pair selection uses the formation window only; backtest is run on the test window.
- Taiwan short selling, single-stock futures availability, borrow fee, liquidity, and
  price-limit constraints are not fully modeled.
"""

import argparse
import ast
import contextlib
import io
import itertools
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

DEFAULT_PERIOD = "3y"
DEFAULT_INTERVAL = "1d"

MIN_OBS = 120
DEFAULT_CORR_THRESHOLD = 0.80
DEFAULT_ADF_PVALUE = 0.05
DEFAULT_LOOKBACK = 120
DEFAULT_ENTRY_Z = 2.0
DEFAULT_EXIT_Z = 0.3
DEFAULT_STOP_Z = 3.5
DEFAULT_TOP_N = 10
DEFAULT_FORMATION_RATIO = 0.70

DEFAULT_CAPITAL = 1_000_000.0
DEFAULT_BROKER_FEE = 0.001425
DEFAULT_SELL_TAX = 0.001425


@dataclass(frozen=True)
class StockInfo:
    code: str
    name: str
    industry: str

    @property
    def label(self) -> str:
        return f"{self.code}_{self.name}"


@dataclass(frozen=True)
class PairConfig:
    lookback: int = DEFAULT_LOOKBACK
    entry_z: float = DEFAULT_ENTRY_Z
    exit_z: float = DEFAULT_EXIT_Z
    stop_z: float = DEFAULT_STOP_Z
    capital_per_pair: float = DEFAULT_CAPITAL
    broker_fee: float = DEFAULT_BROKER_FEE
    sell_tax: float = DEFAULT_SELL_TAX
    integer_shares: bool = True


def load_user_stock_universe(app_path: Path = Path("streamlit_app.py")) -> list[StockInfo]:
    """Read `single_stock_futures` from the existing Streamlit app without importing Streamlit."""
    if not app_path.exists():
        raise FileNotFoundError(
            f"Cannot find {app_path}. Run this script inside trading-strategy-lab "
            "or pass a valid streamlit_app.py path."
        )

    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "single_stock_futures":
                    raw = ast.literal_eval(node.value)
                    universe = []
                    for item in raw:
                        code = str(item.get("code", "")).strip()
                        name = str(item.get("name", "")).strip()
                        industry = str(item.get("industry", "")).strip()
                        if code and name and industry:
                            universe.append(StockInfo(code=code, name=name, industry=industry))
                    if universe:
                        return universe

    raise ValueError("Could not find a valid `single_stock_futures` list in streamlit_app.py.")


def filter_universe(
    universe: list[StockInfo],
    industry: str | None,
    codes: list[str] | None,
    max_symbols: int | None,
) -> list[StockInfo]:
    selected = universe

    if industry:
        selected = [s for s in selected if s.industry == industry]

    if codes:
        code_set = {str(c).strip().upper() for c in codes}
        selected = [s for s in selected if s.code.upper() in code_set]

    if max_symbols is not None and max_symbols > 0:
        selected = selected[:max_symbols]

    if len(selected) < 2:
        raise ValueError("Selected universe must contain at least two stocks.")

    return selected


def candidate_yahoo_tickers(code: str) -> list[str]:
    code = str(code).strip().upper()
    if "." in code or code.startswith("^") or "-" in code or "=" in code:
        return [code]
    return [f"{code}.TW", f"{code}.TWO"]


def extract_ohlc_from_single_download(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        fields: dict[str, pd.Series] = {}
        for field in ["Open", "Close"]:
            if field in raw.columns.get_level_values(0):
                obj = raw[field]
                fields[field] = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj
            elif field in raw.columns.get_level_values(1):
                for col in raw.columns:
                    if col[1] == field:
                        fields[field] = raw[col]
                        break
        if "Open" in fields and "Close" in fields:
            return pd.DataFrame(fields)
        return pd.DataFrame()

    if "Open" not in raw.columns or "Close" not in raw.columns:
        return pd.DataFrame()

    return raw[["Open", "Close"]].copy()


def normalize_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    out = obj.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out


def download_one_stock(
    stock: StockInfo,
    period: str,
    interval: str,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    downloaded = pd.DataFrame()

    for ticker in candidate_yahoo_tickers(stock.code):
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

            downloaded = extract_ohlc_from_single_download(raw)
            downloaded = downloaded.apply(pd.to_numeric, errors="coerce").dropna()
            downloaded = downloaded[(downloaded["Open"] > 0) & (downloaded["Close"] > 0)]

            if len(downloaded) >= MIN_OBS:
                downloaded.index = pd.to_datetime(downloaded.index).tz_localize(None).normalize()
                downloaded = downloaded[~downloaded.index.duplicated(keep="last")]
                time.sleep(sleep_seconds)
                return downloaded

        except Exception:
            downloaded = pd.DataFrame()

    time.sleep(sleep_seconds)
    return pd.DataFrame()


def download_universe_prices(
    universe: list[StockInfo],
    period: str,
    interval: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    open_dict: dict[str, pd.Series] = {}
    close_dict: dict[str, pd.Series] = {}
    meta_rows: list[dict[str, object]] = []

    print(f"Downloading {len(universe)} Taiwan stocks from Yahoo Finance...")

    for i, stock in enumerate(universe, start=1):
        data = download_one_stock(stock, period=period, interval=interval)
        status = "ok" if not data.empty else "missing"

        if not data.empty:
            open_dict[stock.label] = data["Open"]
            close_dict[stock.label] = data["Close"]

        meta_rows.append(
            {
                "code": stock.code,
                "name": stock.name,
                "industry": stock.industry,
                "label": stock.label,
                "status": status,
                "observations": int(len(data)),
            }
        )

        print(f"[{i:>3}/{len(universe)}] {stock.code} {stock.name}: {status}, obs={len(data)}")

    open_df = pd.DataFrame(open_dict).sort_index().ffill()
    close_df = pd.DataFrame(close_dict).sort_index().ffill()
    meta_df = pd.DataFrame(meta_rows)

    valid_cols = close_df.columns[close_df.notna().sum() >= MIN_OBS].tolist()
    open_df = open_df[valid_cols].dropna(how="all")
    close_df = close_df[valid_cols].dropna(how="all")

    complete = open_df.notna().sum(axis=1) >= 2
    open_df = normalize_index(open_df.loc[complete])
    close_df = normalize_index(close_df.loc[complete])

    if close_df.shape[1] < 2:
        raise ValueError("Fewer than two stocks have enough usable price observations.")

    return open_df, close_df, meta_df


def train_test_split_by_time(
    open_df: pd.DataFrame,
    close_df: pd.DataFrame,
    formation_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if not 0.3 <= formation_ratio <= 0.9:
        raise ValueError("formation_ratio should be between 0.3 and 0.9.")

    common = open_df.index.intersection(close_df.index)
    open_df = open_df.loc[common]
    close_df = close_df.loc[common]

    split_idx = int(len(common) * formation_ratio)
    split_idx = max(MIN_OBS, min(split_idx, len(common) - MIN_OBS // 2))
    split_date = common[split_idx]

    formation_open = open_df.iloc[:split_idx].copy()
    formation_close = close_df.iloc[:split_idx].copy()
    test_open = open_df.iloc[split_idx:].copy()
    test_close = close_df.iloc[split_idx:].copy()

    return formation_open, formation_close, test_open, test_close, split_date


def ols_spread(y: pd.Series, x: pd.Series) -> tuple[float, float, pd.Series]:
    pair = pd.concat([y, x], axis=1).dropna()
    y_clean = pair.iloc[:, 0].astype(float)
    x_clean = pair.iloc[:, 1].astype(float)

    X = sm.add_constant(x_clean)
    model = sm.OLS(y_clean, X).fit()

    alpha = float(model.params.iloc[0])
    beta = float(model.params.iloc[1])
    spread = y_clean - (alpha + beta * x_clean)
    spread.name = f"{y.name}-{x.name}_spread"

    return alpha, beta, spread


def adf_summary(series: pd.Series) -> tuple[float, float]:
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 30 or s.std(ddof=1) == 0:
        return float("nan"), float("nan")

    try:
        stat, pvalue, *_ = adfuller(s, autolag="AIC")
        return float(stat), float(pvalue)
    except Exception:
        return float("nan"), float("nan")


def estimate_half_life(spread: pd.Series) -> float:
    s = pd.Series(spread).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 30:
        return float("nan")

    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()
    idx = lagged.index.intersection(delta.index)

    if len(idx) < 30:
        return float("nan")

    try:
        X = sm.add_constant(lagged.loc[idx])
        model = sm.OLS(delta.loc[idx], X).fit()
        beta = float(model.params.iloc[1])
    except Exception:
        return float("nan")

    if beta >= 0:
        return float("nan")

    return float(-math.log(2) / beta)


def residual_trend_strength(spread: pd.Series) -> tuple[float, float, float]:
    s = pd.Series(spread).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 30:
        return float("nan"), float("nan"), float("inf")

    t = np.arange(len(s), dtype=float)
    try:
        X = sm.add_constant(t)
        model = sm.OLS(s.values, X).fit()
        slope = float(model.params[1])
        pvalue = float(model.pvalues[1])
        strength = abs(slope) * len(s) / float(s.std(ddof=1))
        return slope, pvalue, strength
    except Exception:
        return float("nan"), float("nan"), float("inf")


def count_mean_crossings(spread: pd.Series, bars_per_year: int = 252) -> tuple[int, float]:
    s = pd.Series(spread).replace([np.inf, -np.inf], np.nan).dropna()
    centered = s - s.mean()
    sign = np.sign(centered)
    crossings = int(((sign.shift(1) * sign) < 0).sum())
    years = max(len(s) / bars_per_year, 1e-9)
    return crossings, float(crossings / years)


def pair_score(row: pd.Series) -> float:
    score = 0.0

    corr = row.get("correlation", np.nan)
    adf_p = row.get("adf_pvalue", np.nan)
    half_life = row.get("half_life", np.nan)
    trend_strength = row.get("trend_strength", np.nan)
    crossings_per_year = row.get("crossings_per_year", np.nan)

    if pd.notna(corr):
        score += max(0.0, min(20.0, (float(corr) - 0.5) * 50.0))

    if pd.notna(adf_p):
        if adf_p < 0.01:
            score += 35.0
        elif adf_p < 0.05:
            score += 25.0
        elif adf_p < 0.10:
            score += 10.0

    if pd.notna(half_life):
        if 2 <= half_life <= 60:
            score += 20.0
        if 5 <= half_life <= 30:
            score += 10.0

    if pd.notna(trend_strength):
        if trend_strength < 0.5:
            score += 20.0
        elif trend_strength < 1.0:
            score += 10.0

    if pd.notna(crossings_per_year):
        score += max(0.0, min(10.0, crossings_per_year))

    return float(score)


def build_candidate_pairs(
    formation_close: pd.DataFrame,
    meta_df: pd.DataFrame,
    corr_threshold: float,
    adf_pvalue: float,
    same_industry_only: bool = True,
) -> pd.DataFrame:
    close = formation_close.dropna(axis=1, thresh=MIN_OBS).copy()
    log_price = np.log(close)
    returns = log_price.diff().dropna()

    meta = meta_df.set_index("label").to_dict("index")
    corr = returns.corr()

    records: list[dict[str, object]] = []

    for y_col, x_col in itertools.combinations(log_price.columns, 2):
        if same_industry_only:
            if meta.get(y_col, {}).get("industry") != meta.get(x_col, {}).get("industry"):
                continue

        c = corr.loc[y_col, x_col]
        if pd.isna(c) or float(c) < corr_threshold:
            continue

        pair_log = log_price[[y_col, x_col]].dropna()
        if len(pair_log) < MIN_OBS:
            continue

        alpha, beta, spread = ols_spread(pair_log[y_col], pair_log[x_col])
        if not np.isfinite(beta) or beta == 0:
            continue

        adf_stat, adf_p = adf_summary(spread)
        half_life = estimate_half_life(spread)
        trend_slope, trend_p, trend_strength = residual_trend_strength(spread)
        crossings, crossings_per_year = count_mean_crossings(spread)

        y_meta = meta.get(y_col, {})
        x_meta = meta.get(x_col, {})
        record = {
            "pair": f"{y_col}|{x_col}",
            "y_label": y_col,
            "x_label": x_col,
            "y_code": y_meta.get("code", y_col.split("_", 1)[0]),
            "y_name": y_meta.get("name", y_col),
            "x_code": x_meta.get("code", x_col.split("_", 1)[0]),
            "x_name": x_meta.get("name", x_col),
            "industry": y_meta.get("industry", ""),
            "correlation": float(c),
            "alpha": float(alpha),
            "beta": float(beta),
            "adf_stat": float(adf_stat),
            "adf_pvalue": float(adf_p),
            "half_life": float(half_life) if pd.notna(half_life) else np.nan,
            "trend_slope": float(trend_slope),
            "trend_pvalue": float(trend_p),
            "trend_strength": float(trend_strength),
            "mean_crossings": int(crossings),
            "crossings_per_year": float(crossings_per_year),
            "formation_observations": int(len(spread)),
            "spread_mean": float(spread.mean()),
            "spread_std": float(spread.std(ddof=1)),
            "adf_pass": bool(pd.notna(adf_p) and adf_p <= adf_pvalue),
        }
        records.append(record)

    out = pd.DataFrame(records)
    if out.empty:
        return out

    out["score"] = out.apply(pair_score, axis=1)
    out["suitable"] = (
        out["adf_pass"]
        & out["half_life"].between(2, 60, inclusive="both")
        & (out["trend_strength"] < 1.0)
        & (out["crossings_per_year"] >= 4)
    )
    out = out.sort_values(
        ["suitable", "score", "adf_pvalue", "correlation"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def rolling_spread_signals(
    open_df: pd.DataFrame,
    close_df: pd.DataFrame,
    y_label: str,
    x_label: str,
    config: PairConfig,
) -> pd.DataFrame:
    common = open_df.index.intersection(close_df.index)
    open_pair = open_df.loc[common, [y_label, x_label]].dropna()
    close_pair = close_df.loc[common, [y_label, x_label]].dropna()
    common = open_pair.index.intersection(close_pair.index)

    open_pair = open_pair.loc[common]
    close_pair = close_pair.loc[common]
    logp = np.log(close_pair)

    rows: list[dict[str, object]] = []
    for i in range(config.lookback, len(logp) - 1):
        signal_date = logp.index[i]
        exec_date = logp.index[i + 1]

        train = logp.iloc[i - config.lookback : i]
        alpha, beta, train_spread = ols_spread(train[y_label], train[x_label])

        spread_mean = float(train_spread.mean())
        spread_std = float(train_spread.std(ddof=1))
        if not np.isfinite(spread_std) or spread_std <= 0:
            continue

        spread = float(logp.iloc[i][y_label] - (alpha + beta * logp.iloc[i][x_label]))
        zscore = float((spread - spread_mean) / spread_std)

        rows.append(
            {
                "signal_date": signal_date,
                "exec_date": exec_date,
                "alpha": float(alpha),
                "beta": float(beta),
                "spread": spread,
                "spread_mean": spread_mean,
                "spread_std": spread_std,
                "zscore": zscore,
                "y_open": float(open_pair.loc[exec_date, y_label]),
                "x_open": float(open_pair.loc[exec_date, x_label]),
                "y_close": float(close_pair.loc[exec_date, y_label]),
                "x_close": float(close_pair.loc[exec_date, x_label]),
            }
        )

    if not rows:
        return pd.DataFrame()

    signals = pd.DataFrame(rows).set_index("signal_date")
    signals["position"] = stateful_position(signals["zscore"], config)
    return signals


def stateful_position(zscore: pd.Series, config: PairConfig) -> pd.Series:
    pos = np.zeros(len(zscore), dtype="int8")
    current = 0

    for i, z in enumerate(zscore.astype(float).values):
        if not np.isfinite(z):
            pos[i] = current
            continue

        if current == 0:
            if z >= config.entry_z:
                current = -1   # short spread: short Y, long X
            elif z <= -config.entry_z:
                current = 1    # long spread: long Y, short X
        else:
            if abs(z) <= config.exit_z or abs(z) >= config.stop_z:
                current = 0

        pos[i] = current

    return pd.Series(pos, index=zscore.index, name="position")


def trade_cost(notional: float, side: str, config: PairConfig) -> float:
    cost = abs(notional) * config.broker_fee
    if side.lower() == "sell":
        cost += abs(notional) * config.sell_tax
    return float(cost)


def target_shares(
    position: int,
    beta: float,
    y_price: float,
    x_price: float,
    config: PairConfig,
) -> tuple[float, float]:
    if position == 0:
        return 0.0, 0.0

    beta_abs = abs(float(beta))
    if not np.isfinite(beta_abs) or beta_abs <= 0:
        beta_abs = 1.0

    y_weight = 1.0 / (1.0 + beta_abs)
    x_weight = beta_abs / (1.0 + beta_abs)

    y_notional = config.capital_per_pair * y_weight
    x_notional = config.capital_per_pair * x_weight

    y_shares = position * y_notional / y_price
    x_shares = -position * x_notional / x_price

    if config.integer_shares:
        y_shares = math.copysign(math.floor(abs(y_shares)), y_shares)
        x_shares = math.copysign(math.floor(abs(x_shares)), x_shares)

    return float(y_shares), float(x_shares)


def backtest_one_pair(
    signals: pd.DataFrame,
    y_label: str,
    x_label: str,
    config: PairConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()

    records: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []

    y_shares = 0.0
    x_shares = 0.0
    cash = config.capital_per_pair
    equity = config.capital_per_pair

    open_trade: dict[str, object] | None = None

    prev_y_close = np.nan
    prev_x_close = np.nan

    for _, row in signals.iterrows():
        exec_date = pd.Timestamp(row["exec_date"])
        z = float(row["zscore"])
        beta = float(row["beta"])
        target_pos = int(row["position"])

        y_open = float(row["y_open"])
        x_open = float(row["x_open"])
        y_close = float(row["y_close"])
        x_close = float(row["x_close"])

        if np.isfinite(prev_y_close) and np.isfinite(prev_x_close):
            equity += y_shares * (y_close - prev_y_close) + x_shares * (x_close - prev_x_close)

        current_pos = int(np.sign(y_shares))
        position_changed = target_pos != current_pos

        if position_changed:
            target_y_shares, target_x_shares = target_shares(target_pos, beta, y_open, x_open, config)

            delta_y = target_y_shares - y_shares
            delta_x = target_x_shares - x_shares

            y_side = "buy" if delta_y > 0 else "sell"
            x_side = "buy" if delta_x > 0 else "sell"

            costs = 0.0
            if delta_y != 0:
                costs += trade_cost(delta_y * y_open, y_side, config)
            if delta_x != 0:
                costs += trade_cost(delta_x * x_open, x_side, config)

            equity -= costs
            cash -= delta_y * y_open + delta_x * x_open + costs

            if current_pos != 0 and open_trade is not None:
                pnl = equity - float(open_trade["entry_equity"])
                closed = dict(open_trade)
                closed.update(
                    {
                        "exit_date": exec_date,
                        "exit_zscore": z,
                        "exit_y_open": y_open,
                        "exit_x_open": x_open,
                        "pnl": pnl,
                        "return_on_capital": pnl / config.capital_per_pair,
                        "exit_cost": costs,
                        "holding_days": (exec_date - pd.Timestamp(open_trade["entry_date"])).days,
                    }
                )
                trades.append(closed)
                open_trade = None

            y_shares = target_y_shares
            x_shares = target_x_shares

            if target_pos != 0:
                open_trade = {
                    "pair": f"{y_label}|{x_label}",
                    "y_label": y_label,
                    "x_label": x_label,
                    "direction": "long_spread" if target_pos == 1 else "short_spread",
                    "entry_date": exec_date,
                    "entry_zscore": z,
                    "entry_beta": beta,
                    "entry_y_open": y_open,
                    "entry_x_open": x_open,
                    "entry_y_shares": y_shares,
                    "entry_x_shares": x_shares,
                    "entry_equity": equity,
                    "entry_cost": costs,
                }

        records.append(
            {
                "date": exec_date,
                "pair": f"{y_label}|{x_label}",
                "zscore": z,
                "beta": beta,
                "position": int(np.sign(y_shares)),
                "y_shares": y_shares,
                "x_shares": x_shares,
                "cash": cash,
                "equity": equity,
                "daily_pnl": 0.0,
            }
        )

        prev_y_close = y_close
        prev_x_close = x_close

    equity_df = pd.DataFrame(records).set_index("date")
    if not equity_df.empty:
        equity_df["daily_pnl"] = equity_df["equity"].diff().fillna(equity_df["equity"] - config.capital_per_pair)
        equity_df["daily_return"] = equity_df["daily_pnl"] / config.capital_per_pair
        equity_df["cumulative_return"] = equity_df["equity"] / config.capital_per_pair - 1.0
        equity_df["drawdown"] = equity_df["equity"] / equity_df["equity"].cummax() - 1.0

    if open_trade is not None:
        last = equity_df.iloc[-1]
        closed = dict(open_trade)
        closed.update(
            {
                "exit_date": pd.NaT,
                "exit_zscore": float(last["zscore"]),
                "exit_y_open": np.nan,
                "exit_x_open": np.nan,
                "pnl": float(last["equity"]) - float(open_trade["entry_equity"]),
                "return_on_capital": (float(last["equity"]) - float(open_trade["entry_equity"])) / config.capital_per_pair,
                "exit_cost": 0.0,
                "holding_days": np.nan,
                "status": "open",
            }
        )
        trades.append(closed)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty and "status" not in trades_df.columns:
        trades_df["status"] = "closed"

    return equity_df, trades_df


def performance_summary(equity: pd.DataFrame, trades: pd.DataFrame, capital: float) -> dict[str, object]:
    if equity.empty:
        return {
            "total_return": np.nan,
            "ending_equity": capital,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "trade_count": 0,
            "win_rate": np.nan,
            "total_pnl": 0.0,
        }

    returns = equity["daily_return"].replace([np.inf, -np.inf], np.nan).dropna()
    ending = float(equity["equity"].iloc[-1])
    closed = trades[trades["status"].eq("closed")] if not trades.empty and "status" in trades else trades
    wins = closed[closed["pnl"] > 0] if not closed.empty else closed

    if len(returns) > 2 and returns.std(ddof=1) > 0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std(ddof=1))
    else:
        sharpe = float("nan")

    return {
        "total_return": ending / capital - 1.0,
        "ending_equity": ending,
        "sharpe": sharpe,
        "max_drawdown": float(equity["drawdown"].min()) if "drawdown" in equity else np.nan,
        "trade_count": int(len(closed)),
        "win_rate": float((closed["pnl"] > 0).mean()) if len(closed) else np.nan,
        "total_pnl": ending - capital,
    }


def aggregate_portfolio(pair_equities: list[pd.DataFrame], capital_per_pair: float) -> pd.DataFrame:
    if not pair_equities:
        return pd.DataFrame()

    aligned = []
    for eq in pair_equities:
        if not eq.empty:
            aligned.append(eq["equity"].rename(eq["pair"].iloc[0]))

    if not aligned:
        return pd.DataFrame()

    combined = pd.concat(aligned, axis=1).ffill()
    combined = combined.dropna(how="all")
    pnl = combined.sub(capital_per_pair).sum(axis=1)

    total_capital = capital_per_pair * combined.shape[1]
    portfolio = pd.DataFrame(index=combined.index)
    portfolio["portfolio_equity"] = total_capital + pnl
    portfolio["portfolio_daily_pnl"] = portfolio["portfolio_equity"].diff().fillna(
        portfolio["portfolio_equity"] - total_capital
    )
    portfolio["portfolio_daily_return"] = portfolio["portfolio_daily_pnl"] / total_capital
    portfolio["portfolio_cumulative_return"] = portfolio["portfolio_equity"] / total_capital - 1.0
    portfolio["portfolio_drawdown"] = portfolio["portfolio_equity"] / portfolio["portfolio_equity"].cummax() - 1.0
    return portfolio


def run_research(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    universe = load_user_stock_universe(Path(args.app_path))
    selected = filter_universe(
        universe=universe,
        industry=args.industry,
        codes=args.codes,
        max_symbols=args.max_symbols,
    )

    meta_df = pd.DataFrame([asdict(s) | {"label": s.label} for s in selected])
    meta_df.to_csv(output_dir / "selected_universe.csv", index=False, encoding="utf-8-sig")

    open_df, close_df, download_meta = download_universe_prices(
        universe=selected,
        period=args.period,
        interval=args.interval,
    )
    download_meta.to_csv(output_dir / "download_status.csv", index=False, encoding="utf-8-sig")
    open_df.to_csv(output_dir / "open_prices.csv", encoding="utf-8-sig")
    close_df.to_csv(output_dir / "close_prices.csv", encoding="utf-8-sig")

    formation_open, formation_close, test_open, test_close, split_date = train_test_split_by_time(
        open_df,
        close_df,
        formation_ratio=args.formation_ratio,
    )

    print(f"Formation observations: {len(formation_close)}")
    print(f"Test observations: {len(test_close)}")
    print(f"Split date: {split_date.date()}")

    candidates = build_candidate_pairs(
        formation_close=formation_close,
        meta_df=meta_df,
        corr_threshold=args.corr_threshold,
        adf_pvalue=args.adf_pvalue,
        same_industry_only=not args.allow_cross_industry,
    )
    candidates.to_csv(output_dir / "pair_screening.csv", index=False, encoding="utf-8-sig")

    if candidates.empty:
        raise RuntimeError("No candidate pairs passed the basic correlation and data filters.")

    selected_pairs = candidates.head(args.top_n).copy()
    selected_pairs.to_csv(output_dir / "selected_pairs.csv", index=False, encoding="utf-8-sig")

    config = PairConfig(
        lookback=args.lookback,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        capital_per_pair=args.capital_per_pair,
        broker_fee=args.broker_fee,
        sell_tax=args.sell_tax,
        integer_shares=not args.allow_fractional_shares,
    )

    combined_open = pd.concat([formation_open.tail(config.lookback + 5), test_open]).sort_index()
    combined_close = pd.concat([formation_close.tail(config.lookback + 5), test_close]).sort_index()
    test_start = test_close.index.min()

    pair_equities: list[pd.DataFrame] = []
    trade_ledgers: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for _, pair_row in selected_pairs.iterrows():
        y_label = str(pair_row["y_label"])
        x_label = str(pair_row["x_label"])

        print(f"Backtesting {y_label} / {x_label}...")

        signals = rolling_spread_signals(combined_open, combined_close, y_label, x_label, config)
        if signals.empty:
            continue

        signals = signals[pd.to_datetime(signals["exec_date"]) >= test_start].copy()
        signals.to_csv(output_dir / f"signals_{safe_filename(y_label)}__{safe_filename(x_label)}.csv", encoding="utf-8-sig")

        equity, trades = backtest_one_pair(signals, y_label, x_label, config)

        if not equity.empty:
            equity.to_csv(output_dir / f"equity_{safe_filename(y_label)}__{safe_filename(x_label)}.csv", encoding="utf-8-sig")
            pair_equities.append(equity)

        if not trades.empty:
            trades.to_csv(output_dir / f"trades_{safe_filename(y_label)}__{safe_filename(x_label)}.csv", index=False, encoding="utf-8-sig")
            trade_ledgers.append(trades)

        stats = performance_summary(equity, trades, config.capital_per_pair)
        summary_rows.append(
            {
                "pair": f"{y_label}|{x_label}",
                "industry": pair_row.get("industry", ""),
                "formation_correlation": pair_row.get("correlation", np.nan),
                "formation_adf_pvalue": pair_row.get("adf_pvalue", np.nan),
                "formation_beta": pair_row.get("beta", np.nan),
                **stats,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")

    all_trades = pd.concat(trade_ledgers, ignore_index=True) if trade_ledgers else pd.DataFrame()
    all_trades.to_csv(output_dir / "all_trades.csv", index=False, encoding="utf-8-sig")

    portfolio = aggregate_portfolio(pair_equities, config.capital_per_pair)
    if not portfolio.empty:
        portfolio.to_csv(output_dir / "portfolio_equity.csv", encoding="utf-8-sig")

    run_config = {
        "args": vars(args),
        "pair_config": asdict(config),
        "split_date": str(split_date.date()),
        "selected_symbols": len(selected),
        "downloaded_symbols": int(close_df.shape[1]),
        "candidate_pairs": int(len(candidates)),
        "backtested_pairs": int(len(summary)),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nTop selected pairs:")
    print(selected_pairs[["rank", "y_code", "y_name", "x_code", "x_name", "industry", "correlation", "adf_pvalue", "half_life", "score", "suitable"]].to_string(index=False))

    print("\nBacktest summary:")
    if summary.empty:
        print("No pairs were backtested.")
    else:
        cols = ["pair", "total_return", "sharpe", "max_drawdown", "trade_count", "win_rate", "total_pnl"]
        print(summary[cols].to_string(index=False))

    print(f"\nOutputs saved to: {output_dir.resolve()}")

    return {
        "selected_pairs": selected_pairs,
        "summary": summary,
        "portfolio": portfolio,
        "output_dir": output_dir,
    }


def safe_filename(text: str) -> str:
    return (
        str(text)
        .replace("/", "-")
        .replace("\\", "-")
        .replace("|", "__")
        .replace(" ", "_")
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Taiwan stock cointegration pairs trading empirical backtest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--app-path", default="streamlit_app.py", help="Path to existing streamlit_app.py containing single_stock_futures.")
    parser.add_argument("--industry", default="半導體業", help="Taiwan industry to screen. Set empty string to use all selected codes.")
    parser.add_argument("--codes", nargs="*", default=None, help="Optional explicit Taiwan stock codes, e.g. --codes 2330 2303 2454.")
    parser.add_argument("--max-symbols", type=int, default=None, help="Optional cap on symbols after filtering.")
    parser.add_argument("--allow-cross-industry", action="store_true", help="Allow pairs across different industries.")

    parser.add_argument("--period", default=DEFAULT_PERIOD, help="Yahoo Finance period, e.g. 1y, 3y, 5y.")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Yahoo Finance interval.")
    parser.add_argument("--formation-ratio", type=float, default=DEFAULT_FORMATION_RATIO, help="Fraction used for pair selection before out-of-sample test.")

    parser.add_argument("--corr-threshold", type=float, default=DEFAULT_CORR_THRESHOLD, help="Minimum return correlation for candidate pairs.")
    parser.add_argument("--adf-pvalue", type=float, default=DEFAULT_ADF_PVALUE, help="ADF p-value threshold for spread stationarity.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Number of selected pairs to backtest.")

    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="Rolling OLS lookback for signal generation.")
    parser.add_argument("--entry-z", type=float, default=DEFAULT_ENTRY_Z, help="Entry z-score threshold.")
    parser.add_argument("--exit-z", type=float, default=DEFAULT_EXIT_Z, help="Exit z-score threshold.")
    parser.add_argument("--stop-z", type=float, default=DEFAULT_STOP_Z, help="Stop z-score threshold.")

    parser.add_argument("--capital-per-pair", type=float, default=DEFAULT_CAPITAL, help="Capital allocated to each pair.")
    parser.add_argument("--broker-fee", type=float, default=DEFAULT_BROKER_FEE, help="Broker fee rate per transaction notional.")
    parser.add_argument("--sell-tax", type=float, default=DEFAULT_SELL_TAX, help="Sell-side transaction tax/fee rate.")
    parser.add_argument("--allow-fractional-shares", action="store_true", help="Use fractional shares instead of integer shares.")

    parser.add_argument("--output-dir", default="results/tw_cointegration_pairs", help="Output directory.")

    args = parser.parse_args(argv)

    if args.industry == "":
        args.industry = None

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run_research(args)


if __name__ == "__main__":
    main()
