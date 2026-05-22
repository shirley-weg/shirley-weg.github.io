# ============================================================
# strategy3_ff5_pipeline.py
# VERSION: V5_STREAMLIT_PROGRESS_REALDATA_2019_2026
# Notes:
# - Strategy 3 official backtest period: 2019-01-01 to 2026-05-19
# - Strict mode: backtest starts only after real price_df/formations/ff5/alpha_scores are generated
# - Strategy 3 supports monthly and biweekly rebalancing
# - Streamlit web progress bar added for raw data reading, FF5 construction, rolling alpha, backtest, and benchmarks
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm


FACTOR_COLS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
DEFAULT_START = pd.Timestamp("2019-01-01")
DEFAULT_END = pd.Timestamp("2026-05-19")


ProgressCallback = Callable[[str, int, int, str], None]


def emit_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int = 100,
    message: str = "",
) -> None:
    """Safely emit progress to Streamlit or any UI callback.

    Callback signature: (stage, current, total, message).
    current/total are intended as overall progress, usually 0~100.
    """
    if progress_callback is None:
        return
    try:
        total = max(int(total), 1)
        current = max(0, min(int(current), total))
        progress_callback(str(stage), current, total, str(message or stage))
    except Exception:
        # Progress reporting must never break the data pipeline.
        return


def scaled_progress(start: int, end: int, idx: int, total: int) -> int:
    if total <= 0:
        return int(end)
    frac = max(0.0, min(float(idx) / float(total), 1.0))
    return int(round(start + (end - start) * frac))


@dataclass(frozen=True)
class FF5RawBuildConfig:
    market_cap_top_n: int = 150
    lookback_days: int = 252
    min_obs: int = 180
    start_date: pd.Timestamp | str | None = DEFAULT_START
    end_date: pd.Timestamp | str | None = DEFAULT_END
    # "monthly" = 每月調倉；"biweekly" = 雙周調倉。
    # 這個變數同時控制股票池形成日、FF5 因子持有區間、alpha 訊號形成日。
    rebalance_frequency: str = "monthly"


# ============================================================
# Generic robust helpers
# ============================================================

def _as_timestamp(value, default: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """Safely convert scalar to Timestamp; prevents Timestamp/float comparisons."""
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return default
    if pd.isna(ts):
        return default
    return pd.Timestamp(ts).normalize()


def _date_bounds(config: FF5RawBuildConfig) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    return _as_timestamp(config.start_date, DEFAULT_START), _as_timestamp(config.end_date, DEFAULT_END)


def canonical_col_key(col: object) -> str:
    return re.sub(r"[\s_\-()（）/\\.\[\]【】:：]+", "", str(col).strip().lower())


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or len(df.columns) == 0:
        return None

    exact_map = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in exact_map:
            return exact_map[cand]

    norm_map = {canonical_col_key(c): c for c in df.columns}
    for cand in candidates:
        key = canonical_col_key(cand)
        if key in norm_map:
            return norm_map[key]

    # Partial matching for columns with units, e.g. 收盤價(元), 股票代號/名稱
    for cand in candidates:
        key = canonical_col_key(cand)
        if not key:
            continue
        for col in df.columns:
            ckey = canonical_col_key(col)
            if key == ckey or key in ckey or ckey in key:
                return col

    return None


def rename_by_alias(df: pd.DataFrame, alias_map: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    ren = {}
    for target, aliases in alias_map.items():
        if target in out.columns:
            continue
        found = find_column(out, aliases)
        if found is not None and found != target:
            ren[found] = target
    if ren:
        out = out.rename(columns=ren)
    return out


def clean_stock_id(s: pd.Series) -> pd.Series:
    extracted = s.astype(str).str.extract(r"(\d+)")[0]
    return extracted.where(extracted.notna(), "").str.zfill(4)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("--", "", regex=False)
        .str.replace("- -", "", regex=False)
        .str.replace("—", "", regex=False)
        .str.replace("－", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def parse_date(s: pd.Series) -> pd.Series:
    """Robust date parser for CMoney exports: yyyy/mm/dd, yyyymmdd, Excel serial, Timestamp."""
    if s is None:
        return pd.Series(dtype="datetime64[ns]")

    raw = s.copy()
    raw_str = raw.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    # 20191231 / 20191231.0
    compact = raw_str.str.replace(r"\.0$", "", regex=True)
    is_yyyymmdd = compact.str.fullmatch(r"\d{8}")
    if is_yyyymmdd.any():
        out.loc[is_yyyymmdd] = pd.to_datetime(compact.loc[is_yyyymmdd], format="%Y%m%d", errors="coerce")

    # 2019/12/31, 2019-12-31, Timestamp
    remain = out.isna()
    if remain.any():
        out.loc[remain] = pd.to_datetime(raw.loc[remain], errors="coerce")

    # Excel serial only when it looks like a serial number, not yyyyMMdd
    numeric = pd.to_numeric(raw, errors="coerce")
    serial_like = numeric.between(20000, 60000) & compact.str.fullmatch(r"\d+")
    serial_like = serial_like & out.isna()
    if serial_like.any():
        out.loc[serial_like] = pd.to_datetime(
            numeric.loc[serial_like],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    return out.dt.normalize().astype("datetime64[ns]")


def parse_cmoney_quarter(s: pd.Series) -> pd.Series:
    """Parse strings like 2023Q1, 2023/1, 2023年第1季, 2023 1 into Period[Q]."""
    if s is None:
        return pd.Series(dtype=object)
    text = s.astype(str).str.strip()
    parsed = text.str.extract(r"(\d{4})\D*(?:q|Q|第)?\D*(\d{1,2})")
    years = pd.to_numeric(parsed[0], errors="coerce")
    quarters = pd.to_numeric(parsed[1], errors="coerce")
    out = pd.Series(pd.NaT, index=s.index, dtype=object)
    valid = years.notna() & quarters.notna() & quarters.between(1, 4)
    if valid.any():
        out.loc[valid] = pd.PeriodIndex.from_fields(
            year=years.loc[valid].astype(int),
            quarter=quarters.loc[valid].astype(int),
            freq="Q",
        )
    return out


def filter_date_range(df: pd.DataFrame, date_col: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = parse_date(out[date_col])
    out = out.dropna(subset=[date_col])
    if start is not None:
        out = out[out[date_col] >= start]
    if end is not None:
        out = out[out[date_col] <= end]
    return out


# ============================================================
# File discovery / reading
# ============================================================

def discover_strategy3_raw_files(raw_dir: str | Path) -> dict[str, list[Path]]:
    base = Path(raw_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Strategy 3 raw data folder not found: {base}")

    candidates = [
        p for p in base.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".xlsx", ".xls", ".csv"}
        and not p.name.startswith("~$")
    ]

    def blob(path: Path) -> str:
        rel = str(path.relative_to(base)).upper().replace("\\", "/")
        return rel

    def is_group(path: Path, group: str) -> bool:
        b = blob(path)
        # use both folder and filename; supports data/xxx.xlsx, ad_price/xxx.xlsx
        if group == "ad_price":
            return ("AD_PRICE" in b or "AD PRICE" in b or "ADPRICE" in b or "/AD_PRICE/" in b or "/AD PRICE/" in b)
        if group == "asset":
            return ("ASSET" in b or "/ASSET/" in b)
        if group == "rmw":
            return ("RMW" in b or "/RMW/" in b)
        if group == "data":
            return (
                ("DATA" in b or "/DATA/" in b)
                and not is_group(path, "ad_price")
                and not is_group(path, "asset")
                and not is_group(path, "rmw")
                and "A09012" not in b
            )
        return False

    ad_price = sorted(p for p in candidates if is_group(p, "ad_price"))
    asset = sorted(p for p in candidates if is_group(p, "asset"))
    rmw = sorted(p for p in candidates if is_group(p, "rmw"))
    data = sorted(p for p in candidates if is_group(p, "data"))

    if not data:
        found = "\n".join(str(p.relative_to(base)) for p in candidates[:120])
        raise FileNotFoundError(
            "Missing required DATA files. "
            f"Checked folder: {base}\nFound files:\n{found}"
        )

    # AD PRICE / ASSET / RMW are allowed to be absent; downstream uses DATA/neutral fallback.
    return {"data": data, "ad_price": ad_price, "asset": asset, "rmw": rmw}


def build_file_manifest(files: dict[str, list[Path]]) -> pd.DataFrame:
    rows = []
    for group, paths in files.items():
        for p in paths:
            rows.append({
                "group": group,
                "file_name": p.name,
                "path": str(p),
                "size_mb": p.stat().st_size / 1_000_000,
            })
    return pd.DataFrame(rows)


def read_one_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        for enc in ["utf-8-sig", "cp950", "big5", "utf-8"]:
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except Exception:
                df = None
        if df is None:
            raise ValueError(f"Cannot read CSV file: {path}")
    else:
        df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    df["source_file"] = path.name
    return df


def read_many_excel(
    paths: list[Path],
    progress_callback: ProgressCallback | None = None,
    stage: str = "讀取 Excel",
    progress_start: int = 0,
    progress_end: int = 15,
) -> pd.DataFrame:
    dfs = []
    total = len(paths)
    if total == 0:
        emit_progress(progress_callback, stage, progress_end, 100, "沒有檔案需要讀取")
        return pd.DataFrame()

    for i, path in enumerate(paths, start=1):
        emit_progress(
            progress_callback,
            stage,
            scaled_progress(progress_start, progress_end, i - 1, total),
            100,
            f"{stage}: {i}/{total} - {path.name}",
        )
        try:
            df = read_one_table(path)
            if not df.empty:
                dfs.append(df)
        except Exception as exc:
            raise ValueError(f"Failed to read raw file {path}: {exc}") from exc
        emit_progress(
            progress_callback,
            stage,
            scaled_progress(progress_start, progress_end, i, total),
            100,
            f"{stage}: {i}/{total} - {path.name}",
        )
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)


# ============================================================
# Normalizers
# ============================================================

COMMON_ALIASES = {
    "date": ["date", "日期", "年月日", "交易日期", "資料日期"],
    "stock_id": ["stock_id", "stock_code", "code", "ticker", "symbol", "股票代號", "股票代碼", "證券代碼", "證券代號", "公司代號", "代號"],
    "stock_name": ["stock_name", "name", "股票名稱", "證券名稱", "簡稱", "公司名稱"],
}

def normalize_market_data(data_raw: pd.DataFrame, start_date=None, end_date=None) -> pd.DataFrame:
    if data_raw is None or data_raw.empty:
        raise ValueError("DATA files are empty.")

    df = rename_by_alias(data_raw, {
        **COMMON_ALIASES,
        "open": ["open", "開盤價"],
        "close": ["close", "收盤價", "收盤價(元)", "價格"],
        "volume": ["volume", "成交量"],
        "trading_value_thousand": ["trading_value_thousand", "成交金額(千)", "成交金額", "成交值"],
        "capital_million": ["capital_million", "股本(百萬)", "股本"],
        "market_cap_100m": ["market_cap_100m", "總市值(億)", "市值(億)", "總市值", "市值"],
        "pb": ["pb", "股價淨值比", "PBR", "PB"],
        "turnover_pct": ["turnover_pct", "週轉率(%)", "週轉率"],
    })

    # Some exports combine code and name into one column.
    if "stock_id" not in df.columns:
        combined = find_column(df, ["股票代號/名稱", "證券代碼/名稱", "代號名稱"])
        if combined is not None:
            df["stock_id"] = df[combined]
            if "stock_name" not in df.columns:
                df["stock_name"] = df[combined].astype(str).str.replace(r"^\s*\d+\s*", "", regex=True)

    required = ["date", "stock_id", "market_cap_100m"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DATA files are missing required columns: {missing}; current columns={list(data_raw.columns)}")

    df["date"] = parse_date(df["date"])
    df["stock_id"] = clean_stock_id(df["stock_id"])
    if "stock_name" not in df.columns:
        df["stock_name"] = ""

    for col in ["open", "close", "volume", "trading_value_thousand", "capital_million", "market_cap_100m", "pb", "turnover_pct"]:
        if col in df.columns:
            df[col] = to_num(df[col])
        else:
            df[col] = np.nan

    start, end = _as_timestamp(start_date, None), _as_timestamp(end_date, None)
    df = filter_date_range(df, "date", start, end)

    columns = ["date", "stock_id", "stock_name", "open", "close", "volume", "trading_value_thousand", "capital_million", "market_cap_100m", "pb", "turnover_pct"]
    out = (
        df.dropna(subset=["date", "stock_id", "market_cap_100m"])
        .query("stock_id != ''")
        .drop_duplicates(["date", "stock_id"], keep="last")
        .loc[:, columns]
        .sort_values(["date", "stock_id"])
        .reset_index(drop=True)
    )
    if out.empty:
        raise ValueError("DATA normalization produced zero rows. Please check date range and required columns.")
    return out


def normalize_ad_price(ad_raw: pd.DataFrame, data: pd.DataFrame | None = None, start_date=None, end_date=None) -> pd.DataFrame:
    # If AD PRICE is missing, use DATA close as adjusted close.
    if ad_raw is None or ad_raw.empty:
        if data is None or data.empty or "close" not in data.columns:
            raise ValueError("AD PRICE files are missing and DATA close is unavailable.")
        df = data[["date", "stock_id", "stock_name", "close", "volume", "trading_value_thousand", "market_cap_100m"]].copy()
        df = df.rename(columns={"close": "adj_close"})
    else:
        df = rename_by_alias(ad_raw, {
            **COMMON_ALIASES,
            "adj_close": ["adj_close", "close", "Close", "收盤價", "收盤價(元)", "還原收盤價", "調整收盤價", "Adj Close", "adjclose"],
            "volume": ["volume", "成交量"],
            "trading_value_thousand": ["trading_value_thousand", "成交金額(千)", "成交金額", "成交值"],
            "market_cap_100m": ["market_cap_100m", "總市值(億)", "市值(億)", "總市值", "市值"],
        })

    required = ["date", "stock_id", "adj_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"AD PRICE files are missing required columns: {missing}; current columns={list(ad_raw.columns) if ad_raw is not None else []}")

    if "stock_name" not in df.columns:
        df["stock_name"] = ""
    df["date"] = parse_date(df["date"])
    df["stock_id"] = clean_stock_id(df["stock_id"])
    df["adj_close"] = to_num(df["adj_close"])
    for col in ["volume", "trading_value_thousand", "market_cap_100m"]:
        if col in df.columns:
            df[col] = to_num(df[col])
        else:
            df[col] = np.nan

    start, end = _as_timestamp(start_date, None), _as_timestamp(end_date, None)
    df = filter_date_range(df, "date", start, end)

    out = (
        df.dropna(subset=["date", "stock_id", "adj_close"])
        .query("stock_id != '' and adj_close > 0")
        .drop_duplicates(["date", "stock_id"], keep="last")
        .sort_values(["stock_id", "date"])
        .reset_index(drop=True)
    )
    out["ret"] = out.groupby("stock_id")["adj_close"].pct_change()
    out.loc[~np.isfinite(out["ret"]), "ret"] = np.nan
    out = out.sort_values(["date", "stock_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("Price normalization produced zero rows. Please check AD PRICE/DATA date range.")
    return out


def normalize_asset(asset_raw: pd.DataFrame) -> pd.DataFrame:
    if asset_raw is None or asset_raw.empty:
        return pd.DataFrame()
    asset = rename_by_alias(asset_raw, {
        **COMMON_ALIASES,
        "year_quarter": ["year_quarter", "年季", "季度", "季別"],
        "total_assets_thousand": ["total_assets_thousand", "資產總計(千)", "資產總額(千)", "資產總計", "資產總額"],
        "total_liabilities_thousand": ["total_liabilities_thousand", "負債總計(千)", "負債總額(千)", "負債總計", "負債總額"],
        "equity_total_thousand": ["equity_total_thousand", "權益總計(千)", "權益總額(千)", "股東權益總計(千)", "權益總計", "淨值"],
        "announce_date_asset": ["announce_date_asset", "公告日期", "發布日期", "財報公告日"],
    })
    required = ["stock_id", "year_quarter"]
    if any(c not in asset.columns for c in required):
        return pd.DataFrame()
    asset["stock_id"] = clean_stock_id(asset["stock_id"])
    asset["fiscal_q"] = parse_cmoney_quarter(asset["year_quarter"])
    if "announce_date_asset" in asset.columns:
        asset["announce_date_asset"] = parse_date(asset["announce_date_asset"])
    else:
        # Conservative public availability: quarter end + 90 days
        q_end = asset["fiscal_q"].apply(lambda p: p.end_time if pd.notna(p) else pd.NaT)
        asset["announce_date_asset"] = pd.to_datetime(q_end, errors="coerce") + pd.Timedelta(days=90)
    for col in ["total_assets_thousand", "total_liabilities_thousand", "equity_total_thousand"]:
        if col in asset.columns:
            asset[col] = to_num(asset[col])
        else:
            asset[col] = np.nan
    asset = asset.dropna(subset=["stock_id", "fiscal_q", "announce_date_asset"]).sort_values(["stock_id", "fiscal_q"])
    if asset.empty:
        return pd.DataFrame()
    asset["asset_growth"] = asset.groupby("stock_id")["total_assets_thousand"].pct_change(4)
    return asset[["stock_id", "stock_name", "fiscal_q", "announce_date_asset", "total_assets_thousand", "total_liabilities_thousand", "equity_total_thousand", "asset_growth"]]


def normalize_rmw(rmw_raw: pd.DataFrame) -> pd.DataFrame:
    if rmw_raw is None or rmw_raw.empty:
        return pd.DataFrame()
    rmw = rename_by_alias(rmw_raw, {
        **COMMON_ALIASES,
        "year_quarter": ["year_quarter", "年季", "季度", "季別"],
        "revenue_thousand": ["revenue_thousand", "營業收入淨額(千)", "營業收入(千)", "營收(千)", "營業收入淨額"],
        "operating_income_thousand": ["operating_income_thousand", "營業利益(千)", "營業利益", "營業淨利(千)", "營業淨利"],
        "pretax_income_thousand": ["pretax_income_thousand", "稅前純益(千)", "稅前淨利(千)", "稅前純益"],
        "net_income_thousand": ["net_income_thousand", "稅後純益(千)", "稅後淨利(千)", "本期淨利(千)", "稅後純益"],
        "eps": ["eps", "每股稅後盈餘(元)", "EPS"],
        "announce_date_rmw": ["announce_date_rmw", "公告日期", "發布日期", "財報公告日"],
    })
    required = ["stock_id", "year_quarter"]
    if any(c not in rmw.columns for c in required):
        return pd.DataFrame()
    rmw["stock_id"] = clean_stock_id(rmw["stock_id"])
    rmw["fiscal_q"] = parse_cmoney_quarter(rmw["year_quarter"])
    if "announce_date_rmw" in rmw.columns:
        rmw["announce_date_rmw"] = parse_date(rmw["announce_date_rmw"])
    else:
        q_end = rmw["fiscal_q"].apply(lambda p: p.end_time if pd.notna(p) else pd.NaT)
        rmw["announce_date_rmw"] = pd.to_datetime(q_end, errors="coerce") + pd.Timedelta(days=90)
    for col in ["revenue_thousand", "operating_income_thousand", "pretax_income_thousand", "net_income_thousand", "eps"]:
        if col in rmw.columns:
            rmw[col] = to_num(rmw[col])
        else:
            rmw[col] = np.nan
    rmw = rmw.dropna(subset=["stock_id", "fiscal_q", "announce_date_rmw"])
    if rmw.empty:
        return pd.DataFrame()
    if "stock_name" not in rmw.columns:
        rmw["stock_name"] = ""
    return rmw[["stock_id", "stock_name", "fiscal_q", "announce_date_rmw", "revenue_thousand", "operating_income_thousand", "pretax_income_thousand", "net_income_thousand", "eps"]]


def build_financial_panel(asset_raw: pd.DataFrame, rmw_raw: pd.DataFrame) -> pd.DataFrame:
    asset = normalize_asset(asset_raw)
    rmw = normalize_rmw(rmw_raw)

    if asset.empty and rmw.empty:
        return pd.DataFrame(columns=["stock_id", "stock_name", "fiscal_q", "announce_date", "equity_total_thousand", "total_assets_thousand", "operating_income_thousand", "profitability", "asset_growth"])

    if asset.empty:
        fin = rmw.copy()
        fin["equity_total_thousand"] = np.nan
        fin["total_assets_thousand"] = np.nan
        fin["asset_growth"] = np.nan
        fin["announce_date"] = fin["announce_date_rmw"]
    elif rmw.empty:
        fin = asset.copy()
        fin["operating_income_thousand"] = np.nan
        fin["announce_date"] = fin["announce_date_asset"]
    else:
        fin = pd.merge(asset, rmw, on=["stock_id", "fiscal_q"], how="outer", suffixes=("_asset", "_rmw"))
        if "stock_name_asset" in fin.columns and "stock_name_rmw" in fin.columns:
            fin["stock_name"] = fin["stock_name_asset"].combine_first(fin["stock_name_rmw"])
        elif "stock_name" not in fin.columns:
            fin["stock_name"] = ""
        fin["announce_date"] = fin[["announce_date_asset", "announce_date_rmw"]].max(axis=1)

    if "profitability" not in fin.columns:
        fin["profitability"] = fin["operating_income_thousand"] / fin["equity_total_thousand"]
    fin["announce_date"] = parse_date(fin["announce_date"])
    cols = ["stock_id", "stock_name", "fiscal_q", "announce_date", "equity_total_thousand", "total_assets_thousand", "operating_income_thousand", "profitability", "asset_growth"]
    for c in cols:
        if c not in fin.columns:
            fin[c] = np.nan
    return fin[cols].dropna(subset=["stock_id", "announce_date"]).sort_values(["stock_id", "announce_date"]).reset_index(drop=True)


# ============================================================
# FF5 construction
# ============================================================

def merge_latest_fin_by_stock(formations: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    if fin is None or fin.empty or "announce_date" not in fin.columns:
        out = formations.copy()
        for c in ["equity_total_thousand", "total_assets_thousand", "operating_income_thousand", "profitability", "asset_growth"]:
            out[c] = np.nan
        return out

    results = []
    fin = fin.copy()
    fin["announce_date"] = parse_date(fin["announce_date"])
    fin = fin.dropna(subset=["stock_id", "announce_date"]).sort_values(["stock_id", "announce_date"])
    formations = formations.copy()
    formations["formation_date"] = parse_date(formations["formation_date"])
    formations = formations.dropna(subset=["stock_id", "formation_date"]).sort_values(["stock_id", "formation_date"])

    for stock_id, group in formations.groupby("stock_id"):
        stock_fin = fin[fin["stock_id"] == stock_id].sort_values("announce_date")
        group = group.sort_values("formation_date")
        if stock_fin.empty:
            tmp = group.copy()
            for c in ["equity_total_thousand", "total_assets_thousand", "operating_income_thousand", "profitability", "asset_growth"]:
                tmp[c] = np.nan
            results.append(tmp)
            continue
        merged = pd.merge_asof(
            group,
            stock_fin,
            left_on="formation_date",
            right_on="announce_date",
            direction="backward",
            suffixes=("", "_fin"),
        )
        if "stock_id_fin" in merged.columns:
            merged = merged.drop(columns=["stock_id_fin"])
        if "stock_name_fin" in merged.columns and "stock_name" in merged.columns:
            merged["stock_name"] = merged["stock_name"].combine_first(merged["stock_name_fin"])
        results.append(merged)

    return pd.concat(results, ignore_index=True) if results else formations


def assign_tertile(s: pd.Series, low_label: str, mid_label: str, high_label: str, neutral_label: str | None = None) -> pd.Series:
    out = pd.Series(index=s.index, dtype="object")
    valid = s.dropna()
    if valid.nunique() < 3:
        out.loc[:] = neutral_label if neutral_label is not None else mid_label
        return out
    q30, q70 = valid.quantile([0.3, 0.7])
    out.loc[s <= q30] = low_label
    out.loc[(s > q30) & (s < q70)] = mid_label
    out.loc[s >= q70] = high_label
    out.loc[out.isna()] = neutral_label if neutral_label is not None else mid_label
    return out



def normalize_rebalance_frequency(value: object) -> str:
    """Normalize user-facing rebalance frequency into canonical values."""
    raw = str(value or "monthly").strip().lower()
    mapping = {
        "m": "monthly",
        "month": "monthly",
        "monthly": "monthly",
        "每月": "monthly",
        "月調倉": "monthly",
        "月調整": "monthly",
        "monthly_rebalance": "monthly",
        "2w": "biweekly",
        "2wk": "biweekly",
        "2week": "biweekly",
        "2weeks": "biweekly",
        "biweekly": "biweekly",
        "bi-weekly": "biweekly",
        "雙周": "biweekly",
        "雙週": "biweekly",
        "雙周調倉": "biweekly",
        "雙週調倉": "biweekly",
        "兩周": "biweekly",
        "兩週": "biweekly",
    }
    return mapping.get(raw, "monthly")


def build_formation_dates(data: pd.DataFrame, rebalance_frequency: str = "monthly") -> pd.Series:
    """
    Build rebalancing formation dates from available DATA dates.

    monthly:
        每個月最後一個有 DATA 的交易日形成股票池。
    biweekly:
        每兩週最後一個有 DATA 的交易日形成股票池。
        使用 2W-FRI calendar bins；若該 bin 沒有週五交易，會取該 bin 內最後一個有資料日。
    """
    if data is None or data.empty or "date" not in data.columns:
        return pd.Series(dtype="datetime64[ns]")

    dates = (
        pd.Series(parse_date(data["date"]))
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    if dates.empty:
        return pd.Series(dtype="datetime64[ns]")

    freq = normalize_rebalance_frequency(rebalance_frequency)

    if freq == "biweekly":
        date_series = pd.Series(dates.values, index=pd.DatetimeIndex(dates.values))
        out = (
            date_series
            .resample("2W-FRI")
            .max()
            .dropna()
            .sort_values()
            .drop_duplicates()
            .reset_index(drop=True)
        )
    else:
        out = (
            pd.DataFrame({"date": dates})
            .assign(month=lambda x: x["date"].dt.to_period("M"))
            .groupby("month")["date"]
            .max()
            .sort_values()
            .reset_index(drop=True)
        )

    return pd.Series(pd.to_datetime(out, errors="coerce")).dropna().reset_index(drop=True)


def build_top_market_cap_formations(
    data: pd.DataFrame,
    fin: pd.DataFrame,
    top_n: int,
    rebalance_frequency: str = "monthly",
    progress_callback: ProgressCallback | None = None,
    progress_start: int = 30,
    progress_end: int = 45,
) -> pd.DataFrame:
    if data is None or data.empty:
        raise ValueError("Cannot build formations because DATA panel is empty.")
    data = data.copy()
    data["date"] = parse_date(data["date"])
    data = data.dropna(subset=["date", "stock_id", "market_cap_100m"])
    formation_dates = build_formation_dates(data, rebalance_frequency=rebalance_frequency)

    rows = []
    total_formations = len(formation_dates)
    for idx, formation_date in enumerate(formation_dates, start=1):
        emit_progress(
            progress_callback,
            "建立股票池",
            scaled_progress(progress_start, progress_end, idx - 1, total_formations),
            100,
            f"建立股票池: {idx}/{total_formations} - {pd.Timestamp(formation_date).date()}",
        )
        snap = data[data["date"] == formation_date].copy()
        snap = snap.dropna(subset=["market_cap_100m"])
        snap = snap.sort_values("market_cap_100m", ascending=False).head(int(top_n))
        if snap.empty:
            continue
        snap["rank"] = np.arange(1, len(snap) + 1)
        snap["formation_date"] = pd.Timestamp(formation_date)
        rows.append(snap)
        emit_progress(
            progress_callback,
            "建立股票池",
            scaled_progress(progress_start, progress_end, idx, total_formations),
            100,
            f"建立股票池: {idx}/{total_formations} - {pd.Timestamp(formation_date).date()}",
        )

    if not rows:
        raise ValueError("No market-cap formation rows were generated for the selected rebalance frequency.")

    formations = pd.concat(rows, ignore_index=True)
    formations["rebalance_frequency"] = normalize_rebalance_frequency(rebalance_frequency)
    formations = formations[["formation_date", "rebalance_frequency", "rank", "stock_id", "stock_name", "market_cap_100m", "pb", "trading_value_thousand"]]
    formations = merge_latest_fin_by_stock(formations, fin)

    formations["bm_from_fin"] = np.nan
    valid_fin = formations.get("equity_total_thousand", pd.Series(np.nan, index=formations.index)).notna()
    if valid_fin.any():
        formations.loc[valid_fin, "bm_from_fin"] = (
            formations.loc[valid_fin, "equity_total_thousand"] * 1000
        ) / (formations.loc[valid_fin, "market_cap_100m"] * 100_000_000)

    formations["bm_from_pb"] = 1 / formations["pb"]
    formations.loc[formations["pb"] <= 0, "bm_from_pb"] = np.nan
    formations["bm"] = formations["bm_from_fin"].where(formations["bm_from_fin"].notna(), formations["bm_from_pb"])

    def assign_groups(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        median_cap = group["market_cap_100m"].median()
        group["size_grp"] = np.where(group["market_cap_100m"] <= median_cap, "S", "B")
        group["bm_grp"] = assign_tertile(group["bm"], "L", "M", "H", neutral_label="M")
        group["op_grp"] = assign_tertile(group.get("profitability", pd.Series(np.nan, index=group.index)), "W", "N", "R", neutral_label="N")
        group["inv_grp"] = assign_tertile(group.get("asset_growth", pd.Series(np.nan, index=group.index)), "C", "N", "A", neutral_label="N")
        return group

    formations = pd.concat([assign_groups(g) for _, g in formations.groupby("formation_date")], ignore_index=True)
    formations["weight"] = formations["market_cap_100m"]
    return formations.sort_values(["formation_date", "rank"]).reset_index(drop=True)


def vw_ret(group: pd.DataFrame, condition: pd.Series) -> float:
    subset = group.loc[condition & group["ret"].notna() & group["weight"].notna()].copy()
    if subset.empty or subset["weight"].sum() == 0:
        return np.nan
    return float(np.average(subset["ret"], weights=subset["weight"]))


def avg_ignore_nan(values: list[float]) -> float:
    clean = [v for v in values if pd.notna(v)]
    return np.nan if not clean else float(np.mean(clean))


def zero_if_nan(x: float) -> float:
    return 0.0 if pd.isna(x) or not np.isfinite(x) else float(x)


def build_ff5_factor_returns(
    returns: pd.DataFrame,
    formations: pd.DataFrame,
    progress_callback: ProgressCallback | None = None,
    progress_start: int = 45,
    progress_end: int = 65,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fdates = list(sorted(pd.to_datetime(formations["formation_date"].dropna().unique())))
    panels = []
    returns = returns.copy()
    returns["date"] = parse_date(returns["date"])
    total_fdates = len(fdates)
    for i, fdate in enumerate(fdates):
        emit_progress(
            progress_callback,
            "建立 FF5 因子資料",
            scaled_progress(progress_start, int((progress_start + progress_end) / 2), i, max(total_fdates, 1)),
            100,
            f"建立 factor panel: {i + 1}/{total_fdates} - {pd.Timestamp(fdate).date()}",
        )
        fdate = pd.Timestamp(fdate)
        next_fdate = pd.Timestamp(fdates[i + 1]) if i + 1 < len(fdates) else returns["date"].max()
        ret_slice = returns.loc[
            (returns["date"] > fdate) & (returns["date"] <= next_fdate),
            ["date", "stock_id", "ret"],
        ].copy()
        members = formations[formations["formation_date"] == fdate].copy()
        panel = pd.merge(
            ret_slice,
            members[["formation_date", "stock_id", "stock_name", "rank", "market_cap_100m", "weight", "size_grp", "bm_grp", "op_grp", "inv_grp"]],
            on="stock_id",
            how="inner",
        )
        if not panel.empty:
            panels.append(panel)

    factor_panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    if factor_panel.empty:
        raise ValueError("No daily factor panel rows were generated. Check price dates versus formation dates.")

    def calc_ff5_one_day(group: pd.DataFrame) -> pd.Series:
        mkt = vw_ret(group, pd.Series(True, index=group.index))

        sl = vw_ret(group, (group["size_grp"] == "S") & (group["bm_grp"] == "L"))
        sm = vw_ret(group, (group["size_grp"] == "S") & (group["bm_grp"] == "M"))
        sh = vw_ret(group, (group["size_grp"] == "S") & (group["bm_grp"] == "H"))
        bl = vw_ret(group, (group["size_grp"] == "B") & (group["bm_grp"] == "L"))
        bm = vw_ret(group, (group["size_grp"] == "B") & (group["bm_grp"] == "M"))
        bh = vw_ret(group, (group["size_grp"] == "B") & (group["bm_grp"] == "H"))

        sw = vw_ret(group, (group["size_grp"] == "S") & (group["op_grp"] == "W"))
        sn = vw_ret(group, (group["size_grp"] == "S") & (group["op_grp"] == "N"))
        sr = vw_ret(group, (group["size_grp"] == "S") & (group["op_grp"] == "R"))
        bw = vw_ret(group, (group["size_grp"] == "B") & (group["op_grp"] == "W"))
        bn = vw_ret(group, (group["size_grp"] == "B") & (group["op_grp"] == "N"))
        br = vw_ret(group, (group["size_grp"] == "B") & (group["op_grp"] == "R"))

        sc = vw_ret(group, (group["size_grp"] == "S") & (group["inv_grp"] == "C"))
        sin = vw_ret(group, (group["size_grp"] == "S") & (group["inv_grp"] == "N"))
        sa = vw_ret(group, (group["size_grp"] == "S") & (group["inv_grp"] == "A"))
        bc = vw_ret(group, (group["size_grp"] == "B") & (group["inv_grp"] == "C"))
        bin_ = vw_ret(group, (group["size_grp"] == "B") & (group["inv_grp"] == "N"))
        ba = vw_ret(group, (group["size_grp"] == "B") & (group["inv_grp"] == "A"))

        smb_bm = avg_ignore_nan([sl, sm, sh]) - avg_ignore_nan([bl, bm, bh])
        smb_op = avg_ignore_nan([sw, sn, sr]) - avg_ignore_nan([bw, bn, br])
        smb_inv = avg_ignore_nan([sc, sin, sa]) - avg_ignore_nan([bc, bin_, ba])
        smb = avg_ignore_nan([smb_bm, smb_op, smb_inv])
        hml = avg_ignore_nan([sh, bh]) - avg_ignore_nan([sl, bl])
        rmw = avg_ignore_nan([sr, br]) - avg_ignore_nan([sw, bw])
        cma = avg_ignore_nan([sc, bc]) - avg_ignore_nan([sa, ba])

        return pd.Series({
            "MKT": zero_if_nan(mkt),
            "SMB": zero_if_nan(smb),
            "HML": zero_if_nan(hml),
            "RMW": zero_if_nan(rmw),
            "CMA": zero_if_nan(cma),
            "SMB_BM": zero_if_nan(smb_bm),
            "SMB_OP": zero_if_nan(smb_op),
            "SMB_INV": zero_if_nan(smb_inv),
            "n_stocks": group["stock_id"].nunique(),
        })

    grouped_days = list(factor_panel.groupby("date"))
    ff5_rows = []
    total_days = len(grouped_days)
    mid = int((progress_start + progress_end) / 2)
    for j, (dt, group) in enumerate(grouped_days, start=1):
        row = calc_ff5_one_day(group)
        row["date"] = dt
        ff5_rows.append(row)
        if j == 1 or j == total_days or j % 20 == 0:
            emit_progress(
                progress_callback,
                "計算 FF5 因子",
                scaled_progress(mid, progress_end, j, total_days),
                100,
                f"計算 FF5 因子: {j}/{total_days} - {pd.Timestamp(dt).date()}",
            )

    ff5 = pd.DataFrame(ff5_rows).sort_values("date").reset_index(drop=True)
    ff5["RF"] = 0.0
    ff5["MKT_RF"] = ff5["MKT"] - ff5["RF"]
    return factor_panel, ff5[["date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF", "MKT", "SMB_BM", "SMB_OP", "SMB_INV", "n_stocks"]]


def run_ols_alpha(reg_df: pd.DataFrame, config: FF5RawBuildConfig) -> dict[str, object] | None:
    clean = reg_df[["excess_ret"] + FACTOR_COLS].dropna().tail(int(config.lookback_days))
    if len(clean) < int(config.min_obs):
        return None
    y = clean["excess_ret"]
    x = sm.add_constant(clean[FACTOR_COLS], has_constant="add")
    try:
        model = sm.OLS(y, x).fit()
    except Exception:
        return None
    alpha = model.params.get("const", np.nan)
    return {
        "alpha": alpha,
        "alpha_annualized": alpha * 252 if pd.notna(alpha) else np.nan,
        "pvalue_alpha": model.pvalues.get("const", np.nan),
        "beta_mkt": model.params.get("MKT_RF", np.nan),
        "beta_smb": model.params.get("SMB", np.nan),
        "beta_hml": model.params.get("HML", np.nan),
        "beta_rmw": model.params.get("RMW", np.nan),
        "beta_cma": model.params.get("CMA", np.nan),
        "pvalue_mkt": model.pvalues.get("MKT_RF", np.nan),
        "pvalue_smb": model.pvalues.get("SMB", np.nan),
        "pvalue_hml": model.pvalues.get("HML", np.nan),
        "pvalue_rmw": model.pvalues.get("RMW", np.nan),
        "pvalue_cma": model.pvalues.get("CMA", np.nan),
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "n_obs": int(model.nobs),
        "reg_start": clean.index.min(),
        "reg_end": clean.index.max(),
    }


def next_trading_date(trading_dates: pd.DatetimeIndex, current_date: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    current_date = pd.Timestamp(current_date)
    future = trading_dates[trading_dates > current_date]
    if len(future) == 0:
        return pd.NaT
    return pd.Timestamp(future[0])


def build_rolling_alpha_scores(
    ff5: pd.DataFrame,
    formations: pd.DataFrame,
    returns: pd.DataFrame,
    config: FF5RawBuildConfig,
    progress_callback: ProgressCallback | None = None,
    progress_start: int = 65,
    progress_end: int = 95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ff5 = ff5.copy()
    ff5["date"] = parse_date(ff5["date"])
    for col in FACTOR_COLS + ["RF"]:
        if col not in ff5.columns:
            ff5[col] = 0.0
        ff5[col] = pd.to_numeric(ff5[col], errors="coerce").fillna(0.0)
    ff5 = ff5[["date"] + FACTOR_COLS + ["RF"]].dropna(subset=["date"]).drop_duplicates("date").sort_values("date")

    stock_returns = returns[["date", "stock_id", "stock_name", "adj_close", "ret"]].copy()
    stock_returns["date"] = parse_date(stock_returns["date"])
    panel = pd.merge(stock_returns, ff5, on="date", how="inner")
    panel["excess_ret"] = panel["ret"] - panel["RF"]
    panel = panel.sort_values(["stock_id", "date"]).reset_index(drop=True)

    stock_panel = {
        stock_id: group.set_index("date").sort_index()
        for stock_id, group in panel.groupby("stock_id")
    }

    trading_dates = pd.DatetimeIndex(sorted(ff5["date"].dropna().unique()))
    formation_dates = sorted(pd.to_datetime(formations["formation_date"].dropna().unique()))
    formation_meta = []
    for i, fdate in enumerate(formation_dates):
        fdate = pd.Timestamp(fdate)
        formation_meta.append({
            "formation_date": fdate,
            "holding_start": next_trading_date(trading_dates, fdate),
            "holding_end": pd.Timestamp(formation_dates[i + 1]) if i + 1 < len(formation_dates) else pd.NaT,
        })
    formation_meta_df = pd.DataFrame(formation_meta)

    rows = []
    total_alpha_dates = len(formation_dates)
    for fidx, fdate in enumerate(formation_dates, start=1):
        emit_progress(
            progress_callback,
            "估計 rolling alpha",
            scaled_progress(progress_start, progress_end, fidx - 1, total_alpha_dates),
            100,
            f"估計 rolling alpha: {fidx}/{total_alpha_dates} - {pd.Timestamp(fdate).date()}",
        )
        fdate = pd.Timestamp(fdate)
        members = formations[formations["formation_date"] == fdate].copy()
        for _, row in members.iterrows():
            stock_id = row["stock_id"]
            if stock_id not in stock_panel:
                continue
            history = stock_panel[stock_id]
            history = history[history.index <= fdate]
            valid = history[["excess_ret"] + FACTOR_COLS].dropna().tail(int(config.lookback_days))
            if len(valid) < int(config.min_obs):
                continue
            result = run_ols_alpha(valid, config)
            if result is None:
                continue
            rows.append({
                "formation_date": fdate,
                "stock_id": stock_id,
                "stock_name": row.get("stock_name", np.nan),
                "rank_in_top150": row.get("rank", np.nan),
                **result,
            })
        emit_progress(
            progress_callback,
            "估計 rolling alpha",
            scaled_progress(progress_start, progress_end, fidx, total_alpha_dates),
            100,
            f"估計 rolling alpha: {fidx}/{total_alpha_dates} - {pd.Timestamp(fdate).date()}；累計 alpha={len(rows)}",
        )

    alpha_scores = pd.DataFrame(rows)
    if alpha_scores.empty:
        raise ValueError(
            "No alpha scores were generated. "
            f"Try reducing min_obs ({config.min_obs}) or lookback_days ({config.lookback_days}), "
            "or verify that price data has enough history before the first formation month."
        )

    alpha_scores = pd.merge(alpha_scores, formation_meta_df, on="formation_date", how="left")
    alpha_scores["alpha_rank"] = alpha_scores.groupby("formation_date")["alpha"].rank(ascending=False, method="first")
    alpha_scores = alpha_scores.sort_values(["formation_date", "alpha_rank", "stock_id"]).reset_index(drop=True)
    return alpha_scores, panel


def run_ff5_raw_pipeline(
    raw_dir: str | Path,
    config: FF5RawBuildConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, pd.DataFrame]:
    start, end = _date_bounds(config)
    emit_progress(progress_callback, "掃描原始資料", 1, 100, "掃描原始資料資料夾...")
    files = discover_strategy3_raw_files(raw_dir)

    data_raw = read_many_excel(files["data"], progress_callback, "讀取 DATA 檔", 2, 12)
    ad_raw = read_many_excel(files["ad_price"], progress_callback, "讀取 AD PRICE 檔", 12, 18) if files["ad_price"] else pd.DataFrame()
    asset_raw = read_many_excel(files["asset"], progress_callback, "讀取 ASSET 檔", 18, 22) if files["asset"] else pd.DataFrame()
    rmw_raw = read_many_excel(files["rmw"], progress_callback, "讀取 RMW 檔", 22, 26) if files["rmw"] else pd.DataFrame()

    emit_progress(progress_callback, "標準化原始資料", 27, 100, "標準化 DATA / AD PRICE / 財報欄位...")
    data = normalize_market_data(data_raw, start, end)
    price_df = normalize_ad_price(ad_raw, data=data, start_date=start, end_date=end)
    returns = price_df[["date", "stock_id", "stock_name", "adj_close", "ret"]].copy()

    fin = build_financial_panel(asset_raw, rmw_raw)
    formations = build_top_market_cap_formations(
        data,
        fin,
        config.market_cap_top_n,
        rebalance_frequency=config.rebalance_frequency,
        progress_callback=progress_callback,
        progress_start=30,
        progress_end=45,
    )
    factor_panel, ff5 = build_ff5_factor_returns(
        returns,
        formations,
        progress_callback=progress_callback,
        progress_start=45,
        progress_end=65,
    )
    alpha_scores, regression_panel = build_rolling_alpha_scores(
        ff5,
        formations,
        returns,
        config,
        progress_callback=progress_callback,
        progress_start=65,
        progress_end=95,
    )

    # Data validation: do not pretend success if core outputs are empty.
    if data.empty or price_df.empty or formations.empty or ff5.empty or alpha_scores.empty:
        raise ValueError(
            "Pipeline output validation failed: "
            f"data={len(data)}, price_df={len(price_df)}, formations={len(formations)}, ff5={len(ff5)}, alpha_scores={len(alpha_scores)}"
        )

    emit_progress(progress_callback, "完成資料建構", 100, 100, "FF5 pipeline 完成，準備回傳資料。")

    return {
        "file_manifest": build_file_manifest(files),
        "ff5": ff5,
        "formations": formations,
        "factor_panel": factor_panel,
        "alpha_scores": alpha_scores,
        "price_df": price_df,
        "regression_panel": regression_panel,
    }
