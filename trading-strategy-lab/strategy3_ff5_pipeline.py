from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import re

import numpy as np
import pandas as pd
import statsmodels.api as sm


FACTOR_COLS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]


@dataclass(frozen=True)
class FF5RawBuildConfig:
    market_cap_top_n: int = 150
    lookback_days: int = 252
    min_obs: int = 180
    # Strategy 3 requested backtest/data window: 2019 through 2026.
    # The pipeline keeps enough rows inside this window and the Streamlit app
    # later uses backtest_start_date/backtest_end_date for the actual NAV range.
    start_date: str | pd.Timestamp | None = "2019-01-01"
    end_date: str | pd.Timestamp | None = "2026-12-31"


def run_ff5_raw_pipeline(raw_dir: str | Path, config: FF5RawBuildConfig) -> dict[str, pd.DataFrame]:
    """
    Robust Strategy 3 FF5 pipeline.

    Key fixes:
    - classify raw files by both file name and parent folder name, so folders like
      data/, ad_price/, asset/, rmw/ work even when filenames only contain dates.
    - DATA is required; AD PRICE is optional and falls back to DATA close price.
    - ASSET/RMW are optional; if missing or unusable, HML/SMB can still be built
      from DATA and RMW/CMA neutral buckets are used instead of crashing.
    - all outputs use canonical columns: date, stock_id, stock_name, adj_close.
    """
    files = discover_strategy3_raw_files(raw_dir)

    data_raw = read_many_excel(files.get("data", []), required=True, label="DATA")
    ad_raw = read_many_excel(files.get("ad_price", []), required=False, label="AD PRICE")
    asset_raw = read_many_excel(files.get("asset", []), required=False, label="ASSET")
    rmw_raw = read_many_excel(files.get("rmw", []), required=False, label="RMW")

    data = normalize_market_data(data_raw)

    if ad_raw is not None and not ad_raw.empty:
        price_df = normalize_ad_price(ad_raw)
    else:
        price_df = normalize_ad_price(data_raw)

    fin = build_financial_panel(asset_raw, rmw_raw)

    start_ts = pd.Timestamp(config.start_date).normalize() if config.start_date is not None else None
    end_ts = pd.Timestamp(config.end_date).normalize() if config.end_date is not None else None
    if start_ts is not None:
        data = data[data["date"] >= start_ts].copy()
        price_df = price_df[price_df["date"] >= start_ts].copy()
    if end_ts is not None:
        data = data[data["date"] <= end_ts].copy()
        price_df = price_df[price_df["date"] <= end_ts].copy()

    if data.empty:
        raise ValueError("DATA files have no valid rows after the requested date filtering.")
    if price_df.empty:
        raise ValueError("Price data has no valid rows after the requested date filtering.")

    returns = price_df[["date", "stock_id", "stock_name", "adj_close", "ret"]].copy()
    formations = build_top_market_cap_formations(data, fin, config.market_cap_top_n)
    factor_panel, ff5 = build_ff5_factor_returns(returns, formations)
    alpha_scores, regression_panel = build_rolling_alpha_scores(ff5, formations, returns, config)

    return {
        "file_manifest": build_file_manifest(files),
        "ff5": ff5,
        "formations": formations,
        "factor_panel": factor_panel,
        "alpha_scores": alpha_scores,
        "price_df": price_df,
        "regression_panel": regression_panel,
    }


def discover_strategy3_raw_files(raw_dir: str | Path) -> dict[str, list[Path]]:
    """Discover CMoney raw files by filename OR folder name."""
    base = Path(raw_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Strategy 3 raw data folder not found: {base}")

    candidates = [
        p for p in base.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".xlsx", ".xls", ".csv"}
        and not p.name.startswith("~$")
    ]

    def key(path: Path) -> str:
        rel = str(path.relative_to(base)).lower().replace("\\", "/")
        return rel.replace("-", "_").replace(" ", "_")

    def classify(path: Path) -> str:
        s = key(path)
        name = path.name.lower().replace("-", "_").replace(" ", "_")
        if any(tok in s for tok in ["a09012", "benchmark", "奔騰", "統一奔騰"]):
            return "ignore"
        if any(tok in s for tok in ["ad_price", "adprice", "adj_price", "adjusted_price", "還原"]):
            return "ad_price"
        if any(tok in s for tok in ["/asset/", "asset/", "_asset", "asset", "資產負債"]):
            return "asset"
        if any(tok in s for tok in ["/rmw/", "rmw/", "_rmw", "rmw", "損益", "profit"]):
            return "rmw"
        if any(tok in s for tok in ["/data/", "data/", "_data", "data", "日收盤", "市值"]):
            return "data"
        if any(tok in name for tok in ["price", "收盤"]):
            return "ad_price"
        return "other"

    groups: dict[str, list[Path]] = {"data": [], "ad_price": [], "asset": [], "rmw": []}
    for p in candidates:
        c = classify(p)
        if c in groups:
            groups[c].append(p)
    for k in groups:
        groups[k] = sorted(groups[k])

    if not groups["data"]:
        found = "\n".join(str(p.relative_to(base)) for p in candidates[:120])
        raise FileNotFoundError(
            f"Missing raw DATA files. Checked folder: {base}\nFound files:\n{found}"
        )
    return groups


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
    return pd.DataFrame(rows, columns=["group", "file_name", "path", "size_mb"])


def read_many_excel(paths: list[Path], required: bool = True, label: str = "Excel") -> pd.DataFrame:
    """Read CSV/XLS/XLSX files and concatenate; try multiple header rows."""
    dfs = []
    for path in paths:
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                last_error = None
                for enc in ["utf-8-sig", "utf-8", "big5", "cp950"]:
                    try:
                        df = pd.read_csv(path, encoding=enc)
                        break
                    except Exception as exc:
                        last_error = exc
                else:
                    raise last_error  # type: ignore[misc]
            elif suffix in {".xlsx", ".xls"}:
                df = None
                errors = []
                for header in [0, 1, 2, 3, 4, 5]:
                    try:
                        tmp = pd.read_excel(path, header=header)
                        tmp.columns = [str(c).strip() for c in tmp.columns]
                        usable_cols = [c for c in tmp.columns if c and not c.lower().startswith("unnamed") and c.lower() != "nan"]
                        joined = "|".join(usable_cols)
                        if len(usable_cols) >= 2 and any(tok in joined for tok in ["日期", "年月", "年季", "股票", "證券", "date", "stock"]):
                            df = tmp
                            break
                        if df is None and len(usable_cols) >= 4:
                            df = tmp
                    except Exception as exc:
                        errors.append(str(exc))
                if df is None:
                    raise ValueError(f"Unable to read {path.name}; {errors[-1] if errors else ''}")
            else:
                continue
            df.columns = [str(c).strip() for c in df.columns]
            df = df.loc[:, [c for c in df.columns if c and not str(c).lower().startswith("unnamed")]]
            df["source_file"] = path.name
            dfs.append(df)
        except Exception as exc:
            if required:
                raise ValueError(f"Failed to read {label} file {path.name}: {exc}") from exc
            continue
    if not dfs:
        if required:
            raise FileNotFoundError(f"No readable {label} files were provided.")
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)


def clean_stock_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)")[0].str.zfill(4)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("--", "", regex=False)
        .str.replace("- -", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def parse_date(s: pd.Series) -> pd.Series:
    raw = s.copy()
    raw_str = raw.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    is_yyyymmdd = raw_str.str.fullmatch(r"\d{8}")
    if is_yyyymmdd.any():
        out.loc[is_yyyymmdd] = pd.to_datetime(raw_str.loc[is_yyyymmdd], format="%Y%m%d", errors="coerce")

    remain = ~is_yyyymmdd
    out.loc[remain] = pd.to_datetime(raw.loc[remain], errors="coerce")

    numeric = pd.to_numeric(raw, errors="coerce")
    is_excel_serial = numeric.between(20000, 60000) & raw_str.str.fullmatch(r"\d+(\.0)?")
    if is_excel_serial.any():
        out.loc[is_excel_serial] = pd.to_datetime(
            numeric.loc[is_excel_serial],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    return out.dt.normalize().astype("datetime64[ns]")


def parse_cmoney_quarter(s: pd.Series) -> pd.Series:
    parsed = s.astype(str).str.extract(r"(\d{4})\D+(\d{1,2})")
    years = pd.to_numeric(parsed[0], errors="coerce")
    quarters = pd.to_numeric(parsed[1], errors="coerce")
    out = pd.Series(pd.NaT, index=s.index, dtype=object)
    valid = years.notna() & quarters.notna()
    if valid.any():
        out.loc[valid] = pd.PeriodIndex.from_fields(
            year=years.loc[valid].astype(int),
            quarter=quarters.loc[valid].astype(int),
            freq="Q",
        )
    return out



# ============================================================
# Robust CMoney Column Helpers
# ============================================================

def canonical_col_key(col: object) -> str:
    """Canonicalize CMoney / English column names for loose matching."""
    s = str(col).strip().lower()
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[\s_\-./\\:：()\[\]{}]+", "", s)
    s = s.replace("％", "%")
    return s


def find_column(df: pd.DataFrame, candidates: list[str], contains: list[str] | None = None) -> str | None:
    if df is None or df.empty:
        return None
    cols = [str(c).strip() for c in df.columns]
    direct = {c: c for c in cols}
    for c in candidates:
        if c in direct:
            return direct[c]
    cmap = {canonical_col_key(c): c for c in cols}
    for c in candidates:
        key = canonical_col_key(c)
        if key in cmap:
            return cmap[key]
    if contains:
        contains_keys = [canonical_col_key(c) for c in contains]
        for col in cols:
            key = canonical_col_key(col)
            if all(token in key for token in contains_keys):
                return col
    return None


def prepare_stock_id_column(df: pd.DataFrame, context: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if "stock_id" not in out.columns:
        stock_col = find_column(out, [
            "stock_id", "stock_code", "code", "ticker", "symbol", "證券代碼", "證券代號",
            "股票代號", "股票代碼", "公司代號", "代號", "簡稱代號",
        ])
        if stock_col is None and out.index.name is not None:
            idx_key = canonical_col_key(out.index.name)
            if idx_key in {canonical_col_key(c) for c in ["stock_id", "股票代號", "證券代碼", "代號"]}:
                out = out.reset_index()
                stock_col = str(out.columns[0])
        if stock_col is None:
            raise ValueError(f"{context} 缺少 stock_id / 股票代號欄位，目前欄位：{list(out.columns)}")
        out = out.rename(columns={stock_col: "stock_id"})

    out["stock_id"] = clean_stock_id(out["stock_id"])
    out = out[out["stock_id"].notna()].copy()
    return out


def filter_by_date_range(df: pd.DataFrame, date_col: str, start_date=None, end_date=None) -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = parse_date(out[date_col])
    if start_date is not None:
        out = out[out[date_col] >= pd.Timestamp(start_date).normalize()].copy()
    if end_date is not None:
        out = out[out[date_col] <= pd.Timestamp(end_date).normalize()].copy()
    return out

def normalize_market_data(data_raw: pd.DataFrame) -> pd.DataFrame:
    df = prepare_stock_id_column(data_raw, "DATA files").copy()

    date_col = find_column(df, ["date", "trading_date", "年月日", "日期"])
    name_col = find_column(df, ["stock_name", "name", "股票名稱", "證券名稱", "簡稱", "公司名稱"])
    close_col = find_column(df, ["close", "收盤價", "收盤價(元)", "收盤", "Close"])
    volume_col = find_column(df, ["volume", "成交量", "成交股數", "成交量(千股)", "成交量(張)"])
    tv_col = find_column(df, ["trading_value_thousand", "成交金額(千)", "成交金額", "成交值", "成交金額(元)"])
    cap_col = find_column(df, ["capital_million", "股本(百萬)", "股本", "實收資本額(百萬)"])
    mcap_col = find_column(df, [
        "market_cap_100m", "總市值(億)", "總市值", "市值(億)", "市場總值(億)",
        "market_cap", "市值", "總市值(百萬)", "市場總值",
    ])
    pb_col = find_column(df, ["pb", "股價淨值比", "PBR", "PB", "price_to_book"])
    turnover_col = find_column(df, ["turnover_pct", "週轉率(%)", "週轉率", "turnover"])

    missing = []
    if date_col is None:
        missing.append("date/日期")
    if mcap_col is None:
        missing.append("market_cap_100m/總市值(億)")
    if missing:
        raise ValueError(f"DATA files are missing required columns: {missing}. Current columns: {list(df.columns)}")

    out = pd.DataFrame({
        "date": parse_date(df[date_col]),
        "stock_id": clean_stock_id(df["stock_id"]),
        "stock_name": df[name_col].astype(str).str.strip() if name_col is not None else "",
        "close": to_num(df[close_col]) if close_col is not None else np.nan,
        "volume": to_num(df[volume_col]) if volume_col is not None else np.nan,
        "trading_value_thousand": to_num(df[tv_col]) if tv_col is not None else np.nan,
        "capital_million": to_num(df[cap_col]) if cap_col is not None else np.nan,
        "market_cap_100m": to_num(df[mcap_col]),
        "pb": to_num(df[pb_col]) if pb_col is not None else np.nan,
        "turnover_pct": to_num(df[turnover_col]) if turnover_col is not None else np.nan,
    })

    return (
        out.dropna(subset=["date", "stock_id", "market_cap_100m"])
        .drop_duplicates(["date", "stock_id"], keep="last")
        .sort_values(["date", "stock_id"])
        .reset_index(drop=True)
    )

def normalize_ad_price(ad_raw: pd.DataFrame) -> pd.DataFrame:
    df = prepare_stock_id_column(ad_raw, "AD PRICE files").copy()

    date_col = find_column(df, ["date", "trading_date", "年月日", "日期"])
    name_col = find_column(df, ["stock_name", "name", "股票名稱", "證券名稱", "簡稱", "公司名稱"])
    close_col = find_column(df, [
        "adj_close", "adjusted_close", "還原收盤價", "調整收盤價", "收盤價", "收盤價(元)",
        "close", "Close", "price",
    ])
    volume_col = find_column(df, ["volume", "成交量", "成交股數", "成交量(千股)", "成交量(張)"])
    tv_col = find_column(df, ["trading_value_thousand", "成交金額(千)", "成交金額", "成交值", "成交金額(元)"])
    mcap_col = find_column(df, ["market_cap_100m", "總市值(億)", "總市值", "市值(億)", "市值", "市場總值"])

    missing = []
    if date_col is None:
        missing.append("date/日期")
    if close_col is None:
        missing.append("adj_close/收盤價")
    if missing:
        raise ValueError(f"AD PRICE files are missing required columns: {missing}. Current columns: {list(df.columns)}")

    out = pd.DataFrame({
        "date": parse_date(df[date_col]),
        "stock_id": clean_stock_id(df["stock_id"]),
        "stock_name": df[name_col].astype(str).str.strip() if name_col is not None else "",
        "adj_close": to_num(df[close_col]),
        "volume": to_num(df[volume_col]) if volume_col is not None else np.nan,
        "trading_value_thousand": to_num(df[tv_col]) if tv_col is not None else np.nan,
        "market_cap_100m": to_num(df[mcap_col]) if mcap_col is not None else np.nan,
    })

    out = (
        out.dropna(subset=["date", "stock_id", "adj_close"])
        .query("adj_close > 0")
        .drop_duplicates(["date", "stock_id"], keep="last")
        .sort_values(["stock_id", "date"])
        .reset_index(drop=True)
    )
    out["ret"] = out.groupby("stock_id")["adj_close"].pct_change()
    out.loc[~np.isfinite(out["ret"]), "ret"] = np.nan
    return out.sort_values(["date", "stock_id"]).reset_index(drop=True)

def build_financial_panel(asset_raw: pd.DataFrame, rmw_raw: pd.DataFrame) -> pd.DataFrame:
    """Build financial characteristics. ASSET/RMW are optional."""
    empty_cols = [
        "stock_id", "stock_name", "fiscal_q", "announce_date",
        "equity_total_thousand", "total_assets_thousand",
        "operating_income_thousand", "profitability", "asset_growth",
    ]

    def empty_panel() -> pd.DataFrame:
        return pd.DataFrame(columns=empty_cols)

    def empty_asset() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "stock_id", "stock_name", "fiscal_q", "announce_date_asset",
            "total_assets_thousand", "total_liabilities_thousand",
            "equity_total_thousand", "asset_growth",
        ])

    def empty_rmw() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "stock_id", "stock_name", "fiscal_q", "announce_date_rmw",
            "revenue_thousand", "operating_income_thousand", "pretax_income_thousand",
            "net_income_thousand", "eps",
        ])

    # ASSET panel
    asset_out = empty_asset()
    if asset_raw is not None and not asset_raw.empty:
        try:
            asset = prepare_stock_id_column(asset_raw, "ASSET files").copy()
            asset_q_col = find_column(asset, ["year_quarter", "年季", "季度", "季別", "年月", "財報年月"])
            if asset_q_col is not None:
                asset_name_col = find_column(asset, ["stock_name", "name", "股票名稱", "證券名稱", "簡稱", "公司簡稱"])
                total_assets_col = find_column(asset, ["total_assets_thousand", "資產總計(千)", "資產總計", "總資產", "資產總額"])
                liabilities_col = find_column(asset, ["total_liabilities_thousand", "負債總計(千)", "負債總計", "總負債", "負債總額"])
                equity_col = find_column(asset, ["equity_total_thousand", "權益總計(千)", "股東權益總計", "權益總計", "淨值", "權益總額", "股東權益"])
                announce_asset_col = find_column(asset, ["announce_date_asset", "公告日期", "發布日期", "date", "日期"])
                fiscal_q = parse_cmoney_quarter(asset[asset_q_col])
                announce = parse_date(asset[announce_asset_col]) if announce_asset_col is not None else pd.Series(pd.NaT, index=asset.index)
                asset_out = pd.DataFrame({
                    "stock_id": clean_stock_id(asset["stock_id"]),
                    "stock_name": asset[asset_name_col].astype(str).str.strip() if asset_name_col is not None else "",
                    "fiscal_q": fiscal_q,
                    "announce_date_asset": announce,
                    "total_assets_thousand": to_num(asset[total_assets_col]) if total_assets_col is not None else np.nan,
                    "total_liabilities_thousand": to_num(asset[liabilities_col]) if liabilities_col is not None else np.nan,
                    "equity_total_thousand": to_num(asset[equity_col]) if equity_col is not None else np.nan,
                })
                miss_ann = asset_out["announce_date_asset"].isna() & asset_out["fiscal_q"].notna()
                if miss_ann.any():
                    asset_out.loc[miss_ann, "announce_date_asset"] = asset_out.loc[miss_ann, "fiscal_q"].apply(lambda p: p.end_time.normalize() + pd.Timedelta(days=90))
                asset_out = asset_out.dropna(subset=["stock_id", "fiscal_q"]).sort_values(["stock_id", "fiscal_q"])
                asset_out["asset_growth"] = asset_out.groupby("stock_id")["total_assets_thousand"].pct_change(4)
                asset_out = asset_out[[
                    "stock_id", "stock_name", "fiscal_q", "announce_date_asset",
                    "total_assets_thousand", "total_liabilities_thousand", "equity_total_thousand", "asset_growth",
                ]]
        except Exception:
            asset_out = empty_asset()

    # RMW/profitability panel
    rmw_out = empty_rmw()
    if rmw_raw is not None and not rmw_raw.empty:
        try:
            rmw = prepare_stock_id_column(rmw_raw, "RMW files").copy()
            rmw_q_col = find_column(rmw, ["year_quarter", "年季", "季度", "季別", "年月", "財報年月"])
            if rmw_q_col is not None:
                rmw_name_col = find_column(rmw, ["stock_name", "name", "股票名稱", "證券名稱", "簡稱", "公司簡稱"])
                announce_rmw_col = find_column(rmw, ["announce_date_rmw", "公告日期", "發布日期", "date", "日期"])
                revenue_col = find_column(rmw, ["revenue_thousand", "營業收入淨額(千)", "營業收入淨額", "營收", "營業收入"])
                op_income_col = find_column(rmw, ["operating_income_thousand", "營業利益(千)", "營業利益", "營業淨利", "營業利益淨額", "營業利益損失"])
                pretax_col = find_column(rmw, ["pretax_income_thousand", "稅前純益(千)", "稅前純益", "稅前淨利", "稅前淨利淨損"])
                net_income_col = find_column(rmw, ["net_income_thousand", "稅後純益(千)", "稅後純益", "本期淨利", "淨利", "稅後淨利"])
                eps_col = find_column(rmw, ["eps", "每股稅後盈餘(元)", "每股盈餘", "EPS"])
                fiscal_q = parse_cmoney_quarter(rmw[rmw_q_col])
                announce = parse_date(rmw[announce_rmw_col]) if announce_rmw_col is not None else pd.Series(pd.NaT, index=rmw.index)
                op_income = to_num(rmw[op_income_col]) if op_income_col is not None else pd.Series(np.nan, index=rmw.index)
                pretax = to_num(rmw[pretax_col]) if pretax_col is not None else pd.Series(np.nan, index=rmw.index)
                net_income = to_num(rmw[net_income_col]) if net_income_col is not None else pd.Series(np.nan, index=rmw.index)
                op_income = op_income.where(op_income.notna(), pretax).where(lambda x: x.notna(), net_income)
                rmw_out = pd.DataFrame({
                    "stock_id": clean_stock_id(rmw["stock_id"]),
                    "stock_name": rmw[rmw_name_col].astype(str).str.strip() if rmw_name_col is not None else "",
                    "fiscal_q": fiscal_q,
                    "announce_date_rmw": announce,
                    "revenue_thousand": to_num(rmw[revenue_col]) if revenue_col is not None else np.nan,
                    "operating_income_thousand": op_income,
                    "pretax_income_thousand": pretax,
                    "net_income_thousand": net_income,
                    "eps": to_num(rmw[eps_col]) if eps_col is not None else np.nan,
                })
                miss_ann = rmw_out["announce_date_rmw"].isna() & rmw_out["fiscal_q"].notna()
                if miss_ann.any():
                    rmw_out.loc[miss_ann, "announce_date_rmw"] = rmw_out.loc[miss_ann, "fiscal_q"].apply(lambda p: p.end_time.normalize() + pd.Timedelta(days=90))
                rmw_out = rmw_out.dropna(subset=["stock_id", "fiscal_q"])
                rmw_out = rmw_out[[
                    "stock_id", "stock_name", "fiscal_q", "announce_date_rmw",
                    "revenue_thousand", "operating_income_thousand", "pretax_income_thousand", "net_income_thousand", "eps",
                ]]
        except Exception:
            rmw_out = empty_rmw()

    if asset_out.empty and rmw_out.empty:
        return empty_panel()

    fin = pd.merge(asset_out, rmw_out, on=["stock_id", "fiscal_q"], how="outer", suffixes=("_asset", "_rmw"))
    if fin.empty:
        return empty_panel()
    fin["stock_name"] = fin.get("stock_name_asset", pd.Series(index=fin.index, dtype=object)).combine_first(
        fin.get("stock_name_rmw", pd.Series(index=fin.index, dtype=object))
    )
    for col in ["announce_date_asset", "announce_date_rmw"]:
        if col not in fin.columns:
            fin[col] = pd.NaT
    fin["announce_date"] = fin[["announce_date_asset", "announce_date_rmw"]].max(axis=1)
    for col in ["equity_total_thousand", "total_assets_thousand", "operating_income_thousand", "asset_growth"]:
        if col not in fin.columns:
            fin[col] = np.nan
    fin["profitability"] = fin["operating_income_thousand"] / fin["equity_total_thousand"]
    fin = fin[empty_cols]
    return fin.dropna(subset=["stock_id", "announce_date"]).sort_values(["stock_id", "announce_date"]).reset_index(drop=True)


def merge_latest_fin_by_stock(formations: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    if formations is None or formations.empty:
        return pd.DataFrame()
    if fin is None or fin.empty or "stock_id" not in fin.columns or "announce_date" not in fin.columns:
        out = formations.copy()
        for col in [
            "fiscal_q", "announce_date", "equity_total_thousand", "total_assets_thousand",
            "operating_income_thousand", "profitability", "asset_growth",
        ]:
            if col not in out.columns:
                out[col] = np.nan
        return out

    results = []
    fin = fin.sort_values(["stock_id", "announce_date"])
    formations = formations.sort_values(["stock_id", "formation_date"])

    for stock_id, group in formations.groupby("stock_id"):
        stock_fin = fin[fin["stock_id"] == stock_id].sort_values("announce_date")
        group = group.sort_values("formation_date")
        if stock_fin.empty:
            results.append(group)
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
            merged["stock_name"] = merged["stock_name"].where(merged["stock_name"].notna(), merged["stock_name_fin"])
            merged = merged.drop(columns=["stock_name_fin"])
        results.append(merged)

    return pd.concat(results, ignore_index=True) if results else formations

def assign_tertile(s: pd.Series, low_label: str, mid_label: str, high_label: str) -> pd.Series:
    out = pd.Series(index=s.index, dtype="object")
    valid = s.dropna()
    if valid.nunique() < 3:
        return out
    q30, q70 = valid.quantile([0.3, 0.7])
    out.loc[s <= q30] = low_label
    out.loc[(s > q30) & (s < q70)] = mid_label
    out.loc[s >= q70] = high_label
    return out


def build_top_market_cap_formations(data: pd.DataFrame, fin: pd.DataFrame, top_n: int) -> pd.DataFrame:
    data = data.copy()
    data["date"] = parse_date(data["date"])
    formation_dates = (
        data[["date"]]
        .drop_duplicates()
        .assign(month=lambda x: x["date"].dt.to_period("M"))
        .groupby("month")["date"]
        .max()
        .sort_values()
        .reset_index(drop=True)
    )

    rows = []
    for formation_date in formation_dates:
        snap = data[data["date"] == formation_date].copy()
        snap = snap.dropna(subset=["market_cap_100m"])
        snap = snap.sort_values("market_cap_100m", ascending=False).head(int(top_n))
        if snap.empty:
            continue
        snap["rank"] = np.arange(1, len(snap) + 1)
        snap["formation_date"] = pd.Timestamp(formation_date)
        rows.append(snap)

    if not rows:
        raise ValueError("No monthly market-cap formation rows were generated.")

    formations = pd.concat(rows, ignore_index=True)
    keep = [
        "formation_date", "rank", "stock_id", "stock_name",
        "market_cap_100m", "pb", "trading_value_thousand",
    ]
    for col in keep:
        if col not in formations.columns:
            formations[col] = np.nan
    formations = formations[keep]
    formations = merge_latest_fin_by_stock(formations, fin)

    if "equity_total_thousand" not in formations.columns:
        formations["equity_total_thousand"] = np.nan
    if "profitability" not in formations.columns:
        formations["profitability"] = np.nan
    if "asset_growth" not in formations.columns:
        formations["asset_growth"] = np.nan

    formations["bm_from_fin"] = (
        formations["equity_total_thousand"] * 1000
    ) / (formations["market_cap_100m"] * 100_000_000)
    formations["bm_from_pb"] = 1 / formations["pb"]
    formations.loc[formations["pb"] <= 0, "bm_from_pb"] = np.nan
    formations["bm"] = formations["bm_from_fin"].where(formations["bm_from_fin"].notna(), formations["bm_from_pb"])

    # If financial characteristics are missing, use neutral/zero characteristics.
    # This prevents hard failures while still allowing MKT and SMB/HML construction.
    formations["profitability"] = pd.to_numeric(formations["profitability"], errors="coerce")
    formations["asset_growth"] = pd.to_numeric(formations["asset_growth"], errors="coerce")

    def assign_groups(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        median_cap = group["market_cap_100m"].median()
        group["size_grp"] = np.where(group["market_cap_100m"] <= median_cap, "S", "B")
        group["bm_grp"] = assign_tertile(group["bm"], "L", "M", "H")
        group["op_grp"] = assign_tertile(group["profitability"], "W", "N", "R")
        group["inv_grp"] = assign_tertile(group["asset_growth"], "C", "N", "A")
        # Fill missing classifications with neutral buckets so factor calculation does not collapse.
        group["bm_grp"] = group["bm_grp"].fillna("M")
        group["op_grp"] = group["op_grp"].fillna("N")
        group["inv_grp"] = group["inv_grp"].fillna("N")
        return group

    formations = pd.concat(
        [assign_groups(group) for _, group in formations.groupby("formation_date")],
        ignore_index=True,
    )
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


def build_ff5_factor_returns(returns: pd.DataFrame, formations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fdates = list(sorted(pd.to_datetime(formations["formation_date"].dropna().unique())))
    panels = []
    for i, fdate in enumerate(fdates):
        fdate = pd.Timestamp(fdate)
        next_fdate = pd.Timestamp(fdates[i + 1]) if i + 1 < len(fdates) else returns["date"].max()
        ret_slice = returns.loc[
            (returns["date"] > fdate) & (returns["date"] <= next_fdate),
            ["date", "stock_id", "ret"],
        ].copy()
        if ret_slice.empty:
            continue
        members = formations[formations["formation_date"] == fdate].copy()
        for col in ["size_grp", "bm_grp", "op_grp", "inv_grp"]:
            if col not in members.columns:
                members[col] = "N" if col in {"op_grp", "inv_grp"} else "M"
        panel = pd.merge(
            ret_slice,
            members[[
                "formation_date", "stock_id", "stock_name", "rank",
                "market_cap_100m", "weight", "size_grp", "bm_grp", "op_grp", "inv_grp",
            ]],
            on="stock_id",
            how="inner",
        )
        if not panel.empty:
            panels.append(panel)

    factor_panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    if factor_panel.empty:
        raise ValueError("No daily factor panel rows were generated.")

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
            "MKT": mkt,
            "SMB": smb,
            "HML": hml,
            "RMW": rmw,
            "CMA": cma,
            "SMB_BM": smb_bm,
            "SMB_OP": smb_op,
            "SMB_INV": smb_inv,
            "n_stocks": group["stock_id"].nunique(),
        })

    ff5 = (
        factor_panel
        .groupby("date")
        .apply(calc_ff5_one_day)
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    ff5["RF"] = 0.0
    for col in ["MKT", "SMB", "HML", "RMW", "CMA", "SMB_BM", "SMB_OP", "SMB_INV"]:
        if col not in ff5.columns:
            ff5[col] = 0.0
        ff5[col] = pd.to_numeric(ff5[col], errors="coerce")
    # Keep MKT as data-driven; fill missing characteristic factors with zero so OLS remains usable.
    ff5["MKT"] = ff5["MKT"].fillna(0.0)
    for col in ["SMB", "HML", "RMW", "CMA", "SMB_BM", "SMB_OP", "SMB_INV"]:
        ff5[col] = ff5[col].fillna(0.0)
    ff5["MKT_RF"] = ff5["MKT"] - ff5["RF"]
    ff5 = ff5[["date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF", "MKT", "SMB_BM", "SMB_OP", "SMB_INV", "n_stocks"]]
    return factor_panel, ff5

def run_ols_alpha(reg_df: pd.DataFrame, config: FF5RawBuildConfig) -> dict[str, object] | None:
    clean = reg_df[["excess_ret"] + FACTOR_COLS].dropna().tail(config.lookback_days)
    if len(clean) < config.min_obs:
        return None
    y = clean["excess_ret"]
    x = sm.add_constant(clean[FACTOR_COLS])
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
    future = trading_dates[trading_dates > current_date]
    if len(future) == 0:
        return pd.NaT
    return pd.Timestamp(future[0])


def build_rolling_alpha_scores(
    ff5: pd.DataFrame,
    formations: pd.DataFrame,
    returns: pd.DataFrame,
    config: FF5RawBuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ff5 = ff5.copy()
    ff5["date"] = parse_date(ff5["date"])
    for col in FACTOR_COLS + ["RF"]:
        ff5[col] = pd.to_numeric(ff5[col], errors="coerce")
    ff5 = ff5[["date"] + FACTOR_COLS + ["RF"]].dropna(subset=["date"]).drop_duplicates("date").sort_values("date")

    stock_returns = returns[["date", "stock_id", "stock_name", "adj_close", "ret"]].copy()
    panel = pd.merge(stock_returns, ff5, on="date", how="inner")
    panel["excess_ret"] = panel["ret"] - panel["RF"]
    panel = panel.sort_values(["stock_id", "date"]).reset_index(drop=True)

    stock_panel = {
        stock_id: group.set_index("date").sort_index()
        for stock_id, group in panel.groupby("stock_id")
    }

    trading_dates = pd.DatetimeIndex(sorted(ff5["date"].dropna().unique()))
    formation_dates = sorted(formations["formation_date"].dropna().unique())
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
    for fdate in formation_dates:
        fdate = pd.Timestamp(fdate)
        members = formations[formations["formation_date"] == fdate].copy()
        for _, row in members.iterrows():
            stock_id = row["stock_id"]
            if stock_id not in stock_panel:
                continue
            history = stock_panel[stock_id]
            history = history[history.index <= fdate]
            valid = history[["excess_ret"] + FACTOR_COLS].dropna().tail(config.lookback_days)
            if len(valid) < config.min_obs:
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

    alpha_scores = pd.DataFrame(rows)
    if alpha_scores.empty:
        return alpha_scores, panel

    alpha_scores = pd.merge(alpha_scores, formation_meta_df, on="formation_date", how="left")
    alpha_scores["alpha_rank"] = (
        alpha_scores
        .groupby("formation_date")["alpha"]
        .rank(ascending=False, method="first")
    )
    alpha_scores = alpha_scores.sort_values(["formation_date", "alpha_rank", "stock_id"]).reset_index(drop=True)
    return alpha_scores, panel
