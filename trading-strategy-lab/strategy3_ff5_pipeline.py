from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


FACTOR_COLS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]


@dataclass(frozen=True)
class FF5RawBuildConfig:
    market_cap_top_n: int = 150
    lookback_days: int = 252
    min_obs: int = 180


def run_ff5_raw_pipeline(raw_dir: str | Path, config: FF5RawBuildConfig) -> dict[str, pd.DataFrame]:
    files = discover_strategy3_raw_files(raw_dir)

    data_raw = read_many_excel(files["data"])
    ad_raw = read_many_excel(files["ad_price"])
    asset_raw = read_many_excel(files["asset"])
    rmw_raw = read_many_excel(files["rmw"])

    data = normalize_market_data(data_raw)
    price_df = normalize_ad_price(ad_raw)
    returns = price_df[["date", "stock_id", "stock_name", "adj_close", "ret"]].copy()
    fin = build_financial_panel(asset_raw, rmw_raw)
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
    base = Path(raw_dir).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Strategy 3 raw data folder not found: {base}")

    candidates = [
        p for p in base.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".xlsx", ".xls"}
        and not p.name.startswith("~$")
    ]

    def has_token(path: Path, token: str) -> bool:
        return token in path.name.upper()

    ad_price = sorted(p for p in candidates if has_token(p, "AD PRICE"))
    asset = sorted(p for p in candidates if has_token(p, "ASSET"))
    rmw = sorted(p for p in candidates if has_token(p, "RMW"))
    data = sorted(
        p for p in candidates
        if has_token(p, "DATA")
        and not has_token(p, "AD PRICE")
        and not has_token(p, "ASSET")
        and not has_token(p, "RMW")
    )

    missing = []
    if not data:
        missing.append("DATA")
    if not ad_price:
        missing.append("AD PRICE")
    if not asset:
        missing.append("ASSET")
    if not rmw:
        missing.append("RMW")
    if missing:
        found = "\n".join(str(p.relative_to(base)) for p in candidates[:80])
        raise FileNotFoundError(
            f"Missing raw file group(s): {', '.join(missing)}. "
            f"Checked folder: {base}\nFound files:\n{found}"
        )

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


def read_many_excel(paths: list[Path]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        df = pd.read_excel(path)
        df.columns = [str(c).strip() for c in df.columns]
        df["source_file"] = path.name
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No Excel files were provided.")
    return pd.concat(dfs, ignore_index=True)


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


def normalize_market_data(data_raw: pd.DataFrame) -> pd.DataFrame:
    df = data_raw.rename(columns={
        "日期": "date",
        "股票代號": "stock_id",
        "股票名稱": "stock_name",
        "收盤價": "close",
        "成交量": "volume",
        "成交金額(千)": "trading_value_thousand",
        "股本(百萬)": "capital_million",
        "總市值(億)": "market_cap_100m",
        "股價淨值比": "pb",
        "週轉率(%)": "turnover_pct",
    }).copy()

    required = ["date", "stock_id", "market_cap_100m"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DATA files are missing required columns: {missing}")

    df["date"] = parse_date(df["date"])
    df["stock_id"] = clean_stock_id(df["stock_id"])
    if "stock_name" not in df.columns:
        df["stock_name"] = ""

    for col in [
        "close", "volume", "trading_value_thousand", "capital_million",
        "market_cap_100m", "pb", "turnover_pct",
    ]:
        if col in df.columns:
            df[col] = to_num(df[col])
        else:
            df[col] = np.nan

    columns = [
        "date", "stock_id", "stock_name", "close", "volume",
        "trading_value_thousand", "capital_million", "market_cap_100m",
        "pb", "turnover_pct",
    ]
    return (
        df.dropna(subset=["date", "stock_id", "market_cap_100m"])
        .drop_duplicates(["date", "stock_id"], keep="last")
        .loc[:, columns]
        .sort_values(["date", "stock_id"])
        .reset_index(drop=True)
    )


def normalize_ad_price(ad_raw: pd.DataFrame) -> pd.DataFrame:
    df = ad_raw.rename(columns={
        "日期": "date",
        "股票代號": "stock_id",
        "股票名稱": "stock_name",
        "收盤價": "adj_close",
        "成交量": "volume",
        "成交金額(千)": "trading_value_thousand",
        "總市值(億)": "market_cap_100m",
    }).copy()

    required = ["date", "stock_id", "adj_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"AD PRICE files are missing required columns: {missing}")

    if "stock_name" not in df.columns:
        df["stock_name"] = ""
    df["date"] = parse_date(df["date"])
    df["stock_id"] = clean_stock_id(df["stock_id"])
    df["adj_close"] = to_num(df["adj_close"])

    df = (
        df.dropna(subset=["date", "stock_id", "adj_close"])
        .drop_duplicates(["date", "stock_id"], keep="last")
        .sort_values(["stock_id", "date"])
        .reset_index(drop=True)
    )
    df["ret"] = df.groupby("stock_id")["adj_close"].pct_change()
    df.loc[~np.isfinite(df["ret"]), "ret"] = np.nan
    return df.sort_values(["date", "stock_id"]).reset_index(drop=True)


def build_financial_panel(asset_raw: pd.DataFrame, rmw_raw: pd.DataFrame) -> pd.DataFrame:
    asset = asset_raw.rename(columns={
        "年季": "year_quarter",
        "股票代號": "stock_id",
        "股票名稱": "stock_name",
        "資產總計(千)": "total_assets_thousand",
        "負債總計(千)": "total_liabilities_thousand",
        "權益總計(千)": "equity_total_thousand",
        "公告日期": "announce_date_asset",
    }).copy()
    asset["stock_id"] = clean_stock_id(asset["stock_id"])
    asset["fiscal_q"] = parse_cmoney_quarter(asset["year_quarter"])
    asset["announce_date_asset"] = parse_date(asset["announce_date_asset"])
    for col in ["total_assets_thousand", "total_liabilities_thousand", "equity_total_thousand"]:
        asset[col] = to_num(asset[col])
    asset = asset.dropna(subset=["stock_id", "fiscal_q"]).sort_values(["stock_id", "fiscal_q"])
    asset["asset_growth"] = asset.groupby("stock_id")["total_assets_thousand"].pct_change(4)
    asset = asset[[
        "stock_id", "stock_name", "fiscal_q", "announce_date_asset",
        "total_assets_thousand", "total_liabilities_thousand",
        "equity_total_thousand", "asset_growth",
    ]]

    rmw = rmw_raw.rename(columns={
        "年季": "year_quarter",
        "股票代號": "stock_id",
        "股票名稱": "stock_name",
        "營業收入淨額(千)": "revenue_thousand",
        "營業成本(千)": "cogs_thousand",
        "營業毛利(千)": "gross_profit_thousand",
        "營業費用(千)": "operating_expense_thousand",
        "營業利益(千)": "operating_income_thousand",
        "稅前純益(千)": "pretax_income_thousand",
        "稅後純益(千)": "net_income_thousand",
        "每股稅後盈餘(元)": "eps",
        "公告日期": "announce_date_rmw",
    }).copy()
    rmw["stock_id"] = clean_stock_id(rmw["stock_id"])
    rmw["fiscal_q"] = parse_cmoney_quarter(rmw["year_quarter"])
    rmw["announce_date_rmw"] = parse_date(rmw["announce_date_rmw"])
    for col in [
        "revenue_thousand", "cogs_thousand", "gross_profit_thousand",
        "operating_expense_thousand", "operating_income_thousand",
        "pretax_income_thousand", "net_income_thousand", "eps",
    ]:
        if col in rmw.columns:
            rmw[col] = to_num(rmw[col])
        else:
            rmw[col] = np.nan
    rmw = rmw.dropna(subset=["stock_id", "fiscal_q"])
    rmw = rmw[[
        "stock_id", "stock_name", "fiscal_q", "announce_date_rmw",
        "revenue_thousand", "operating_income_thousand",
        "pretax_income_thousand", "net_income_thousand", "eps",
    ]]

    fin = pd.merge(asset, rmw, on=["stock_id", "fiscal_q"], how="outer", suffixes=("_asset", "_rmw"))
    fin["stock_name"] = fin["stock_name_asset"].combine_first(fin["stock_name_rmw"])
    fin["announce_date"] = fin[["announce_date_asset", "announce_date_rmw"]].max(axis=1)
    fin["profitability"] = fin["operating_income_thousand"] / fin["equity_total_thousand"]
    fin = fin[[
        "stock_id", "stock_name", "fiscal_q", "announce_date",
        "equity_total_thousand", "total_assets_thousand",
        "operating_income_thousand", "profitability", "asset_growth",
    ]]
    return fin.dropna(subset=["stock_id", "announce_date"]).sort_values(["stock_id", "announce_date"])


def merge_latest_fin_by_stock(formations: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
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
        snap = snap.sort_values("market_cap_100m", ascending=False).head(top_n)
        snap["rank"] = np.arange(1, len(snap) + 1)
        snap["formation_date"] = formation_date
        rows.append(snap)

    if not rows:
        raise ValueError("No monthly market-cap formation rows were generated.")

    formations = pd.concat(rows, ignore_index=True)
    formations = formations[[
        "formation_date", "rank", "stock_id", "stock_name",
        "market_cap_100m", "pb", "trading_value_thousand",
    ]]
    formations = merge_latest_fin_by_stock(formations, fin)
    formations["bm_from_fin"] = (
        formations["equity_total_thousand"] * 1000
    ) / (formations["market_cap_100m"] * 100_000_000)
    formations["bm_from_pb"] = 1 / formations["pb"]
    formations.loc[formations["pb"] <= 0, "bm_from_pb"] = np.nan
    formations["bm"] = formations["bm_from_fin"].where(formations["bm_from_fin"].notna(), formations["bm_from_pb"])

    def assign_groups(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        median_cap = group["market_cap_100m"].median()
        group["size_grp"] = np.where(group["market_cap_100m"] <= median_cap, "S", "B")
        group["bm_grp"] = assign_tertile(group["bm"], "L", "M", "H")
        group["op_grp"] = assign_tertile(group["profitability"], "W", "N", "R")
        group["inv_grp"] = assign_tertile(group["asset_growth"], "C", "N", "A")
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
    fdates = list(sorted(formations["formation_date"].dropna().unique()))
    panels = []
    for i, fdate in enumerate(fdates):
        next_fdate = fdates[i + 1] if i + 1 < len(fdates) else returns["date"].max()
        ret_slice = returns.loc[
            (returns["date"] > fdate) & (returns["date"] <= next_fdate),
            ["date", "stock_id", "ret"],
        ].copy()
        members = formations[formations["formation_date"] == fdate].copy()
        panel = pd.merge(
            ret_slice,
            members[[
                "formation_date", "stock_id", "stock_name", "rank",
                "market_cap_100m", "weight", "size_grp", "bm_grp", "op_grp", "inv_grp",
            ]],
            on="stock_id",
            how="inner",
        )
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
