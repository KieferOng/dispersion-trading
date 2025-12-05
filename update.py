import os
import time
import math
import datetime as dt

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from requests.exceptions import ReadTimeout, RequestException

# -------------------------------------------------
# Config & API
# -------------------------------------------------

load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    raise ValueError("POLYGON_API_KEY not found in environment/.env")

BASE = "https://api.massive.com"  # your proxy for Polygon-style APIs
NY_TZ = "America/New_York"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

DATA2 = "data2"
os.makedirs(DATA2, exist_ok=True)


# -------------------------------------------------
# Black–Scholes + IV helpers (your originals)
# -------------------------------------------------

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    return 1.0 / math.sqrt(2 * math.pi) * math.exp(-0.5 * x * x)


def bs_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * math.exp(-r * T) - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return S * norm_pdf(d1) * math.sqrt(T)


def implied_vol_call(S, K, T, r, price, max_iter=60, tol=1e-6):
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    if price is None or price <= intrinsic + 1e-4 or S <= 0 or K <= 0 or T <= 0:
        return None
    low, high = 1e-4, 3.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        diff = bs_call_price(S, K, T, r, mid) - price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def implied_vol_put(S, K, T, r, price, max_iter=60, tol=1e-6):
    intrinsic = max(K * math.exp(-r * T) - S, 0.0)
    if price is None or price <= intrinsic + 1e-4 or S <= 0 or K <= 0 or T <= 0:
        return None
    low, high = 1e-4, 3.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        diff = bs_put_price(S, K, T, r, mid) - price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


# -------------------------------------------------
# API helpers (your originals)
# -------------------------------------------------

def fetch_underlying_daily(sym, start_date, end_date):
    url = f"{BASE}/v2/aggs/ticker/{sym}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"{sym} daily {start_date}→{end_date}: {r.text[:200]}")
    j = r.json()
    rows = j.get("results", [])
    if not rows:
        return pd.DataFrame(columns=["date_ny", "S_open", "S_close"])
    df = pd.DataFrame(rows).rename(columns={"t": "ts", "o": "open", "c": "close"})
    df["ts_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["ts_ny"] = df["ts_utc"].dt.tz_convert(NY_TZ)
    df["date_ny"] = df["ts_ny"].dt.date
    df = df.sort_values("date_ny")
    return df[["date_ny", "open", "close"]].rename(
        columns={"open": "S_open", "close": "S_close"}
    )


def list_contracts_asof(sym, asof_date):
    url = f"{BASE}/v3/reference/options/contracts"
    params = {
        "underlying_ticker": sym,
        "include_expired": "true",
        "as_of": pd.Timestamp(asof_date).strftime("%Y-%m-%d"),
        "limit": 1000,
        "order": "asc",
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"{sym} contracts {asof_date}: {r.text[:200]}")
    return pd.DataFrame(r.json().get("results", []))


def pick_nearest_expiry_atm(
    df_contracts, spot, asof_date, min_days_ahead=3, max_days_ahead=30
):
    if df_contracts is None or df_contracts.empty:
        return None, None, None
    keep = df_contracts[
        ["ticker", "contract_type", "strike_price", "expiration_date"]
    ].dropna()
    if keep.empty:
        return None, None, None
    keep["strike_price"] = keep["strike_price"].astype(float)
    keep["expiration_date"] = pd.to_datetime(keep["expiration_date"]).dt.date

    as_of = pd.Timestamp(asof_date).date()
    keep["days_to_exp"] = (keep["expiration_date"] - as_of).apply(lambda x: x.days)

    fwd = keep[
        (keep["days_to_exp"] >= min_days_ahead)
        & (keep["days_to_exp"] <= max_days_ahead)
    ].copy()
    if fwd.empty:
        return None, None, None

    expiry = fwd.sort_values(["days_to_exp", "expiration_date"]).iloc[0][
        "expiration_date"
    ]
    near = fwd[fwd["expiration_date"] == expiry]

    calls = near[near["contract_type"] == "call"].copy()
    puts = near[near["contract_type"] == "put"].copy()
    if calls.empty or puts.empty:
        return None, None, None

    calls["dist"] = (calls["strike_price"] - spot).abs()
    puts["dist"] = (puts["strike_price"] - spot).abs()

    call_info = calls.sort_values("dist").iloc[0].to_dict()
    put_info = puts.sort_values("dist").iloc[0].to_dict()
    return expiry, call_info, put_info


def fetch_option_bar(opt_ticker, day):
    start_ymd = pd.Timestamp(day).strftime("%Y-%m-%d")
    end_ymd = pd.Timestamp(day + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    # 1) Try daily bar
    url_day = f"{BASE}/v2/aggs/ticker/{opt_ticker}/range/1/day/{start_ymd}/{end_ymd}"
    try:
        r = requests.get(url_day, headers=HEADERS, timeout=40)  # a bit more generous
        if r.status_code == 200:
            rows = r.json().get("results", [])
            if rows:
                x = rows[0]
                return {"volume": x.get("v", 0), "close": x.get("c", None)}
        else:
            print(f"[{opt_ticker} {day}] daily request status {r.status_code}: {r.text[:120]}")
    except ReadTimeout:
        print(f"[{opt_ticker} {day}] ⚠️ daily request timed out, falling back to minute")
    except RequestException as e:
        print(f"[{opt_ticker} {day}] ⚠️ daily request error: {e}")

    # 2) Fallback: minute bars
    url_min = f"{BASE}/v2/aggs/ticker/{opt_ticker}/range/1/minute/{start_ymd}/{end_ymd}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}
    try:
        r = requests.get(url_min, params=params, headers=HEADERS, timeout=60)
        if r.status_code != 200:
            print(f"[{opt_ticker} {day}] ⚠️ minute request status {r.status_code}: {r.text[:120]}")
            return {"volume": 0, "close": None}

        rows = r.json().get("results", [])
        if not rows:
            return {"volume": 0, "close": None}

        vol = sum(row.get("v", 0) for row in rows)
        last_close = rows[-1].get("c", None)
        return {"volume": vol, "close": last_close}
    except ReadTimeout:
        print(f"[{opt_ticker} {day}] ⚠️ minute request timed out, returning empty bar")
        return {"volume": 0, "close": None}
    except RequestException as e:
        print(f"[{opt_ticker} {day}] ⚠️ minute request error: {e}, returning empty bar")
        return {"volume": 0, "close": None}


# -------------------------------------------------
# Core daily builder (your original build_daily_options)
# -------------------------------------------------

def build_daily_options(sym, start_date, end_date):
    print(f"\n[{sym}] DAILY build {start_date} → {end_date}")
    px = fetch_underlying_daily(sym, start_date, end_date)
    if px.empty:
        print(f"[{sym}] no underlying daily data in that range")
        return pd.DataFrame()

    rows = []

    for _, row in px.iterrows():
        day = row["date_ny"]
        S_open = float(row["S_open"])
        S_close = float(row["S_close"])

        try:
            dfc = list_contracts_asof(sym, day)
        except Exception as e:
            print(f"[{sym}] contracts fail {day}: {e}")
            continue

        expiry, call_info, put_info = pick_nearest_expiry_atm(
            dfc, S_close, day, min_days_ahead=1, max_days_ahead=45
        )
        if expiry is None:
            print(f"[{sym}] no suitable ATM on {day}")
            continue

        call_bar = fetch_option_bar(call_info["ticker"], day)
        put_bar = fetch_option_bar(put_info["ticker"], day)

        call_vol = call_bar["volume"]
        call_price = call_bar["close"]
        put_vol = put_bar["volume"]
        put_price = put_bar["close"]
        total_vol = (call_vol or 0) + (put_vol or 0)

        asof = pd.Timestamp(day).date()
        dte = max((expiry - asof).days, 1)
        T = dte / 365.0
        r = 0.0

        call_iv = call_vega = None
        if call_price is not None and call_price > 0.05:
            Kc = float(call_info["strike_price"])
            call_iv = implied_vol_call(S_close, Kc, T, r, call_price)
            if call_iv is None:
                call_iv = 0.20
            v = bs_vega(S_close, Kc, T, r, call_iv)
            call_vega = v / 100 if v is not None else None

        put_iv = put_vega = None
        if put_price is not None and put_price > 0.05:
            Kp = float(put_info["strike_price"])
            put_iv = implied_vol_put(S_close, Kp, T, r, put_price)
            if put_iv is None:
                put_iv = 0.20
            v = bs_vega(S_close, Kp, T, r, put_iv)
            put_vega = v / 100 if v is not None else None

        ivs = [x for x in [call_iv, put_iv] if x is not None]
        avg_iv = sum(ivs) / len(ivs) if ivs else None

        vegas = [x for x in [call_vega, put_vega] if x is not None]
        avg_vega = sum(vegas) / len(vegas) if vegas else None

        rows.append(
            {
                "date_ny": day,
                "S_open": S_open,
                "S_close": S_close,
                "call_ticker": call_info["ticker"],
                "put_ticker": put_info["ticker"],
                "strike_call": float(call_info["strike_price"]),
                "strike_put": float(put_info["strike_price"]),
                "expiry": pd.Timestamp(expiry),
                "days_to_exp": dte,
                "call_vol": call_vol,
                "put_vol": put_vol,
                "total_vol": total_vol,
                "call_price": call_price,
                "put_price": put_price,
                "call_iv": call_iv,
                "put_iv": put_iv,
                "avg_iv": avg_iv,
                "call_vega": call_vega,
                "put_vega": put_vega,
                "avg_vega": avg_vega,
            }
        )

        print(
            f"[{sym}] {day} | Cvol={call_vol} Pvol={put_vol} | "
            f"Cpx={call_price} Ppx={put_price}"
        )
        time.sleep(0.02)

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows).sort_values("date_ny")
    return df_out


# -------------------------------------------------
# Step 1: append to each *_options_daily_merged.csv
# -------------------------------------------------

def update_merged_ticker(ticker: str) -> None:
    path = os.path.join(DATA2, f"{ticker}_options_daily_merged.csv")
    if not os.path.exists(path):
        print(f"[{ticker}] merged file not found at {path}, skipping")
        return

    df_old = pd.read_csv(path)
    if "date_ny" not in df_old.columns:
        print(f"[{ticker}] merged file missing date_ny, skipping")
        return

    df_old["date_ny"] = pd.to_datetime(df_old["date_ny"], errors="coerce").dt.date
    df_old = df_old[df_old["date_ny"].notna()]
    if df_old.empty:
        print(f"[{ticker}] merged file empty, skipping")
        return

    last_date = max(df_old["date_ny"])
    today = dt.date.today()
    end_date = today - dt.timedelta(days=1)

    if last_date >= end_date:
        print(f"[{ticker}] up to date (last={last_date}, end={end_date})")
        return

    start_date = last_date + dt.timedelta(days=1)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    df_new = build_daily_options(ticker, start_str, end_str)
    if df_new.empty:
        print(f"[{ticker}] no new rows returned, nothing to append")
        return

    df_new["date_ny"] = pd.to_datetime(df_new["date_ny"], errors="coerce").dt.date
    combined = (
        pd.concat([df_old, df_new], ignore_index=True)
        .dropna(subset=["date_ny"])
        .sort_values("date_ny")
        .drop_duplicates(subset=["date_ny"], keep="last")
        .reset_index(drop=True)
    )

    combined.to_csv(path, index=False)
    print(f"[{ticker}] ✅ updated merged file → {len(combined)} rows")


# -------------------------------------------------
# Step 2: clean merged files with cutoffs (your second block)
# -------------------------------------------------

CUTOFFS = {
    "SPY": "2020-05-31",
    "AAPL": "2020-09-01",
    "MSFT": "2020-05-31",
    "NVDA": "2024-06-12",
    "AMZN": "2022-06-07",
    "META": "2022-06-11",
    "AVGO": "2024-07-16",
    "TSLA": "2022-08-26",
    "WMT": "2024-02-27",
    "JPM": "2020-05-31",
    "GOOG": "2022-07-19",
    "V": "2020-05-31",
    "LLY": "2020-05-31",
}

DEFAULT_CUTOFF = "2020-05-31"
KEY_COLS = [
    "S_close",
    "total_vol",
    "avg_iv",
    "avg_vega",
    "call_vol",
    "put_vol",
    "call_iv",
    "put_iv",
]


def clean_merged_files():
    files = [
        f for f in os.listdir(DATA2) if f.endswith("_options_daily_merged.csv")
    ]
    print(f"Cleaning {len(files)} merged files...")

    for fname in sorted(files):
        path = os.path.join(DATA2, fname)
        ticker = fname.split("_options_daily_merged.csv")[0]

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[{ticker}] ❌ failed to read merged file: {e}")
            continue

        if "date_ny" not in df.columns:
            print(f"[{ticker}] ⚠️ no date_ny in merged, skipping")
            continue

        df["date_ny"] = pd.to_datetime(df["date_ny"], errors="coerce").dt.date
        df = df[df["date_ny"].notna()]

        cutoff_str = CUTOFFS.get(ticker, DEFAULT_CUTOFF)
        cutoff = pd.to_datetime(cutoff_str).date()
        df = df[df["date_ny"] >= cutoff]

        existing_keys = [c for c in KEY_COLS if c in df.columns]
        if existing_keys:
            df = df.dropna(subset=existing_keys, how="all")

        df = (
            df.sort_values("date_ny")
            .drop_duplicates(subset=["date_ny"], keep="last")
            .reset_index(drop=True)
        )

        if df.empty:
            print(f"[{ticker}] ⚠️ all rows removed; leaving file unchanged")
            continue

        df.to_csv(path, index=False)
        print(f"[{ticker}] ✅ cleaned → {len(df)} rows ≥ {cutoff}")


# -------------------------------------------------
# Step 3: rebuild master_dispersion_data.csv
#         (your third + fourth blocks combined)
# -------------------------------------------------

MASTER_FILE = os.path.join(DATA2, "master_dispersion_data.csv")


def rebuild_master():
    csv_files = [
        f for f in os.listdir(DATA2) if f.endswith("_options_daily_merged.csv")
    ]
    print(f"Building master from {len(csv_files)} tickers...")

    ticker_dfs = {}

    for file in csv_files:
        ticker = file.split("_options_daily_merged")[0]
        path = os.path.join(DATA2, file)

        df = pd.read_csv(path)
        if "date_ny" not in df.columns:
            print(f"[{ticker}] ⚠️ missing date_ny, skipping")
            continue

        df["date_ny"] = pd.to_datetime(df["date_ny"], errors="coerce")
        df = df[df["date_ny"].notna()]

        required = ["total_vol", "avg_vega", "avg_iv"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[{ticker}] ⚠️ missing {missing}, skipping")
            continue

        df["Vs"] = df["total_vol"] * df["avg_vega"]
        df = (
            df.sort_values("date_ny")
            .drop_duplicates("date_ny", keep="last")
            .reset_index(drop=True)
        )

        ticker_dfs[ticker] = df

    if not ticker_dfs:
        raise RuntimeError("No valid ticker data found for master build.")

    start_dates = {t: ticker_dfs[t]["date_ny"].min() for t in ticker_dfs}
    end_dates = {t: ticker_dfs[t]["date_ny"].max() for t in ticker_dfs}
    min_date = min(start_dates.values())
    max_date = max(end_dates.values())

    full_range = pd.date_range(start=min_date, end=max_date, freq="D")
    master = pd.DataFrame({"date_ny": full_range})

    ordered_tickers = ["SPY"] + sorted(
        [t for t in ticker_dfs.keys() if t != "SPY"],
        key=lambda x: start_dates[x],
    )

    print("Merge order:", ordered_tickers)

    for ticker in ordered_tickers:
        df = ticker_dfs[ticker][
            ["date_ny", "total_vol", "avg_vega", "Vs", "avg_iv"]
        ].copy()
        df = df.rename(
            columns={
                "total_vol": f"{ticker}_vol",
                "avg_vega": f"{ticker}_avgVega",
                "Vs": f"{ticker}_Vs",
                "avg_iv": f"{ticker}_IV",
            }
        )
        master = master.merge(df, on="date_ny", how="left")

    ticker_cols = [col for col in master.columns if col != "date_ny"]
    master = master.dropna(subset=ticker_cols, how="all")

    # ---- add Singles_* aggregates (your fourth block) ----
    vs_cols = [c for c in master.columns if c.endswith("_Vs")]
    iv_cols = [c for c in master.columns if c.endswith("_IV")]

    single_vs_cols = [c for c in vs_cols if not c.startswith("SPY_")]
    single_iv_cols = [c for c in iv_cols if not c.startswith("SPY_")]

    master["Singles_count"] = master[single_vs_cols].notna().sum(axis=1)
    master["Singles_Vs_sum"] = master[single_vs_cols].sum(axis=1, skipna=True)
    master["Singles_Vs_avg"] = master[single_vs_cols].mean(axis=1, skipna=True)
    master["Singles_IV_avg"] = master[single_iv_cols].mean(axis=1, skipna=True)

    front_cols = [
        "date_ny",
        "Singles_count",
        "Singles_Vs_sum",
        "Singles_Vs_avg",
        "Singles_IV_avg",
    ]
    other_cols = [c for c in master.columns if c not in front_cols]
    master = master[front_cols + other_cols]

    master["date_ny"] = master["date_ny"].dt.strftime("%d/%m/%Y")

    master.to_csv(MASTER_FILE, index=False)
    print(f"✅ Master dataset saved → {MASTER_FILE}")
    print(f"Rows: {len(master)}, Columns: {len(master.columns)}")


# -------------------------------------------------
# Step 4: rebuild dispersion_factors.csv
#         (your last two blocks)
# -------------------------------------------------

FACTORS_FILE = os.path.join(DATA2, "dispersion_factors.csv")


def rebuild_factors():
    df = pd.read_csv(MASTER_FILE)
    df["date_ny"] = pd.to_datetime(
        df["date_ny"], format="mixed", dayfirst=True, errors="coerce"
    )

    if df["date_ny"].isna().sum() > 0:
        print("⚠️ Some date_ny entries could not be parsed in master.")

    df = df.dropna(subset=["SPY_IV", "SPY_Vs"]).reset_index(drop=True)

    stock_iv_cols = [
        c for c in df.columns if c.endswith("_IV") and not c.startswith("SPY_")
    ]

    df["F1"] = df["Singles_Vs_sum"] / (df["Singles_Vs_sum"] + df["SPY_Vs"])
    df["F2"] = df["Singles_IV_avg"] - df["SPY_IV"]

    F3 = []
    for _, row in df.iterrows():
        sigma_index = row["SPY_IV"]
        sigmas = row[stock_iv_cols].dropna().values
        N = len(sigmas)
        if N < 2:
            F3.append(np.nan)
            continue

        sum_sigma = np.sum(sigmas)
        sum_sigma_sq = np.sum(sigmas ** 2)
        numerator = (N ** 2) * (sigma_index ** 2) - sum_sigma_sq
        denominator = (sum_sigma ** 2) - sum_sigma_sq
        if denominator <= 0:
            F3.append(np.nan)
            continue
        rho_impl = numerator / denominator
        if rho_impl < -0.5 or rho_impl > 1.5:
            F3.append(np.nan)
            continue
        F3.append(1 - rho_impl)

    df["F3"] = F3

    df_out = df[["date_ny", "F1", "F2", "F3"]].copy()
    df_out.to_csv(FACTORS_FILE, index=False)

    print(f"✅ Dispersion factors saved → {FACTORS_FILE}")
    print(f"Total rows after dropping SPY-missing: {len(df_out)}")

    # now add Z-scores + EWMA/SMA as in your last block
    df_z = df_out.copy()
    df_z["date_ny"] = pd.to_datetime(df_z["date_ny"], dayfirst=True, errors="coerce")

    for col in ["F1", "F2", "F3"]:
        df_z[col] = pd.to_numeric(df_z[col], errors="coerce")

    window = 60
    min_pts = 40

    def rolling_z(series, win=60, min_pts=40):
        m = series.rolling(win, min_periods=min_pts).mean()
        s = series.rolling(win, min_periods=min_pts).std()
        return (series - m) / s

    df_z["F1_z60"] = rolling_z(df_z["F1"], window, min_pts)
    df_z["F2_z60"] = rolling_z(df_z["F2"], window, min_pts)
    df_z["F3_z60"] = rolling_z(df_z["F3"], window, min_pts)

    w1, w2, w3 = 0.2, 0.4, 0.4
    df_z["Dispersion_Z_60d"] = (
        w1 * df_z["F1_z60"] + w2 * df_z["F2_z60"] + w3 * df_z["F3_z60"]
    )

    for span in [20, 30, 60]:
        df_z[f"Dispersion_Z_EWMA_{span}"] = df_z["Dispersion_Z_60d"].ewm(
            span=span, adjust=False, min_periods=span
        ).mean()

    for win in [60, 90, 120]:
        df_z[f"Dispersion_Z_SMA_{win}"] = df_z["Dispersion_Z_60d"].rolling(
            window=win, min_periods=win
        ).mean()

    df_z.to_csv(FACTORS_FILE, index=False)
    print(
        "✅ Saved Dispersion_Z_60d + EWMA(20,30,60) + SMA(60,90,120) "
        f"→ {FACTORS_FILE}"
    )


# -------------------------------------------------
# main
# -------------------------------------------------

def main():
    merged_files = [
        f for f in os.listdir(DATA2) if f.endswith("_options_daily_merged.csv")
    ]
    tickers = sorted(f.split("_options_daily_merged.csv")[0] for f in merged_files)
    print("Tickers detected:", tickers)

    for t in tickers:
        update_merged_ticker(t)

    clean_merged_files()
    rebuild_master()
    rebuild_factors()
    print("✅ update.py finished successfully")


if __name__ == "__main__":
    main()
