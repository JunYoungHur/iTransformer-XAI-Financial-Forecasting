"""

from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

Standalone label distribution sensitivity check.
=================================================

Runs completely independently of gateformer.py. Downloads S&P 500 prices,
computes the minimum features needed for labeling (daily returns + rolling
volatility), then applies the same dynamic-threshold labeling across several
(horizon, multiplier) combinations and reports the resulting class
distributions on the training split.

Does NOT produce the full ~40-feature set — that would take much longer and
is not needed for label statistics. Only `Close` prices matter for labels.

Safe to run concurrently with the main training pipeline: uses separate
cache file names and does not touch any gateformer.py artifacts.
"""

import os
import requests
from io import StringIO

import numpy as np
import pandas as pd
import yfinance as yf


# ========================================================================
# Config (intentionally duplicated from gateformer.py for independence)
# ========================================================================
TRAIN_END   = '2024-10-31'
PRICE_CACHE = None  # resolved to DATA_DIR at call time
MIN_DAYS    = 1000

COMBOS = [
    # (horizon, multiplier)  — main axis: horizon, secondary axis: multiplier
    (5,  0.5),
    (10, 0.5),
    (21, 0.5),
    (21, 0.3),   # looser threshold → fewer flat
    (21, 0.7),   # stricter threshold → more flat
]


# ========================================================================
# 1. Download prices (minimal — Close only)
# ========================================================================
def download_sp500_close(save_path: str = None) -> pd.DataFrame:
    if save_path is None:
        save_path = str(DATA_DIR / "sp500_prices_for_labelcheck.csv")
    if os.path.exists(save_path):
        print(f"📁 Using cached prices: {save_path}")
        return pd.read_csv(save_path, index_col=0, parse_dates=True)

    print("📥 Downloading S&P 500 Close prices (2000-01-01 → today)...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    tickers  = [t.replace('.', '-')
                for t in pd.read_html(StringIO(response.text))[0]['Symbol'].tolist()]

    data = yf.download(tickers, start='2000-01-01',
                       group_by='ticker', auto_adjust=True, progress=True)

    rows = []
    for tk in tickers:
        if tk in data.columns.levels[0]:
            s = data[tk]['Close'].dropna()
            if len(s) >= MIN_DAYS:
                rows.append(pd.DataFrame({'Ticker': tk, 'Close': s}))

    df = pd.concat(rows)
    df.index.name = 'Date'
    df.to_csv(save_path)
    print(f"✅ Saved {len(df):,} rows across {df['Ticker'].nunique()} tickers")
    return df


# ========================================================================
# 2. Dynamic-threshold labeling (same formula as gateformer.py)
# ========================================================================
def create_labels(df: pd.DataFrame,
                  horizon: int,
                  multiplier: float,
                  lookback: int = 20) -> pd.DataFrame:
    df = df.copy()
    df['Target_Return'] = df.groupby('Ticker')['Close'].shift(-horizon)
    df['Log_Return']    = np.log((df['Target_Return'] + 1e-8) / (df['Close'] + 1e-8))

    df['Daily_Log_Ret'] = df.groupby('Ticker')['Close'].transform(
        lambda x: np.log(x / (x.shift(1) + 1e-8))
    )
    df['Vol_Rolling']   = df.groupby('Ticker')['Daily_Log_Ret'].transform(
        lambda x: x.rolling(window=lookback).std()
    )
    df['Dynamic_Threshold'] = (
        df['Vol_Rolling'] * np.sqrt(horizon) * multiplier
    ).clip(lower=0.01)

    df['Target'] = 1
    df.loc[df['Log_Return'] >  df['Dynamic_Threshold'], 'Target'] = 2
    df.loc[df['Log_Return'] < -df['Dynamic_Threshold'], 'Target'] = 0

    # Drop the last `horizon` rows per ticker (no future return available)
    df = df.dropna(subset=['Log_Return'])
    return df


# ========================================================================
# 3. Main
# ========================================================================
def main():
    prices = download_sp500_close()
    prices.index = pd.to_datetime(prices.index, utc=True).tz_localize(None)

    results = []
    for horizon, multiplier in COMBOS:
        labeled = create_labels(prices, horizon=horizon, multiplier=multiplier)

        # TRAIN split only — that's what the model actually learns from
        train = labeled[labeled.index <= pd.to_datetime(TRAIN_END)]
        counts = train['Target'].value_counts().sort_index()
        total  = counts.sum()
        pct    = (counts / total * 100).round(2)

        # Also capture the median threshold for a sanity feel
        median_thr = train['Dynamic_Threshold'].median()

        results.append({
            'horizon':     horizon,
            'multiplier':  multiplier,
            'train_rows':  int(total),
            'down_%':      float(pct.get(0, 0.0)),
            'flat_%':      float(pct.get(1, 0.0)),
            'up_%':        float(pct.get(2, 0.0)),
            'median_thr':  round(float(median_thr), 4),
        })

    result_df = pd.DataFrame(results)
    print("\n📊 Label distribution across (horizon, multiplier) combinations")
    print("   Computed on TRAIN split only; 'median_thr' is median Dynamic_Threshold\n")
    print(result_df.to_string(index=False))

    result_df.to_csv(RESULTS_DIR / "label_distribution_sensitivity.csv", index=False)
    print("\n💾 Saved → label_distribution_sensitivity.csv")

    # Quick textual commentary
    print("\n🧭 Reading guide:")
    print("   • Fixing multiplier=0.5 and varying horizon shows how much the")
    print("     sqrt(horizon) normalization stabilizes class ratios across")
    print("     different prediction windows.")
    print("   • Fixing horizon=21 and varying multiplier shows how sensitive")
    print("     flat-vs-directional balance is to the threshold multiplier.")


if __name__ == "__main__":
    main()
