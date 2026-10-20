"""Supplemental indicators for ChanStructure.

Sea Bottom Moon is a transparent four-tier stochastic/RSI approximation based
on the local thinkScript source.  CD+RSI is a Python port of the local confirmed
thinkScript Buy/Sell logic.  Neither measures actual institutional ownership.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


def _wilder_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    change = close.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(loss != 0, 100).where(gain != 0, 0).fillna(50)


def _cooldown(raw: pd.Series, bars: int) -> pd.Series:
    result = np.zeros(len(raw), dtype=bool)
    last = -bars
    for i, value in enumerate(raw.fillna(False).to_numpy(dtype=bool)):
        if value and i - last >= bars:
            result[i] = True
            last = i
    return pd.Series(result, index=raw.index)


SEA_PRESETS = {
    # The shorter the bar, the quicker the oscillator and the less extreme a
    # pullback may be before it is worth flagging.  This prevents intraday
    # charts from waiting for a daily-chart-sized capitulation.
    "15m": {"rsi": 10, "stoch": 14, "tide": 2, "signal": 4, "small": 38, "medium": 30, "large": 23, "super": 16, "medium_low": 20, "large_low": 34, "super_low": 55, "volume": 1.10, "cooldown": 5},
    "30m": {"rsi": 11, "stoch": 16, "tide": 3, "signal": 4, "small": 35, "medium": 28, "large": 21, "super": 15, "medium_low": 24, "large_low": 40, "super_low": 65, "volume": 1.15, "cooldown": 6},
    "1h":  {"rsi": 12, "stoch": 18, "tide": 3, "signal": 5, "small": 32, "medium": 25, "large": 18, "super": 13, "medium_low": 28, "large_low": 46, "super_low": 80, "volume": 1.20, "cooldown": 7},
    "1d":  {"rsi": 14, "stoch": 21, "tide": 3, "signal": 5, "small": 29, "medium": 22, "large": 16, "super": 11, "medium_low": 30, "large_low": 50, "super_low": 110, "volume": 1.25, "cooldown": 8},
    "1wk": {"rsi": 14, "stoch": 14, "tide": 3, "signal": 5, "small": 33, "medium": 26, "large": 19, "super": 13, "medium_low": 10, "large_low": 18, "super_low": 34, "volume": 1.15, "cooldown": 4},
}


def sea_bottom_moon(data: pd.DataFrame, interval: str = "1d", min_bars_between: int | None = None) -> pd.DataFrame:
    """Adaptive four-level RSI/stochastic bottom oscillator.

    It is deliberately more responsive on intraday charts and more selective on
    daily/weekly charts.  A signal marks a potential exhausted pullback, not a
    prediction that the final price low is already in place.
    """
    p = SEA_PRESETS.get(interval, SEA_PRESETS["1d"])
    cooldown = p["cooldown"] if min_bars_between is None else min_bars_between
    close, high, low, volume = data.Close, data.High, data.Low, data.Volume.fillna(0)
    rsi = _wilder_rsi(close, p["rsi"])
    lo, hi = low.rolling(p["stoch"], min_periods=1).min(), high.rolling(p["stoch"], min_periods=1).max()
    stoch = (100 * (close - lo) / (hi - lo).replace(0, np.nan)).fillna(50)
    tide = ((rsi + stoch) / 2).ewm(span=p["tide"], adjust=False).mean()
    signal = tide.ewm(span=p["signal"], adjust=False).mean()
    cross_up = (tide > signal) & (tide.shift(1) <= signal.shift(1))
    turn_up = (tide > tide.shift(1)) & (tide.shift(1) <= tide.shift(1).rolling(3, min_periods=1).min())
    turning = cross_up | turn_up
    recent_low = lambda n: low <= low.shift(1).rolling(n, min_periods=max(2, n // 4)).min()
    volume_spike = volume > volume.rolling(20, min_periods=5).mean() * p["volume"]

    tide_low = tide.rolling(5, min_periods=1).min()
    small_raw = turning & (tide_low < p["small"]) & (tide < p["small"] + 12)
    medium_raw = turning & (tide_low < p["medium"]) & recent_low(p["medium_low"])
    large_raw = turning & ((tide_low < p["large"]) | ((rsi < p["large"] + 5) & (stoch < p["large"] + 5))) & recent_low(p["large_low"])
    super_raw = turning & (tide_low < p["super"]) & (rsi < p["super"] + 5) & (stoch < p["super"] + 5) & recent_low(p["super_low"]) & volume_spike

    # One event per bar, highest qualifying tier wins; every tier has its own cooldown.
    super_event = _cooldown(super_raw, cooldown)
    large_event = _cooldown(large_raw & ~super_raw, cooldown)
    medium_event = _cooldown(medium_raw & ~large_raw & ~super_raw, cooldown)
    small_event = _cooldown(small_raw & ~medium_raw & ~large_raw & ~super_raw, cooldown)
    tier = pd.Series(pd.NA, index=data.index, dtype="object")
    tier.loc[small_event] = "小底"
    tier.loc[medium_event] = "中底"
    tier.loc[large_event] = "大底"
    tier.loc[super_event] = "超大底"
    return pd.DataFrame({
        "tide": tide, "signal": signal, "rsi": rsi, "stoch": stoch,
        "pressure": (20 - tide).clip(lower=0), "tier": tier,
        "small": small_event, "medium": medium_event, "large": large_event, "super": super_event,
    }, index=data.index)


def cd_rsi_signals(data: pd.DataFrame, require_rsi_divergence: bool = True, rsi_confirm_bars: int = 5) -> pd.DataFrame:
    """Port of LZK_CD_RSI_Confirmed's confirmed BUY/SELL state logic."""
    close, high, low = data.Close.reset_index(drop=True), data.High.reset_index(drop=True), data.Low.reset_index(drop=True)
    diff = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    dea = diff.ewm(span=9, adjust=False).mean()
    hist = 2 * (diff - dea)
    n = len(data)
    down = (hist.shift(1) >= 0) & (hist < 0)
    up = (hist.shift(1) <= 0) & (hist > 0)
    cc1 = np.empty(n); difl1 = np.empty(n); cc2 = np.empty(n); cc3 = np.empty(n); difl2 = np.empty(n); difl3 = np.empty(n)
    ch1 = np.empty(n); difh1 = np.empty(n); ch2 = np.empty(n); ch3 = np.empty(n); difh2 = np.empty(n); difh3 = np.empty(n)
    for i in range(n):
        prev = max(0, i - 1)
        cc1[i] = close.iloc[i] if down.iloc[i] or i == 0 else min(close.iloc[i], cc1[prev])
        difl1[i] = diff.iloc[i] if down.iloc[i] or i == 0 else min(diff.iloc[i], difl1[prev])
        cc2[i] = cc1[prev] if up.iloc[i] else (close.iloc[i] if i == 0 else cc2[prev])
        cc3[i] = cc2[prev] if up.iloc[i] else (close.iloc[i] if i == 0 else cc3[prev])
        difl2[i] = difl1[prev] if up.iloc[i] else (diff.iloc[i] if i == 0 else difl2[prev])
        difl3[i] = difl2[prev] if up.iloc[i] else (diff.iloc[i] if i == 0 else difl3[prev])
        ch1[i] = close.iloc[i] if up.iloc[i] or i == 0 else max(close.iloc[i], ch1[prev])
        difh1[i] = diff.iloc[i] if up.iloc[i] or i == 0 else max(diff.iloc[i], difh1[prev])
        ch2[i] = ch1[prev] if down.iloc[i] else (close.iloc[i] if i == 0 else ch2[prev])
        ch3[i] = ch2[prev] if down.iloc[i] else (close.iloc[i] if i == 0 else ch3[prev])
        difh2[i] = difh1[prev] if down.iloc[i] else (diff.iloc[i] if i == 0 else difh2[prev])
        difh3[i] = difh2[prev] if down.iloc[i] else (diff.iloc[i] if i == 0 else difh3[prev])
    cc1s, cc2s, cc3s = map(lambda x: pd.Series(x), (cc1, cc2, cc3))
    dl1, dl2, dl3 = map(lambda x: pd.Series(x), (difl1, difl2, difl3))
    ch1s, ch2s, ch3s = map(lambda x: pd.Series(x), (ch1, ch2, ch3))
    dh1, dh2, dh3 = map(lambda x: pd.Series(x), (difh1, difh2, difh3))
    aaa = (cc1s < cc2s) & (dl1 > dl2) & (hist.shift(1) < 0) & (diff < 0)
    bbb = (cc1s < cc3s) & (dl1 < dl2) & (dl1 > dl3) & (hist.shift(1) < 0) & (diff < 0)
    ccc = (aaa | bbb) & (diff < 0)
    jjj = ccc.shift(1).fillna(False) & (diff.shift(1).abs() >= diff.abs() * 1.01)
    cd_bottom = jjj & ~jjj.shift(1).fillna(False)
    zj = (ch1s > ch2s) & (dh1 < dh2) & (hist.shift(1) > 0) & (diff > 0)
    gx = (ch1s > ch3s) & (dh1 > dh2) & (dh1 < dh3) & (hist.shift(1) > 0) & (diff > 0)
    db = (zj | gx) & (diff > 0)
    dbjg = db.shift(1).fillna(False) & (diff.shift(1) >= diff * 1.01)
    cd_top = dbjg & ~dbjg.shift(1).fillna(False)
    rsi = _wilder_rsi(close, 14)
    bull_div = (low < low.shift(1).rolling(10, min_periods=3).min()) & (rsi > rsi.shift(1).rolling(10, min_periods=3).min()) & (rsi <= 45)
    bear_div = (high > high.shift(1).rolling(10, min_periods=3).max()) & (rsi < rsi.shift(1).rolling(10, min_periods=3).max()) & (rsi >= 55)
    bull_recent = bull_div.rolling(rsi_confirm_bars, min_periods=1).max().astype(bool)
    bear_recent = bear_div.rolling(rsi_confirm_bars, min_periods=1).max().astype(bool)
    bottom = cd_bottom & (bull_recent if require_rsi_divergence else True)
    top = cd_top & (bear_recent if require_rsi_divergence else True)
    return pd.DataFrame({
        "diff": diff.to_numpy(), "dea": dea.to_numpy(), "hist": hist.to_numpy(), "rsi": rsi.to_numpy(),
        "bull_div": bull_div.to_numpy(), "bear_div": bear_div.to_numpy(), "bottom": bottom.to_numpy(), "top": top.to_numpy(),
    }, index=data.index)


def _quote_source_text(csv_path: str | Path) -> str:
    """Read a local quote list or a GitHub/raw HTTPS quote list."""
    source = str(csv_path).strip()
    if source.startswith("https://github.com/") and "/blob/" in source:
        source = source.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    if source.startswith(("https://", "http://")):
        request = Request(source, headers={"User-Agent": "ChanStructure/1.0"})
        with urlopen(request, timeout=20) as response:
            return response.read().decode("utf-8-sig", errors="replace")
    return Path(source).read_text(encoding="utf-8-sig", errors="replace")


def quote_symbols(csv_path: str | Path) -> list[str]:
    """Read a thinkorswim quote-list CSV from local storage or an HTTPS URL."""
    text = _quote_source_text(csv_path)
    header_row = None
    for number, line in enumerate(text.splitlines()):
        if line.strip().startswith("Symbol,"):
            header_row = number
            break
    if header_row is None:
        raise ValueError("CSV 中没有找到 Symbol 表头。")
    frame = pd.read_csv(StringIO(text), skiprows=header_row, usecols=["Symbol"], dtype=str)
    symbols = frame.Symbol.dropna().str.strip().str.upper()
    return list(dict.fromkeys(symbols[symbols.str.fullmatch(r"[A-Z0-9.\-]+")].tolist()))
