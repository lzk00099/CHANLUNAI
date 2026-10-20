"""ChanLite: a research/visualisation tool, not investment advice.

Run locally:  pip install -r requirements.txt
              streamlit run streamlit_chanlite_app.py
Deploy: commit this file and requirements.txt to GitHub, then choose that repo
in Streamlit Community Cloud.  Yahoo Finance data can be delayed/incomplete.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


@dataclass(frozen=True)
class Pivot:
    pos: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["H", "L"]
    confirmed_at: int


@dataclass(frozen=True)
class Center:
    start: int
    end: int
    low: float
    high: float
    stroke_count: int


@st.cache_data(ttl=900, show_spinner=False)
def load_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True,
                     progress=False, multi_level_index=False)
    if df.empty:
        raise ValueError("No price data returned. Check the ticker and period.")
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def raw_fractals(df: pd.DataFrame, radius: int) -> list[Pivot]:
    """Fractals are confirmed radius bars later; the last radius bars are unknown."""
    highs, lows = df.High.to_numpy(), df.Low.to_numpy()
    found: list[Pivot] = []
    for i in range(radius, len(df) - radius):
        hi_window = highs[i - radius:i + radius + 1]
        lo_window = lows[i - radius:i + radius + 1]
        # Require a unique extreme so flat sections do not create duplicate pivots.
        if highs[i] == hi_window.max() and (hi_window == highs[i]).sum() == 1:
            found.append(Pivot(i, df.index[i], float(highs[i]), "H", i + radius))
        if lows[i] == lo_window.min() and (lo_window == lows[i]).sum() == 1:
            found.append(Pivot(i, df.index[i], float(lows[i]), "L", i + radius))
    return sorted(found, key=lambda p: (p.pos, 0 if p.kind == "H" else 1))


def build_strokes(df: pd.DataFrame, radius: int, min_bars: int) -> list[Pivot]:
    """Simplified pen rule: alternating confirmed fractals, minimum separation,
    and replace a same-side candidate with its more extreme value.

    This intentionally does NOT claim to implement every Chan inclusion or
    segment-school variant. It is deterministic and suitable for experimentation.
    """
    accepted: list[Pivot] = []
    for p in raw_fractals(df, radius):
        if not accepted:
            accepted.append(p)
            continue
        prev = accepted[-1]
        if p.kind == prev.kind:
            better = p.price > prev.price if p.kind == "H" else p.price < prev.price
            if better:
                accepted[-1] = p
        elif p.pos - prev.pos >= min_bars:
            accepted.append(p)
    return accepted


def range_for(a: Pivot, b: Pivot) -> tuple[float, float]:
    return min(a.price, b.price), max(a.price, b.price)


def build_centers(pivots: list[Pivot]) -> list[Center]:
    """Create non-overlapping, consecutive 3-stroke centers and extend each
    while the next stroke overlaps the original center.

    Zone color in the chart expresses the number of strokes it survived:
    purple=3–4, blue=5–6, orange=7–8, green=9+.  This makes 7/9-stroke zones
    immediately visible without calling them buy/sell signals.
    """
    centers: list[Center] = []
    # A three-stroke center needs four endpoints.
    stroke_i = 0
    while stroke_i + 2 < len(pivots) - 1:
        ranges = [range_for(pivots[j], pivots[j + 1]) for j in range(stroke_i, stroke_i + 3)]
        low, high = max(x[0] for x in ranges), min(x[1] for x in ranges)
        if low >= high:
            stroke_i += 1
            continue
        end_stroke = stroke_i + 2
        while end_stroke + 1 < len(pivots) - 1:
            nlow, nhigh = range_for(pivots[end_stroke + 1], pivots[end_stroke + 2])
            if nhigh <= low or nlow >= high:
                break
            end_stroke += 1
        centers.append(Center(
            start=pivots[stroke_i].pos,
            end=pivots[end_stroke + 1].pos,
            low=low, high=high, stroke_count=end_stroke - stroke_i + 1,
        ))
        stroke_i = end_stroke + 1
    return centers


def zone_color(strokes: int) -> str:
    if strokes >= 9:
        return "rgba(46, 204, 113, 0.25)"   # green
    if strokes >= 7:
        return "rgba(243, 156, 18, 0.25)"   # orange
    if strokes >= 5:
        return "rgba(52, 152, 219, 0.23)"   # blue
    return "rgba(155, 89, 182, 0.22)"       # purple


def chart(df: pd.DataFrame, pivots: list[Pivot], centers: list[Center]) -> go.Figure:
    fig = go.Figure(go.Candlestick(
        x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
        name="Price",
    ))
    # Individual strokes are green for upward and red for downward movement.
    for a, b in zip(pivots, pivots[1:]):
        fig.add_shape(type="line", x0=a.timestamp, y0=a.price, x1=b.timestamp, y1=b.price,
                      line={"color": "#16a085" if b.price > a.price else "#e74c3c", "width": 2})
    for c in centers:
        fig.add_shape(type="rect", x0=df.index[c.start], x1=df.index[c.end],
                      y0=c.low, y1=c.high, line={"color": zone_color(c.stroke_count).replace("0.25", "0.95").replace("0.23", "0.95").replace("0.22", "0.95"), "width": 1},
                      fillcolor=zone_color(c.stroke_count), layer="below")
        fig.add_annotation(x=df.index[c.start], y=c.high, text=f"{c.stroke_count}笔中枢",
                           showarrow=False, xanchor="left", yanchor="bottom", font={"size": 10})
    fig.update_layout(height=720, xaxis_rangeslider_visible=False, template="plotly_white",
                      legend={"orientation": "h"}, margin={"l": 20, "r": 20, "t": 35, "b": 20})
    return fig


st.set_page_config(page_title="ChanLite 多周期诊股", layout="wide")
st.title("ChanLite 多周期结构图")
st.caption("研究与可视化工具，不构成投资建议。分型在右侧确认 K 线出现前均为未确认。")

with st.sidebar:
    symbol = st.text_input("股票代码", "AAPL").strip().upper()
    interval = st.selectbox("K线周期", ["1d", "1h", "30m", "15m", "1wk"])
    periods_by_interval = {
        "1d": ["6mo", "1y", "2y", "5y", "10y"],
        "1h": ["60d", "180d", "730d"],
        "30m": ["30d", "60d"],
        "15m": ["30d", "60d"],
        "1wk": ["1y", "2y", "5y", "10y"],
    }
    period = st.selectbox("取数范围", periods_by_interval[interval], index=0)
    radius = st.slider("分型确认半径（K线）", 1, 10, 3)
    min_bars = st.slider("相邻笔最少间隔（K线）", 1, 30, 5)

try:
    data = load_data(symbol, period, interval)
    pivots = build_strokes(data, radius, min_bars)
    centers = build_centers(pivots)
    st.plotly_chart(chart(data, pivots, centers), use_container_width=True)
    st.write(f"已确认端点：{len(pivots)}；中枢：{len(centers)}。")
    st.dataframe(pd.DataFrame([{
        "开始": data.index[c.start], "结束": data.index[c.end], "ZD": round(c.low, 2),
        "ZG": round(c.high, 2), "笔数": c.stroke_count,
        "级别色": "绿 ≥9" if c.stroke_count >= 9 else "橙 7–8" if c.stroke_count >= 7 else "蓝 5–6" if c.stroke_count >= 5 else "紫 3–4",
    } for c in centers]), use_container_width=True, hide_index=True)
except Exception as exc:
    st.error(f"无法载入或计算 {symbol}：{exc}")

