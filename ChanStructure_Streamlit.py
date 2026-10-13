"""ChanStructure — transparent, rule-based Chan-style chart research app.

This is an independently implemented approximation inspired by the supplied
chart image. It does not reverse engineer or reproduce a proprietary script.
Use it for research only, not as a trading recommendation.

Run: streamlit run ChanStructure_Streamlit.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

Kind = Literal["H", "L"]


@dataclass(frozen=True)
class Pivot:
    i: int
    t: pd.Timestamp
    price: float
    kind: Kind
    confirmed_i: int


@dataclass(frozen=True)
class Center:
    start_i: int
    end_i: int
    low: float       # ZD
    high: float      # ZG
    strokes: int
    direction: Literal["up", "down"]
    tier: Literal["major", "minor"]
    start_pivot: int
    end_pivot: int


@dataclass(frozen=True)
class Signal:
    i: int
    t: pd.Timestamp
    price: float
    text: str
    side: Literal["buy", "sell"]
    confidence: str


@st.cache_data(ttl=900, show_spinner=False)
def load_ohlcv(symbol: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(symbol, period=period, interval=interval, auto_adjust=True,
                       progress=False, multi_level_index=False)
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    if data.empty:
        raise ValueError("没有返回行情。请检查股票代码、周期和 Yahoo Finance 的盘中数据范围。")
    return data


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    fast = result.Close.ewm(span=12, adjust=False).mean()
    slow = result.Close.ewm(span=26, adjust=False).mean()
    result["macd_hist"] = (fast - slow) - (fast - slow).ewm(span=9, adjust=False).mean()
    tr = pd.concat([
        result.High - result.Low,
        (result.High - result.Close.shift()).abs(),
        (result.Low - result.Close.shift()).abs(),
    ], axis=1).max(axis=1)
    result["atr"] = tr.rolling(14, min_periods=1).mean()
    return result


def raw_fractals(data: pd.DataFrame, radius: int) -> list[Pivot]:
    """A pivot at bar i becomes known only at i + radius.
    Tied highs/lows are ignored, avoiding duplicate flat-range pivots.
    """
    high, low = data.High.to_numpy(), data.Low.to_numpy()
    out: list[Pivot] = []
    for i in range(radius, len(data) - radius):
        wh, wl = high[i-radius:i+radius+1], low[i-radius:i+radius+1]
        if high[i] == wh.max() and (wh == high[i]).sum() == 1:
            out.append(Pivot(i, data.index[i], float(high[i]), "H", i + radius))
        if low[i] == wl.min() and (wl == low[i]).sum() == 1:
            out.append(Pivot(i, data.index[i], float(low[i]), "L", i + radius))
    return sorted(out, key=lambda x: (x.i, x.kind))


def build_pivots(data: pd.DataFrame, radius: int, min_bars: int) -> list[Pivot]:
    """Simplified pen construction: alternate tops/bottoms; retain the more
    extreme candidate of the same type; require min_bars between opposite ends.
    It is deliberately deterministic rather than claiming any one school is canonical.
    """
    kept: list[Pivot] = []
    for p in raw_fractals(data, radius):
        if not kept:
            kept.append(p)
            continue
        prev = kept[-1]
        if p.kind == prev.kind:
            better = p.price > prev.price if p.kind == "H" else p.price < prev.price
            if better:
                kept[-1] = p
        elif p.i - prev.i >= min_bars:
            kept.append(p)
    return kept


def stroke_range(a: Pivot, b: Pivot) -> tuple[float, float]:
    return min(a.price, b.price), max(a.price, b.price)


def build_centers(pivots: list[Pivot], tier: Literal["major", "minor"]) -> list[Center]:
    """Build non-overlapping three-stroke overlaps and extend them while each
    following stroke overlaps the original [ZD, ZG]. The persisted stroke count
    is what makes 7/9-stroke zones visible.
    """
    centers: list[Center] = []
    stroke = 0
    while stroke + 3 < len(pivots):
        ranges = [stroke_range(pivots[k], pivots[k+1]) for k in range(stroke, stroke + 3)]
        zd, zg = max(x[0] for x in ranges), min(x[1] for x in ranges)
        if zd >= zg:
            stroke += 1
            continue
        end_stroke = stroke + 2
        while end_stroke + 2 < len(pivots):
            nzd, nzg = stroke_range(pivots[end_stroke+1], pivots[end_stroke+2])
            if nzg <= zd or nzd >= zg:
                break
            end_stroke += 1
        end_pivot = end_stroke + 1
        direction = "up" if pivots[end_pivot].price >= pivots[stroke].price else "down"
        centers.append(Center(
            pivots[stroke].i, pivots[end_pivot].i, zd, zg,
            end_stroke - stroke + 1, direction, tier, stroke, end_pivot,
        ))
        stroke = end_stroke + 1
    return centers


def segment_force(data: pd.DataFrame, start: int, end: int) -> float:
    """A reproducible proxy for '力度': absolute MACD histogram area."""
    lo, hi = sorted((start, end))
    return float(data.macd_hist.iloc[lo:hi+1].abs().sum())


def first_second_signals(data: pd.DataFrame, pivots: list[Pivot], force_ratio: float) -> list[Signal]:
    """Candidate 1/2 points. A 1 point needs a new price extreme plus weaker
    same-direction MACD force. A 2 point needs a later non-breaking retest.
    All are labelled 'candidate' because lower-level confirmation is not inferred.
    """
    signals: list[Signal] = []
    last_one_buy: Pivot | None = None
    last_one_sell: Pivot | None = None
    prior_low: Pivot | None = None
    prior_high: Pivot | None = None
    prior_low_force: float | None = None
    prior_high_force: float | None = None
    for n, p in enumerate(pivots):
        if n < 1:
            continue
        leg_force = segment_force(data, pivots[n-1].i, p.i)
        if p.kind == "L":
            if prior_low is not None and p.price < prior_low.price and prior_low_force is not None and leg_force < prior_low_force * force_ratio:
                signals.append(Signal(p.i, p.t, p.price, "1买", "buy", "背驰候选"))
                last_one_buy = p
            elif last_one_buy is not None and p.i > last_one_buy.i and p.price > last_one_buy.price:
                signals.append(Signal(p.i, p.t, p.price, "2买", "buy", "回踩候选"))
                last_one_buy = None
            prior_low, prior_low_force = p, leg_force
        else:
            if prior_high is not None and p.price > prior_high.price and prior_high_force is not None and leg_force < prior_high_force * force_ratio:
                signals.append(Signal(p.i, p.t, p.price, "1卖", "sell", "背驰候选"))
                last_one_sell = p
            elif last_one_sell is not None and p.i > last_one_sell.i and p.price < last_one_sell.price:
                signals.append(Signal(p.i, p.t, p.price, "2卖", "sell", "回抽候选"))
                last_one_sell = None
            prior_high, prior_high_force = p, leg_force
    return signals


def third_signals(pivots: list[Pivot], centers: list[Center], strict: bool) -> list[Signal]:
    """For each center: detect an exit, then only the first opposite pivot.
    3buy: first pullback low stays at/above ZG after upward departure.
    3sell: first rebound high stays at/below ZD after downward departure.
    """
    signals: list[Signal] = []
    for c in centers:
        after = pivots[c.end_pivot + 1:]
        state: str | None = None
        for p in after:
            if state is None:
                if p.kind == "H" and p.price > c.high:
                    state = "up"
                elif p.kind == "L" and p.price < c.low:
                    state = "down"
            elif state == "up" and p.kind == "L":
                valid = p.price > c.high if strict else p.price >= c.high
                if valid:
                    signals.append(Signal(p.i, p.t, p.price, "3买", "buy", "中枢外回踩"))
                break
            elif state == "down" and p.kind == "H":
                valid = p.price < c.low if strict else p.price <= c.low
                if valid:
                    signals.append(Signal(p.i, p.t, p.price, "3卖", "sell", "中枢外回抽"))
                break
    # Deduplicate identical pivot/type signals from nested centers.
    return list({(x.i, x.text): x for x in signals}.values())


def zone_fill(c: Center, large_at: int) -> tuple[str, str]:
    """Four palette based on trend and actual center extension length.
    A zone becomes 'large' at large_at strokes (9 by default), irrespective of
    which fractal scale discovered it. This keeps the colour meaning identical
    in Streamlit and Colab.
    """
    is_large = c.strokes >= large_at
    if is_large and c.direction == "up":
        return "rgba(174, 255, 0, .28)", "#AFFF00"       # bright lime
    if not is_large and c.direction == "up":
        return "rgba(0, 210, 128, .24)", "#00D280"        # green
    if not is_large and c.direction == "down":
        return "rgba(174, 45, 182, .24)", "#AE2DB6"       # purple
    return "rgba(231, 76, 60, .25)", "#E74C3C"            # red


def unique_levels(data: pd.DataFrame, centers: list[Center], pivots: list[Pivot], limit: int) -> list[tuple[float, str]]:
    """Current support/resistance candidates from recent center edges and pivots."""
    current, atr = float(data.Close.iloc[-1]), float(data.atr.iloc[-1])
    candidates: list[float] = []
    for c in centers[-8:]:
        candidates.extend([c.low, c.high])
    candidates.extend(p.price for p in pivots[-12:])
    distinct: list[float] = []
    for x in sorted(candidates, key=lambda v: abs(v-current)):
        if not any(abs(x-y) <= max(atr * .16, current * .001) for y in distinct):
            distinct.append(x)
    result = [(x, "阻力" if x > current else "支撑") for x in distinct[:limit]]
    return sorted(result, key=lambda z: z[0])


def plot_chart(data: pd.DataFrame, minor: list[Pivot], major: list[Pivot], centers: list[Center], signals: list[Signal], levels: list[tuple[float, str]], show_minor: bool, large_at: int) -> go.Figure:
    fig = go.Figure(go.Candlestick(x=data.index, open=data.Open, high=data.High,
                                   low=data.Low, close=data.Close, name="价格"))
    # Nested center rectangles, major first so minor zones remain visible on top.
    for c in sorted(centers, key=lambda x: 0 if x.tier == "major" else 1):
        fill, edge = zone_fill(c, large_at)
        fig.add_shape(type="rect", x0=data.index[c.start_i], x1=data.index[c.end_i],
                      y0=c.low, y1=c.high, fillcolor=fill,
                      line={"color": edge, "width": 1.3}, layer="below")
        label = f"{'大' if c.strokes >= large_at else '小'}中枢 {c.strokes}笔"
        if c.strokes >= 7:
            label += " ★"
        if c.strokes >= 9:
            label += "★"
        fig.add_annotation(x=data.index[c.start_i], y=c.high, text=label, showarrow=False,
                           font={"color": edge, "size": 10}, xanchor="left", yanchor="bottom")
    for pivs, width, name in ((major, 2.3, "大级别结构"), (minor, 1.25, "小级别结构")):
        if pivs is minor and not show_minor:
            continue
        for a, b in zip(pivs, pivs[1:]):
            fig.add_shape(type="line", x0=a.t, y0=a.price, x1=b.t, y1=b.price,
                          line={"color": "#00d68f" if b.price > a.price else "#ff4d5a", "width": width})
    # Candidate labels are deliberately smaller and say '候选' in the hover text.
    for side, color, symbol in (("buy", "#00C98D", "triangle-up"), ("sell", "#FF5A5F", "triangle-down")):
        s = [x for x in signals if x.side == side]
        if s:
            fig.add_trace(go.Scatter(x=[x.t for x in s], y=[x.price for x in s], mode="markers+text",
                text=[x.text for x in s], textposition="bottom center" if side == "buy" else "top center",
                marker={"size": 17, "color": color, "symbol": symbol, "line": {"color": "white", "width": 1.5}},
                customdata=[[x.confidence] for x in s], hovertemplate="%{text}<br>%{customdata[0]}<extra></extra>", name="买卖点候选"))
    for value, name in levels:
        color = "#00a9e8" if name == "阻力" else "#f5a623"
        fig.add_hline(y=value, line={"color": color, "dash": "dot", "width": 1},
                      annotation_text=f"{name} {value:.2f}", annotation_position="right")
    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False,
        title="ChanStructure：嵌套中枢、1/2/3 买卖候选、支撑阻力",
        margin={"l": 15, "r": 140, "t": 45, "b": 20}, legend={"orientation": "h"})
    return fig


st.set_page_config(page_title="ChanStructure", layout="wide")
st.title("ChanStructure 多层缠论结构图")
st.caption("透明规则近似版：中枢、信号和支撑阻力均可在代码中审计；不构成投资建议。")

with st.sidebar:
    symbol = st.text_input("股票代码", "AAPL").strip().upper()
    interval = st.selectbox("K线周期", ["1d", "1h", "30m", "15m", "1wk"])
    options = {"1d": ["6mo", "1y", "2y", "5y", "10y"], "1h": ["60d", "180d", "730d"],
               "30m": ["30d", "60d"], "15m": ["30d", "60d"], "1wk": ["1y", "2y", "5y", "10y"]}
    period = st.selectbox("取数范围", options[interval])
    st.subheader("小级别结构")
    radius = st.slider("分型确认半径", 1, 10, 3)
    min_bars = st.slider("一笔最少 K 线", 1, 30, 5)
    st.subheader("大级别嵌套结构")
    scale = st.slider("大级别倍率", 2, 5, 2)
    strict_three = st.checkbox("三买/卖严格不触及中枢", value=True)
    large_at = st.slider("大中枢最少延伸笔数", 7, 15, 9)
    force_ratio = st.slider("一买/卖力度阈值", .30, .95, .72, .01)
    level_count = st.slider("支撑阻力条数", 2, 14, 8)
    show_minor = st.checkbox("显示小级别笔线", value=True)

try:
    data = add_indicators(load_ohlcv(symbol, period, interval))
    minor = build_pivots(data, radius, min_bars)
    major = build_pivots(data, radius * scale, min_bars * scale)
    minor_centers = build_centers(minor, "minor")
    major_centers = build_centers(major, "major")
    centers = minor_centers + major_centers
    signals = (first_second_signals(data, minor, force_ratio)
               + third_signals(minor, minor_centers, strict_three)
               + third_signals(major, major_centers, strict_three))
    levels = unique_levels(data, centers, minor + major, level_count)
    st.plotly_chart(plot_chart(data, minor, major, centers, signals, levels, show_minor, large_at), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.metric("已确认小级别端点", len(minor))
        st.dataframe(pd.DataFrame([{"层级": "大" if x.strokes >= large_at else "小", "方向": "上" if x.direction == "up" else "下", "开始": data.index[x.start_i], "结束": data.index[x.end_i], "ZD": round(x.low, 2), "ZG": round(x.high, 2), "笔数": x.strokes} for x in sorted(centers, key=lambda c: c.end_i, reverse=True)]), hide_index=True, use_container_width=True)
    with right:
        st.metric("1/2/3 买卖候选", len(signals))
        st.dataframe(pd.DataFrame([{"时间": x.t, "信号": x.text, "价格": round(x.price, 2), "说明": x.confidence} for x in sorted(signals, key=lambda x: x.i, reverse=True)]), hide_index=True, use_container_width=True)
except Exception as exc:
    st.error(f"无法载入或计算 {symbol}：{exc}")

