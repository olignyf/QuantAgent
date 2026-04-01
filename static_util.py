import base64
import io

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

import color_style as color
from graph_util import (
    fit_trendlines_high_low,
    fit_trendlines_single,
    get_line_points,
    split_line_into_segments,
)

matplotlib.use("Agg")

HIGH_RES_DPI = 600
HIGH_RES_FIGSIZE = (24, 12)


def _infer_bar_minutes(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 60.0
    deltas = pd.Series(index).diff().dropna().dt.total_seconds() / 60.0
    if deltas.empty:
        return 60.0
    return float(deltas.median())


def _apply_time_ticks(ax, index: pd.DatetimeIndex) -> None:
    """Set readable time ticks without hitting Locator.MAXTICKS."""
    if len(index) == 0:
        return

    bar_minutes = _infer_bar_minutes(index)

    # Tick step in number of bars.
    if bar_minutes <= 1.5:
        step = 5     # 1m bars -> every 5m
    elif bar_minutes <= 5.5:
        step = 3     # 5m bars -> every 15m
    elif bar_minutes <= 15.5:
        step = 2     # 15m bars -> every 30m
    elif bar_minutes <= 30.5:
        step = 2     # 30m bars -> every 1h
    elif bar_minutes <= 60.5:
        step = 2     # 60m bars -> every 2h
    elif bar_minutes <= 240.5:
        step = 2     # 4h bars -> every 8h
    elif bar_minutes <= 24 * 60 + 1:
        step = 1
    else:
        step = 1

    positions = np.arange(0, len(index), step, dtype=int)
    if len(positions) == 0 or positions[-1] != len(index) - 1:
        positions = np.append(positions, len(index) - 1)

    labels = []
    for pos in positions:
        ts = index[pos]
        if bar_minutes <= 240.5:
            labels.append(ts.strftime("%m-%d %H:%M"))
        else:
            labels.append(ts.strftime("%Y-%m-%d"))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


def generate_kline_image(kline_data) -> dict:
    """
    Generate a candlestick (K-line) chart from OHLCV data, save it locally, and return a base64-encoded image.

    Args:
        kline_data (dict): Dictionary with keys including 'Datetime', 'Open', 'High', 'Low', 'Close'.
        filename (str): Name of the file to save the image locally (default: 'kline_chart.png').

    Returns:
        dict: Dictionary containing base64-encoded image string and local file path.
    """

    df = pd.DataFrame(kline_data)
    # take recent 40
    df = df.tail(40)

    df.to_csv("record.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    try:
        # df.index = pd.to_datetime(df["Datetime"])
        df.index = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S")

    except ValueError:
        print("ValueError at graph_util.py\n")

    # Save image locally
    fig, axlist = mpf.plot(
        df[["Open", "High", "Low", "Close"]],
        type="candle",
        style=color.my_color_style,
        figsize=HIGH_RES_FIGSIZE,
        returnfig=True,
        block=False,
    )
    axlist[0].set_ylabel("Price", fontweight="normal")
    axlist[0].set_xlabel("Datetime", fontweight="normal")
    _apply_time_ticks(axlist[0], df.index)
    # mplfinance axes are not fully compatible with tight_layout.
    fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.22)

    fig.savefig(
        fname="kline_chart.png",
        dpi=HIGH_RES_DPI,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    # ---------- Encode to base64 -----------------
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=HIGH_RES_DPI,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)  # release memory

    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "pattern_image": img_b64,
        "pattern_image_description": "Candlestick chart saved locally and returned as base64 string.",
    }


def generate_trend_image(kline_data) -> dict:
    """
    Generate a candlestick chart with trendlines from OHLCV data,
    save it locally as 'trend_graph.png', and return a base64-encoded image.

    Returns:
        dict: base64 image and description
    """
    data = pd.DataFrame(kline_data)
    candles = data.iloc[-50:].copy()

    candles["Datetime"] = pd.to_datetime(candles["Datetime"])
    candles.set_index("Datetime", inplace=True)

    # Trendline fit functions assumed to be defined outside this scope
    support_coefs_c, resist_coefs_c = fit_trendlines_single(candles["Close"])
    support_coefs, resist_coefs = fit_trendlines_high_low(
        candles["High"], candles["Low"], candles["Close"]
    )

    # Trendline values
    support_line_c = support_coefs_c[0] * np.arange(len(candles)) + support_coefs_c[1]
    resist_line_c = resist_coefs_c[0] * np.arange(len(candles)) + resist_coefs_c[1]
    support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
    resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]

    # Convert to time-anchored coordinates
    s_seq = get_line_points(candles, support_line)
    r_seq = get_line_points(candles, resist_line)
    s_seq2 = get_line_points(candles, support_line_c)
    r_seq2 = get_line_points(candles, resist_line_c)

    s_segments = split_line_into_segments(s_seq)
    r_segments = split_line_into_segments(r_seq)
    s2_segments = split_line_into_segments(s_seq2)
    r2_segments = split_line_into_segments(r_seq2)

    all_segments = s_segments + r_segments + s2_segments + r2_segments
    colors = (
        ["white"] * len(s_segments)
        + ["white"] * len(r_segments)
        + ["blue"] * len(s2_segments)
        + ["red"] * len(r2_segments)
    )

    # Create addplot lines for close-based support/resistance
    apds = [
        mpf.make_addplot(support_line_c, color="blue", width=1, label="Close Support"),
        mpf.make_addplot(resist_line_c, color="red", width=1, label="Close Resistance"),
    ]

    # Generate figure with legend and save locally
    fig, axlist = mpf.plot(
        candles,
        type="candle",
        style=color.my_color_style,
        addplot=apds,
        alines=dict(alines=all_segments, colors=colors, linewidths=1),
        returnfig=True,
        figsize=HIGH_RES_FIGSIZE,
        block=False,
    )

    axlist[0].set_ylabel("Price", fontweight="normal")
    axlist[0].set_xlabel("Datetime", fontweight="normal")
    _apply_time_ticks(axlist[0], candles.index)

    # Add legend before final renders
    axlist[0].legend(loc="upper left")
    # mplfinance axes are not fully compatible with tight_layout.
    fig.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.22)

    # save fig locally
    fig.savefig(
        "trend_graph.png",
        format="png",
        dpi=HIGH_RES_DPI,
        bbox_inches="tight",
        pad_inches=0.1,
    )

    # Save to base64
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=HIGH_RES_DPI,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return {
        "trend_image": img_b64,
        "trend_image_description": "Trend-enhanced candlestick chart with support/resistance lines.",
    }
