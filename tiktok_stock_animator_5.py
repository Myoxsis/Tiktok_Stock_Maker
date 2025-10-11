#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
import matplotlib.dates as mdates

# ---------------------------
# i18n
# ---------------------------

LANGS = {
    "en": {
        "title_fixed": "{company} — Fixed Investment",
        "title_dca": "{company} — Recurrent Investment ({freq_disp})",
        "subtitle_fixed": "You invested {amount} on {date}",
        "subtitle_dca": "You invest {amount} every {freq} since {date}",
        "label_invested": "Invested",
        "label_value": "Value",
        "label_shares": "Shares",
        "label_return": "Return",
        "axis_date": "Date",
        "axis_price": "Price",
        "pos": "+",
        "neg": "-",
        "freq_weekly_disp": "Weekly",
        "freq_monthly_disp": "Monthly",
        "freq_yearly_disp": "Yearly",
        "freq_weekly": "week",
        "freq_monthly": "month",
        "freq_yearly": "year",
        "handle": "@investir_intelligent",
        "no_frames": "No data to animate: your visible date range is empty. Check your CSV and --start/--end.",
        "warn_shift_start": "[WARN] --start {req} is before CSV. Shifting to {used}.",
        "warn_shift_end": "[WARN] --end {req} is after CSV. Shifting to {used}.",
        "info_csv": "[INFO] CSV covers {min} .. {max}",
        "info_render": "[INFO] Rendering from {start} to {end}",
        "err_writer": "ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`) or export a GIF by using .gif with ImageMagick.",
        "word_gain": "Gain",
        "word_loss": "Loss",
    },
    "fr": {
        "title_fixed": "{company} — Investissement initial",
        "title_dca": "{company} — DCA ({freq_disp})",
        "subtitle_fixed": "Tu investi {amount} le {date}",
        "subtitle_dca": "Tu investi {amount} chaque {freq} depuis le {date}",
        "label_invested": "Investi",
        "label_value": "Valeur",
        "label_shares": "Parts",
        "label_return": "Performance",
        "axis_date": "Date",
        "axis_price": "Prix",
        "pos": "+",
        "neg": "−",
        "freq_weekly_disp": "Hebdomadaire",
        "freq_monthly_disp": "Mensuel",
        "freq_yearly_disp": "Annuel",
        "freq_weekly": "semaine",
        "freq_monthly": "mois",
        "freq_yearly": "an",
        "handle": "@investir_intelligent",
        "no_frames": "Aucune donnée à animer : la plage visible est vide. Vérifiez le CSV et --start/--end.",
        "warn_shift_start": "[AVERT] --start {req} est avant le CSV. Décalé à {used}.",
        "warn_shift_end": "[AVERT] --end {req} est après le CSV. Décalé à {used}.",
        "info_csv": "[INFO] Le CSV couvre {min} .. {max}",
        "info_render": "[INFO] Rendu de {start} à {end}",
        "err_writer": "ffmpeg introuvable dans PATH. Installez-le (`brew install ffmpeg`) ou exportez un GIF avec ImageMagick.",
        "word_gain": "Gain",
        "word_loss": "Perte",
    },
}

def get_lang(lang_code: str):
    return LANGS.get(lang_code, LANGS["en"])

def display_freq(lang: dict, freq_key: str):
    return {
        "weekly": lang["freq_weekly_disp"],
        "monthly": lang["freq_monthly_disp"],
        "yearly": lang["freq_yearly_disp"],
    }[freq_key]

def display_freq_inline(lang: dict, freq_key: str):
    return {
        "weekly": lang["freq_weekly"],
        "monthly": lang["freq_monthly"],
        "yearly": lang["freq_yearly"],
    }[freq_key]

# ---------------------------
# Data loading / utils
# ---------------------------

def load_prices(csv_path, date_col="Date", price_col="Close", tz_aware=False):
    df = pd.read_csv(csv_path)
    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"CSV must have columns '{date_col}' and '{price_col}'. Found: {df.columns.tolist()}")

    df = df[[date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=tz_aware, errors="coerce")
    df = df.dropna(subset=[date_col, price_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    # Force datetime64[ns]
    df[date_col] = pd.DatetimeIndex(df[date_col])
    # Deduplicate dates (keep last)
    df = df.groupby(date_col, as_index=False).agg({price_col: "last"})
    return df.rename(columns={date_col: "date", price_col: "close"})

def clamp_date_to_series(date, dates_like):
    dates_idx = pd.DatetimeIndex(dates_like)
    ts = pd.Timestamp(date)
    pos = dates_idx.searchsorted(ts, side="left")
    pos = min(max(pos, 0), len(dates_idx) - 1)
    return dates_idx[pos]

def generate_schedule(start, end, freq):
    out = []
    cur = start
    if freq == "weekly":
        delta = relativedelta(weeks=1)
    elif freq == "monthly":
        delta = relativedelta(months=1)
    elif freq == "yearly":
        delta = relativedelta(years=1)
    else:
        raise ValueError("freq must be one of: weekly, monthly, yearly")
    while cur <= end:
        out.append(cur)
        cur = cur + delta
    return out

@dataclass
class PortfolioSnapshot:
    date: pd.Timestamp
    price: float
    shares: float
    invested: float
    value: float
    pnl_pct: float

# ---------------------------
# Portfolio series
# ---------------------------

def series_fixed_investment(prices: pd.DataFrame, start_date: datetime, amount: float):
    dates_idx = pd.DatetimeIndex(prices["date"])
    start = clamp_date_to_series(pd.Timestamp(start_date), dates_idx)
    start_idx = dates_idx.get_loc(start)

    buy_price = prices.loc[start_idx, "close"]
    shares = 0.0 if buy_price == 0 else amount / buy_price

    snapshots = []
    for i in range(start_idx, len(prices)):
        d = prices.loc[i, "date"]
        p = prices.loc[i, "close"]
        value = shares * p
        invested = amount
        pnl_pct = 0.0 if invested == 0 else (value / invested - 1.0) * 100
        snapshots.append(PortfolioSnapshot(d, p, shares, invested, value, pnl_pct))

    vis_dates = prices.loc[start_idx:, "date"].to_numpy()
    vis_close = prices.loc[start_idx:, "close"].to_numpy()
    return snapshots, vis_dates, vis_close, start

def series_dca(prices: pd.DataFrame, start_date: datetime, amount_per: float, freq: str, end_date=None):
    if end_date is None:
        end_date = prices["date"].iloc[-1].to_pydatetime()

    schedule = generate_schedule(start_date, end_date, freq)
    dates_idx = pd.DatetimeIndex(prices["date"])

    # Shift each scheduled buy to the next available trading date
    buy_dates = []
    for d in schedule:
        t = clamp_date_to_series(pd.Timestamp(d), dates_idx)
        buy_dates.append(pd.Timestamp(t))
    buy_set = set(buy_dates)

    shares = 0.0
    invested = 0.0
    snapshots = []

    for i in range(len(prices)):
        d = prices.loc[i, "date"]
        p = prices.loc[i, "close"]

        if d in buy_set and p > 0:
            shares += amount_per / p
            invested += amount_per

        value = shares * p
        pnl_pct = 0.0 if invested == 0 else (value / invested - 1.0) * 100
        if d >= pd.Timestamp(start_date):
            snapshots.append(PortfolioSnapshot(d, p, shares, invested, value, pnl_pct))

    vis_dates = np.array([s.date for s in snapshots])
    vis_close = np.array([s.price for s in snapshots])
    return snapshots, vis_dates, vis_close, pd.Timestamp(start_date)

# ---------------------------
# Rendering helpers
# ---------------------------

def style_axes(ax):
    ax.set_facecolor("#0b0f14")
    ax.tick_params(colors="#D8E1E8", labelsize=14)
    for spine in ax.spines.values():
        spine.set_color("#22303C")
        spine.set_linewidth(1.2)

def format_euro(x: float, lang_code: str = "en"):
    # Full figure, no decimals (e.g., 1050€)
    return f"{int(round(x))}€"

# ---------------------------
# Animation
# ---------------------------

def make_animation(
    dates, prices, snapshots, title, subtitle, outfile,
    fps=30, speed=1.0, dpi=100, lang=None,
    reveal_sec=60.0,
    freeze_hold_sec=0.0,  # kept but ignored for the reveal schedule
):
    """
    Plots the data continuously for the requested duration (``reveal_sec`` seconds),
    then shows a 2-second end card (CTA + handle).
    Now with DYNAMIC Y-AXIS (updates every frame with padding).
    Legend removed. Labels visible from frame 1 (follow -> pin at 0.70).
    Dynamic x-axis growth, lowered top text, extra margins, watermark, gain/loss.
    """
    import os, re, shutil
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.animation import FFMpegWriter
    from matplotlib.ticker import FuncFormatter
    from matplotlib.transforms import blended_transform_factory
    from matplotlib.patches import Rectangle
    import matplotlib.dates as mdates
    try:
        from PIL import Image
        PIL_OK = True
    except Exception:
        PIL_OK = False

    # -------- Layout knobs --------
    PLOT_RECT    = [0.14, 0.28, 0.72, 0.54]   # plot (smaller → more margin)
    PBAR_RECT    = [0.14, 0.22, 0.72, 0.015]  # progress bar
    PL_RECT      = [0.14, 0.19, 0.72, 0.022]  # Gain/Loss row
    TITLE_Y      = 0.92                        # lowered for Dynamic Island
    SUBTITLE_Y   = 0.89
    HANDLE_Y     = 0.16
    LABEL_X_FRAC = 0.985
    WATERMARK_ALPHA = 0.25
    BG           = "#0b0f14"
    CTA_TEXT     = "Tu veux que je teste quoi ensuite ?"
    reveal_sec = max(float(reveal_sec), 1.0)    # <-- line reveals for chosen duration
    ENDCARD_SEC  = 2.0                         # 2s end card shown after the reveal

    # ---- Dynamic axes ----
    # X: grow visible span as the animation progresses
    DYNAMIC_XLIM         = True
    MIN_START_FRAC       = 0.15
    RIGHT_MARGIN_FRAC    = 0.02
    RIGHT_MARGIN_MIN_NS  = int(6 * 3600 * 1e9)
    # Y: recompute limits every frame from visible data (+ padding)
    DYNAMIC_YLIM         = True
    YPAD_FRAC            = 0.10   # 10% vertical padding
    CUSHION_FRAC         = 0.03   # label cushion inside y-lims
    FOLLOW_TO_PIN_FRAC   = 0.70   # as requested
    # --------------------------------

    lang = lang or get_lang("en")
    gain_word = lang.get("word_gain", "Gain")
    loss_word = lang.get("word_loss", "Loss")

    # --- Normalize time axis ---
    dates = pd.to_datetime(dates)
    dates_ns = dates.view("i8").astype(np.int64)
    frames_total = len(dates)
    if frames_total == 0:
        raise ValueError(lang["no_frames"])

    # --- Series (EUR) ---
    invested_series = np.array([s.invested for s in snapshots], dtype=float)
    value_series    = np.array([s.value    for s in snapshots], dtype=float)

    # ===== Frame schedule: reveal + 2s end card =====
    reveal_frames  = max(1, int(round(reveal_sec * fps)))
    endcard_frames = max(0, int(round(ENDCARD_SEC * fps)))

    # Map animation frames to data indices linearly across the dataset.
    # Ensures the last reveal frame reaches the final data point.
    data_indices = np.linspace(1, frames_total, reveal_frames, dtype=int)
    data_indices[0]  = max(1, data_indices[0])
    data_indices[-1] = frames_total

    frames = [("chart", int(k)) for k in data_indices] + [("end", i) for i in range(endcard_frames)]
    # =====================================================

    # --- Figure (portrait) ---
    width_px, height_px = 1080, 1920
    fig_w, fig_h = width_px / dpi, height_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(BG)

    # Main axes (scaled down for margin)
    ax = plt.axes(PLOT_RECT); ax.set_zorder(2)
    style_axes(ax)

    # Titles / handle (lowered for Dynamic Island safety)
    fig.text(0.50, TITLE_Y,    title,    ha="center", va="top", color="#FFFFFF", fontsize=28, weight="bold")
    fig.text(0.50, SUBTITLE_Y, subtitle, ha="center", va="top", color="#9FB3C8", fontsize=18)
    fig.text(0.03,  HANDLE_Y,  lang["handle"], ha="left", va="bottom", color="#5A6B7A", fontsize=16)

    # Initial axis limits (will be updated dynamically)
    xmin_dt, xmax_dt = dates[0], dates[-1]
    xmin_ns, xmax_ns = int(dates_ns[0]), int(dates_ns[-1])
    span_ns_data = max(xmax_ns - xmin_ns, 1)
    right_pad_ns_static = max(int(span_ns_data * 0.06), int(12 * 3600 * 1e9))
    xlim_right = xmax_dt + pd.to_timedelta(right_pad_ns_static, unit="ns")
    def _ns(dt) -> int: return int(pd.Timestamp(dt).value)
    xlim_right_ns = _ns(xlim_right)
    span_ns_full  = max(xlim_right_ns - xmin_ns, 1)

    # Start with full padded xlim; ylim will be set in the first update()
    ax.set_xlim(xmin_dt, xlim_right)
    ax.set_xlabel(lang["axis_date"], color="#D8E1E8", fontsize=14)
    ax.set_ylabel("€", color="#D8E1E8", fontsize=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(round(y))}€"))

    # Readable X-axis ticks/labels
    span_days = max(1, int((xlim_right - xmin_dt).days))
    target_ticks = 7
    if span_days <= 40:
        interval = max(1, int(np.ceil(span_days / target_ticks)))
        major_locator = mdates.DayLocator(interval=interval)
        major_fmt     = mdates.DateFormatter('%d %b')
        minor_locator = mdates.DayLocator(interval=max(1, interval // 2))
    elif span_days <= 370:
        span_months = max(1, int(np.ceil(span_days / 30.44)))
        interval = max(1, int(np.ceil(span_months / target_ticks)))
        major_locator = mdates.MonthLocator(interval=interval)
        major_fmt     = mdates.DateFormatter('%b %Y')
        minor_locator = mdates.MonthLocator(interval=max(1, interval // 2))
    else:
        span_years  = max(1, int(np.ceil(span_days / 365.25)))
        interval    = max(1, int(np.ceil(span_years / target_ticks)))
        major_locator = mdates.YearLocator(base=interval)
        major_fmt     = mdates.DateFormatter('%Y')
        minor_locator = mdates.MonthLocator(bymonth=(1, 7))
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.tick_params(axis='x', labelsize=14, rotation=0, pad=6)
    ax.grid(which='major', axis='x', alpha=0.12)

    # Watermark logo inside plot (centered, behind lines)
    def _company_from_title(t: str) -> str:
        for sep in ["—", "–", "-", "|"]:
            if sep in t:
                return t.split(sep)[0].strip()
        return t.strip()
    def _slug(s: str) -> str:
        s = s.strip().replace(" ", "_")
        return re.sub(r"[^A-Za-z0-9_\-]", "", s)
    def _find_logo_path(name: str, assets_dir="assets"):
        if not os.path.isdir(assets_dir):
            return None
        name_variants = [name, name.replace(" ", "_"), _slug(name)]
        for n in name_variants:
            for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
                p = os.path.join(assets_dir, f"{n}{ext}")
                if os.path.exists(p):
                    return p
        return None
    if PIL_OK:
        company_guess = _company_from_title(title)
        logo_path = _find_logo_path(company_guess)
        if logo_path:
            try:
                img = Image.open(logo_path).convert("RGBA")
                iw, ih = img.size
                fig_wpx = fig.get_figwidth() * dpi
                fig_hpx = fig.get_figheight() * dpi
                bbox = ax.get_position()
                ax_wpx = bbox.width * fig_wpx
                ax_hpx = bbox.height * fig_hpx
                af = ax_wpx / ax_hpx
                ai = iw / ih
                max_w, max_h = 0.55, 0.55
                w_frac = max_w
                h_frac = w_frac * (af / ai)
                if h_frac > max_h:
                    h_frac = max_h
                    w_frac = h_frac * (ai / af)
                x0 = 0.5 - w_frac / 2.0
                y0 = 0.5 - h_frac / 2.0
                extent = (x0, x0 + w_frac, y0, y0 + h_frac)
                ax.imshow(
                    img, extent=extent, transform=ax.transAxes,
                    interpolation="bilinear", alpha=WATERMARK_ALPHA,
                    zorder=1, clip_on=True
                )
            except Exception:
                pass

    # Lines + moving dots
    (value_line,)  = ax.plot([], [], lw=3,  color="#6CCFF6")
    (invest_line,) = ax.plot([], [], lw=2,  linestyle="--", color="#F7A072")
    (value_dot,)   = ax.plot([], [], "o", ms=8, color="#6CCFF6", zorder=5)
    (invest_dot,)  = ax.plot([], [], "o", ms=8, color="#F7A072", zorder=5)

    # Labels: FOLLOW + PIN (visible from frame 1)
    trans_right = blended_transform_factory(ax.transAxes, ax.transData)
    value_label_follow = ax.text(
        dates[0], 0, "", transform=ax.transData, ha="left", va="bottom",
        color="#6CCFF6", fontsize=16, weight="bold", zorder=6, clip_on=True,
        bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="#6CCFF6", lw=1), visible=False
    )
    value_label_pin = ax.text(
        LABEL_X_FRAC, 0, "", transform=trans_right, ha="right", va="bottom",
        color="#6CCFF6", fontsize=16, weight="bold", zorder=6, clip_on=False,
        bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="#6CCFF6", lw=1), visible=False
    )
    invest_label_follow = ax.text(
        dates[0], 0, "", transform=ax.transData, ha="left", va="top",
        color="#F7A072", fontsize=16, weight="bold", zorder=6, clip_on=True,
        bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="#F7A072", lw=1), visible=False
    )
    invest_label_pin = ax.text(
        LABEL_X_FRAC, 0, "", transform=trans_right, ha="right", va="top",
        color="#F7A072", fontsize=16, weight="bold", zorder=6, clip_on=False,
        bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="#F7A072", lw=1), visible=False
    )

    # Progress bar + Gain/Loss
    pbar_ax = plt.axes(PBAR_RECT); pbar_ax.set_zorder(3)
    pbar_ax.set_facecolor(BG); pbar_ax.set_xticks([]); pbar_ax.set_yticks([])
    for spine in pbar_ax.spines.values(): spine.set_visible(False)
    pbar_ax.set_xlim(0, 1); pbar_ax.set_ylim(0, 1)
    (pbar_fill,) = pbar_ax.plot([], [], lw=6, color="#6CCFF6")

    pl_ax = plt.axes(PL_RECT); pl_ax.set_zorder(3); pl_ax.axis("off")
    pl_text = pl_ax.text(
        0.5, 0.5, "", transform=pl_ax.transAxes, ha="center", va="center",
        fontsize=22, weight="bold", zorder=10,
        bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="#22303C", lw=1, alpha=0.9)
    )

    # End-card overlay
    end_ax = fig.add_axes([0, 0, 1, 1], zorder=100)
    end_ax.axis("off")
    end_bg = Rectangle((0, 0), 1, 1, transform=end_ax.transAxes, color=BG, alpha=0.0, zorder=0)
    end_ax.add_patch(end_bg)
    end_title = end_ax.text(
        0.5, 0.56, CTA_TEXT, ha="center", va="center",
        color="#FFFFFF", fontsize=28, weight="bold", alpha=0.0, zorder=1
    )
    end_handle = end_ax.text(
        0.5, 0.46, lang["handle"], ha="center", va="center",
        color="#9FB3C8", fontsize=20, alpha=0.0, zorder=1
    )

    denom_ns = xmax_ns - xmin_ns
    follow_xoff_ns = max(int(span_ns_data * 0.02), int(6 * 3600 * 1e9))

    def init():
        value_line.set_data([], [])
        invest_line.set_data([], [])
        value_dot.set_data([], [])
        invest_dot.set_data([], [])
        pbar_fill.set_data([], [])
        pl_text.set_text("")
        end_bg.set_alpha(0.0); end_title.set_alpha(0.0); end_handle.set_alpha(0.0)
        value_label_follow.set_visible(False)
        invest_label_follow.set_visible(False)
        value_label_pin.set_visible(False)
        invest_label_pin.set_visible(False)
        return (value_line, invest_line, value_dot, invest_dot,
                value_label_follow, invest_label_follow,
                value_label_pin, invest_label_pin,
                pbar_fill, pl_text, end_bg, end_title, end_handle)

    def update(frame):
        mode, payload = frame
        if mode == "chart":
            k = min(int(payload), frames_total)
            x  = dates[:k]
            yv = value_series[:k]
            yi = invested_series[:k]

            value_line.set_data(x, yv)
            invest_line.set_data(x, yi)

            val_last  = float(yv[-1]); inv_last = float(yi[-1])
            x_last_ts = x[-1]
            x_last_ns = int(dates_ns[k - 1])

            # ==== Dynamic Y-limits (from data up to k, with padding) ====
            if DYNAMIC_YLIM:
                y_min_raw = float(np.nanmin([np.nanmin(yv), np.nanmin(yi)]))
                y_max_raw = float(np.nanmax([np.nanmax(yv), np.nanmax(yi)]))
                y_range   = max(y_max_raw - y_min_raw, 1e-9)
                y_pad     = max(y_range * YPAD_FRAC, 1e-6)
                y_min_cur = y_min_raw - y_pad
                y_max_cur = y_max_raw + y_pad
                ax.set_ylim(y_min_cur, y_max_cur)
            else:
                # fallback: keep current limits
                y_min_cur, y_max_cur = ax.get_ylim()
                y_range = max(y_max_cur - y_min_cur, 1e-9)
            # ============================================================

            value_dot.set_data([x_last_ts], [val_last])
            invest_dot.set_data([x_last_ts], [inv_last])

            # Cushion for labels inside current y-lims
            cushion = max(y_range * CUSHION_FRAC, 1e-6)
            vy = float(np.clip(val_last, y_min_cur + cushion, y_max_cur - cushion))
            iy = float(np.clip(inv_last, y_min_cur + cushion, y_max_cur - cushion))

            # Separate labels a bit vertically if too close
            if abs(vy - iy) < cushion * 1.2:
                vy = min(vy + cushion, y_max_cur - cushion)
                iy = max(iy - cushion, y_min_cur + cushion)
            else:
                if val_last > inv_last:
                    vy = min(vy + cushion * 0.6, y_max_cur - cushion)
                    iy = max(iy - cushion * 0.6, y_min_cur + cushion)
                else:
                    vy = max(vy - cushion * 0.6, y_min_cur + cushion)
                    iy = min(iy + cushion * 0.6, y_max_cur - cushion)

            # ---- Dynamic x-axis growth ----
            if DYNAMIC_XLIM:
                growth_target_ns = xmin_ns + max(
                    int(MIN_START_FRAC * (xlim_right_ns - xmin_ns)),
                    int((k / frames_total) * (xlim_right_ns - xmin_ns))
                )
                right_margin_ns  = max(int(RIGHT_MARGIN_FRAC * (xlim_right_ns - xmin_ns)), RIGHT_MARGIN_MIN_NS)
                visible_right_ns = min(xlim_right_ns, max(growth_target_ns, x_last_ns + right_margin_ns))
                ax.set_xlim(xmin_dt, pd.to_datetime(visible_right_ns, unit="ns"))
                current_span_ns  = max(1, visible_right_ns - xmin_ns)
                frac             = (x_last_ns - xmin_ns) / current_span_ns
                right_edge_ns    = visible_right_ns
            else:
                frac = (x_last_ns - xmin_ns) / max(1, (xlim_right_ns - xmin_ns))
                right_edge_ns = xlim_right_ns
                current_span_ns = span_ns_full
            # --------------------------------

            # Follow x (clamped just inside current right edge)
            x_follow_ns = min(x_last_ns + follow_xoff_ns, right_edge_ns - int(current_span_ns * 0.01))
            x_follow_ts = pd.to_datetime(x_follow_ns, unit="ns")

            # Texts
            value_text   = f"{lang['label_value']}: {format_euro(val_last)}"
            invested_txt = f"{lang['label_invested']}: {format_euro(inv_last)}"
            value_label_follow.set_text(value_text)
            invest_label_follow.set_text(invested_txt)
            value_label_pin.set_text(value_text)
            invest_label_pin.set_text(invested_txt)

            if frac < FOLLOW_TO_PIN_FRAC:
                # SHOW FOLLOW labels from the very first point
                value_label_follow.set_visible(True)
                invest_label_follow.set_visible(True)
                value_label_pin.set_visible(False)
                invest_label_pin.set_visible(False)

                value_label_follow.set_position((x_follow_ts, vy))
                invest_label_follow.set_position((x_follow_ts, iy))
                value_label_follow.set_ha("left")
                invest_label_follow.set_ha("left")
                value_label_follow.set_va("bottom" if vy >= iy else "top")
                invest_label_follow.set_va("bottom" if iy >= vy else "top")
            else:
                # Switch to PIN labels at the right edge
                value_label_follow.set_visible(False)
                invest_label_follow.set_visible(False)
                value_label_pin.set_visible(True)
                invest_label_pin.set_visible(True)

                value_label_pin.set_position((LABEL_X_FRAC, vy))
                invest_label_pin.set_position((LABEL_X_FRAC, iy))
                value_label_pin.set_va("bottom" if vy >= iy else "top")
                invest_label_pin.set_va("bottom" if iy >= vy else "top")

            # Progress bar
            prog = (x_last_ns - xmin_ns) / denom_ns if denom_ns > 0 else 1.0
            pbar_fill.set_data([0.0, float(np.clip(prog, 0.0, 1.0))], [0.5, 0.5])

            # Gain/Loss
            delta = val_last - inv_last
            if delta > 0:
                pl_text.set_text(f"{gain_word}: {format_euro(delta)}"); pl_text.set_color("#16A34A")
            elif delta < 0:
                pl_text.set_text(f"{loss_word}: {format_euro(abs(delta))}"); pl_text.set_color("#EF4444")
            else:
                pl_text.set_text(f"{gain_word}: {format_euro(0)}");       pl_text.set_color("#9FB3C8")

            # hide end-card during chart frames
            end_bg.set_alpha(0.0); end_title.set_alpha(0.0); end_handle.set_alpha(0.0)

            return (value_line, invest_line, value_dot, invest_dot,
                    value_label_follow, invest_label_follow,
                    value_label_pin, invest_label_pin,
                    pbar_fill, pl_text, end_bg, end_title, end_handle)

        else:  # mode == "end" → 2-second end card
            end_bg.set_alpha(1.0)
            end_title.set_alpha(1.0)
            end_handle.set_alpha(1.0)
            return (end_bg, end_title, end_handle)

    # Full redraw needed (x/y lims change every frame)
    ani = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=False, interval=1000 / fps
    )

    # Save with ffmpeg
    if not str(outfile).lower().endswith(".mp4"):
        outfile = str(outfile) + ".mp4"
    if shutil.which("ffmpeg") is None:
        plt.close(fig)
        raise RuntimeError(lang["err_writer"])

    writer = FFMpegWriter(
        fps=fps, codec="libx264", bitrate=8000, extra_args=["-pix_fmt", "yuv420p"]
    )
    ani.save(outfile, writer=writer, dpi=dpi)
    plt.close(fig)

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Create TikTok-style stock videos: fixed or DCA.")
    parser.add_argument("--csv", required=True, help="Path to CSV with Date/Close columns.")
    parser.add_argument("--date-col", default="Date", help="Date column name (default: Date)")
    parser.add_argument("--price-col", default="Close", help="Price column name (default: Close)")
    parser.add_argument("--company", default="Company Y", help="Company name/ticker for titles")
    parser.add_argument("--mode", choices=["fixed", "dca"], required=True, help="Video mode to render")
    parser.add_argument("--start", required=True, help="Start date, e.g., 2020-01-01")
    parser.add_argument("--end", help="End date (optional, defaults to last date in CSV)")
    parser.add_argument("--amount", type=float, help="Amount: for 'fixed' the lump sum; for 'dca' the per-period contribution")
    parser.add_argument("--freq", choices=["weekly", "monthly", "yearly"], help="DCA frequency (for mode=dca)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default 30)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor (>1 skips points)")
    parser.add_argument("--outfile", help="Output mp4 path (auto if omitted)")
    parser.add_argument("--lang", choices=["en", "fr"], default="en", help="UI language for overlays (en|fr)")
    parser.add_argument(
        "--reveal-sec",
        type=float,
        default=60.0,
        help="Duration of the chart animation before the end screen (seconds)",
    )
    parser.add_argument(
        "--freeze-sec",
        type=float,
        default=0.5,
        help="Extra hold on the last frame (seconds, appended after the reveal)",
    )
    args = parser.parse_args()

    lang = get_lang(args.lang)

    prices = load_prices(args.csv, args.date_col, args.price_col)

    csv_min = prices["date"].min()
    csv_max = prices["date"].max()
    print(lang["info_csv"].format(min=csv_min.date(), max=csv_max.date()))

    # Parse requested dates
    start_dt = pd.to_datetime(args.start).to_pydatetime()
    end_dt = pd.to_datetime(args.end).to_pydatetime() if args.end else csv_max.to_pydatetime()

    # Clamp to CSV range
    if start_dt < csv_min.to_pydatetime():
        print(lang["warn_shift_start"].format(req=pd.to_datetime(args.start).date(), used=csv_min.date()))
        start_dt = csv_min.to_pydatetime()
    if end_dt > csv_max.to_pydatetime():
        print(lang["warn_shift_end"].format(req=(pd.to_datetime(args.end).date() if args.end else end_dt.date()),
                                            used=csv_max.date()))
        end_dt = csv_max.to_pydatetime()
    if start_dt > end_dt:
        raise ValueError(f"--start ({start_dt.date()}) is after --end ({end_dt.date()}).")

    # Compute series
    if args.mode == "fixed":
        if args.amount is None:
            raise ValueError("--amount is required for fixed mode")
        snapshots, vis_dates, vis_close, start_used = series_fixed_investment(prices, start_dt, args.amount)
        title = lang["title_fixed"].format(company=args.company)
        subtitle = lang["subtitle_fixed"].format(amount=format_euro(args.amount, args.lang), date=start_used.date())
        outfile = args.outfile or f"{args.company.replace(' ', '_')}_fixed_{start_used.date()}.mp4"
    else:
        if args.amount is None or not args.freq:
            raise ValueError("--amount and --freq are required for dca mode")
        snapshots, vis_dates, vis_close, start_used = series_dca(prices, start_dt, args.amount, args.freq, end_dt)
        freq_disp = display_freq(lang, args.freq)
        title = lang["title_dca"].format(company=args.company, freq_disp=freq_disp)
        subtitle = lang["subtitle_dca"].format(
            amount=format_euro(args.amount, args.lang),
            freq=display_freq_inline(lang, args.freq),
            date=start_used.date()
        )
        outfile = args.outfile or f"{args.company.replace(' ', '_')}_dca_{args.freq}_{start_used.date()}.mp4"

    if len(vis_dates) == 0:
        raise ValueError(lang["no_frames"])

    print(lang["info_render"].format(
        start=pd.to_datetime(vis_dates[0]).date(),
        end=pd.to_datetime(vis_dates[-1]).date()
    ))

    make_animation(
        vis_dates, vis_close, snapshots, title, subtitle, outfile,
        fps=args.fps, speed=args.speed, dpi=100, lang=lang,
        reveal_sec=args.reveal_sec,
        freeze_hold_sec=args.freeze_sec
    )
    print(f"[OK] Saved: {outfile}")

if __name__ == "__main__":
    main()
