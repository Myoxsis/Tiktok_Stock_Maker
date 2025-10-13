#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory



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
        "info_render": "[INFO] Rendering {n} frames from {start} to {end}",
        "err_writer": "ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`) or export a GIF by using .gif with ImageMagick.",
        "word_gain": "Gain",
        "word_loss": "Loss",
    },
    "fr": {
        "title_fixed": "{company} — Investissement initial",
        "title_dca": "{company} — DCA ({freq_disp})",
        "subtitle_fixed": "Vous avez investi {amount} le {date}",
        "subtitle_dca": "Vous investissez {amount} chaque {freq} depuis le {date}",
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
        "info_render": "[INFO] Rendu de {n} images de {start} à {end}",
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
# Rendering
# ---------------------------

def style_axes(ax):
    ax.set_facecolor("#0b0f14")
    ax.tick_params(colors="#D8E1E8", labelsize=14)
    for spine in ax.spines.values():
        spine.set_color("#22303C")
        spine.set_linewidth(1.2)

def format_euro(x: float, lang_code: str = "en"):
    # Full figure, no decimals (e.g., 1050€), no compact suffixes
    return f"{int(round(x))}€"

def make_animation(
    dates, prices, snapshots, title, subtitle, outfile,
    fps=30, speed=1.0, dpi=100, lang=None,
    min_duration_sec=5.0, freeze_hold_sec=0.5
):
    import shutil
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.animation import FFMpegWriter
    from matplotlib.ticker import FuncFormatter
    from matplotlib.transforms import blended_transform_factory

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

    # --- Frame indices (min duration + end freeze) ---
    step = max(1, int(speed))
    base_indices = list(range(1, frames_total + 1, step))
    if base_indices[-1] != frames_total:
        base_indices.append(frames_total)

    min_frames = int(min_duration_sec * fps)
    if len(base_indices) < min_frames:
        base_indices.extend([frames_total] * (min_frames - len(base_indices)))

    freeze_frames = max(0, int(round(freeze_hold_sec * fps)))
    if freeze_frames:
        tail = 0
        for idx in reversed(base_indices):
            if idx == frames_total:
                tail += 1
            else:
                break
        extra = max(0, freeze_frames - tail)
        if extra:
            base_indices.extend([frames_total] * extra)

    # --- Figure (portrait) ---
    width_px, height_px = 1080, 1920
    fig_w, fig_h = width_px / dpi, height_px / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = plt.axes([0.10, 0.22, 0.80, 0.68])
    style_axes(ax)
    fig.patch.set_facecolor("#0b0f14")

    # Titles / handle (kept high)
    fig.text(0.50, 0.955, title, ha="center", va="top", color="#FFFFFF", fontsize=28, weight="bold")
    fig.text(0.50, 0.925, subtitle, ha="center", va="top", color="#9FB3C8", fontsize=18)
    fig.text(0.03, 0.14, lang["handle"], ha="left", va="bottom", color="#5A6B7A", fontsize=16)

    # Y-lims from both series
    all_y = np.concatenate([invested_series, value_series])
    ymin = float(np.nanmin(all_y)); ymax = float(np.nanmax(all_y))
    ypad = (ymax - ymin) * 0.10 if ymax > ymin else 1.0

    # Small right x padding
    xmin_dt, xmax_dt = dates[0], dates[-1]
    xmin_ns, xmax_ns = int(dates_ns[0]), int(dates_ns[-1])
    span_ns = max(xmax_ns - xmin_ns, 1)
    right_pad_ns = max(int(span_ns * 0.06), int(12 * 3600 * 1e9))  # ≥ 12h
    xlim_right = xmax_dt + pd.to_timedelta(right_pad_ns, unit="ns")
    xlim_right_ns = int(pd.Timestamp(xlim_right).value)

    ax.set_xlim(xmin_dt, xlim_right)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_xlabel(lang["axis_date"], color="#D8E1E8", fontsize=14)
    ax.set_ylabel("€", color="#D8E1E8", fontsize=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(round(y))}€"))

    # Two lines + moving dots
    (value_line,)  = ax.plot([], [], lw=3,  color="#6CCFF6", label=lang["label_value"])
    (invest_line,) = ax.plot([], [], lw=2,  linestyle="--", color="#F7A072", label=lang["label_invested"])
    (value_dot,)   = ax.plot([], [], "o", ms=8, color="#6CCFF6", zorder=5)
    (invest_dot,)  = ax.plot([], [], "o", ms=8, color="#F7A072", zorder=5)

    # Pinned labels at right edge (inside axes)
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    label_x_frac = 0.985
    value_label  = ax.text(
        label_x_frac, ymin, "", transform=trans, ha="right", va="bottom",
        color="#6CCFF6", fontsize=16, weight="bold", zorder=6, clip_on=False,
        bbox=dict(boxstyle="round,pad=0.2", fc="#0b0f14", ec="#6CCFF6", lw=1)
    )
    invest_label = ax.text(
        label_x_frac, ymin, "", transform=trans, ha="right", va="top",
        color="#F7A072", fontsize=16, weight="bold", zorder=6, clip_on=False,
        bbox=dict(boxstyle="round,pad=0.2", fc="#0b0f14", ec="#F7A072", lw=1)
    )

    # Legend
    ax.legend(loc="upper left", frameon=True, facecolor="#0b0f14", edgecolor="#22303C", fontsize=12)

    # Progress bar (own axes)
    pbar_ax = plt.axes([0.10, 0.18, 0.80, 0.015])
    pbar_ax.set_facecolor("#0b0f14")
    pbar_ax.set_xticks([]); pbar_ax.set_yticks([])
    for spine in pbar_ax.spines.values(): spine.set_visible(False)
    pbar_ax.set_xlim(0, 1); pbar_ax.set_ylim(0, 1)
    (pbar_fill,) = pbar_ax.plot([], [], lw=6, color="#6CCFF6")

    # >>> NEW: Gain/Loss axis & text (ensures visibility with blitting)
    pl_ax = plt.axes([0.10, 0.150, 0.80, 0.022])   # just below progress bar
    pl_ax.axis("off")
    pl_text = pl_ax.text(
        0.5, 0.5, "", transform=pl_ax.transAxes, ha="center", va="center",
        fontsize=22, weight="bold", zorder=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="#0b0f14", ec="#22303C", lw=1, alpha=0.9)
    )

    denom_ns = xmax_ns - xmin_ns

    def _clipy(y):
        c = (ymax - ymin) * 0.03 if (ymax > ymin) else 1.0
        return float(np.clip(y, ymin + c, ymax - c)), c

    def init():
        value_line.set_data([], [])
        invest_line.set_data([], [])
        value_dot.set_data([], [])
        invest_dot.set_data([], [])
        value_label.set_text(""); invest_label.set_text("")
        pbar_fill.set_data([], [])
        pl_text.set_text("")
        return (value_line, invest_line, value_dot, invest_dot, value_label, invest_label, pbar_fill, pl_text)

    def update(j):
        k = min(int(j), frames_total)
        x  = dates[:k]
        yv = value_series[:k]
        yi = invested_series[:k]

        value_line.set_data(x, yv)
        invest_line.set_data(x, yi)

        val_last = float(yv[-1]); inv_last = float(yi[-1])
        x_last_ts = x[-1]
        x_last_ns = int(dates_ns[k - 1])

        growth_target_ns = xmin_ns + max(
            int(0.10 * (xlim_right_ns - xmin_ns)),
            int((k / frames_total) * (xlim_right_ns - xmin_ns)),
        )
        visible_right_ns = min(
            xlim_right_ns,
            max(growth_target_ns, x_last_ns + right_pad_ns),
        )
        ax.set_xlim(xmin_dt, pd.to_datetime(visible_right_ns, unit="ns"))

        value_dot.set_data([x_last_ts], [val_last])
        invest_dot.set_data([x_last_ts], [inv_last])

        vy, cushion = _clipy(val_last)
        iy, _       = _clipy(inv_last)

        if abs(vy - iy) < cushion * 1.2:
            vy = min(vy + cushion, ymax - cushion)
            iy = max(iy - cushion, ymin + cushion)
        else:
            if val_last > inv_last:
                vy = min(vy + cushion * 0.6, ymax - cushion)
                iy = max(iy - cushion * 0.6, ymin + cushion)
            else:
                vy = max(vy - cushion * 0.6, ymin + cushion)
                iy = min(iy + cushion * 0.6, ymax - cushion)

        value_label.set_position((label_x_frac, vy))
        value_label.set_va("bottom" if vy >= iy else "top")
        value_label.set_text(f"{lang['label_value']}: {format_euro(val_last)}")

        invest_label.set_position((label_x_frac, iy))
        invest_label.set_va("bottom" if iy >= vy else "top")
        invest_label.set_text(f"{lang['label_invested']}: {format_euro(inv_last)}")

        # Progress bar
        last_ns = int(dates_ns[k - 1])
        prog = (last_ns - xmin_ns) / denom_ns if denom_ns > 0 else 1.0
        pbar_fill.set_data([0.0, float(np.clip(prog, 0.0, 1.0))], [0.5, 0.5])

        # Gain/Loss (centered, colored)
        delta = val_last - inv_last
        if delta > 0:
            pl_text.set_text(f"{gain_word}: {format_euro(delta)}")
            pl_text.set_color("#16A34A")  # green
        elif delta < 0:
            pl_text.set_text(f"{loss_word}: {format_euro(abs(delta))}")
            pl_text.set_color("#EF4444")  # red
        else:
            pl_text.set_text(f"{gain_word}: {format_euro(0)}")
            pl_text.set_color("#9FB3C8")  # neutral

        return (value_line, invest_line, value_dot, invest_dot, value_label, invest_label, pbar_fill, pl_text)

    ani = animation.FuncAnimation(
        fig, update, frames=base_indices, init_func=init, blit=True, interval=1000 / fps
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
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor (>1 skips frames)")
    parser.add_argument("--outfile", help="Output mp4 path (auto if omitted)")
    parser.add_argument("--lang", choices=["en", "fr"], default="en", help="UI language for overlays (en|fr)")
    parser.add_argument("--freeze-sec", type=float, default=0.5,
                    help="Extra hold on the last frame (seconds). Use 0 to disable.")

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

    # Safety: ensure we have frames
    if len(vis_dates) == 0:
        raise ValueError(lang["no_frames"])

    print(lang["info_render"].format(
        n=len(vis_dates),
        start=pd.to_datetime(vis_dates[0]).date(),
        end=pd.to_datetime(vis_dates[-1]).date()
    ))

    make_animation(
        vis_dates, vis_close, snapshots, title, subtitle, outfile,
        fps=args.fps, speed=args.speed, dpi=100, lang=lang,
        min_duration_sec=5.0, freeze_hold_sec=args.freeze_sec
    )

    print(f"[OK] Saved: {outfile}")

if __name__ == "__main__":
    main()
