"""Streamlit application for generating TikTok-style stock videos.

This interface wraps the functions from ``tiktok_stock_animator_5`` and lets
users choose the CSV source, investment mode, and rendering options before
rendering the MP4 animation. When a video is created the preview is displayed
inline in the app.
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import re

import pandas as pd
import streamlit as st

from tiktok_stock_animator_5 import (
    display_freq,
    display_freq_inline,
    format_euro,
    get_lang,
    load_prices,
    make_animation,
    series_dca,
    series_fixed_investment,
)

DATA_DIR = Path("data")
DEFAULT_CSV = Path("stocks.csv")


@st.cache_data(show_spinner=False)
def _load_prices(csv_path: str, date_col: str, price_col: str) -> pd.DataFrame:
    """Wrapper around :func:`load_prices` with Streamlit caching."""
    return load_prices(csv_path, date_col=date_col, price_col=price_col)


def _resolve_csv_path(option: str, uploaded_file) -> Optional[str]:
    """Return the path to the CSV chosen by the user."""
    if option == "Sample dataset":
        if DEFAULT_CSV.exists():
            return str(DEFAULT_CSV)
        sample_files = sorted(DATA_DIR.glob("*.csv"))
        if sample_files:
            return str(sample_files[0])
        st.warning("No sample CSV file was found in the repository.")
        return None

    if uploaded_file is None:
        st.info("Upload a CSV file with Date and Close columns to get started.")
        return None

    suffix = Path(uploaded_file.name).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def _format_currency(value: float, lang_code: str) -> str:
    try:
        return format_euro(value, lang_code)
    except Exception:
        return f"{value:,.2f}"


def _slugify_name(name: str) -> str:
    """Return a filesystem-safe slug for ``name`` suitable for downloads."""
    if not name:
        return "video"
    slug = re.sub(r"\s+", "_", name.strip())
    slug = re.sub(r"[^A-Za-z0-9_\-]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_-")
    return slug or "video"


def _generate_video(
    prices: pd.DataFrame,
    company: str,
    mode: str,
    start_dt: datetime,
    end_dt: Optional[datetime],
    amount: float,
    freq: Optional[str],
    fps: int,
    speed: float,
    freeze_sec: float,
    lang_code: str,
    reveal_duration: float,
) -> Tuple[Path, pd.Timestamp]:
    lang = get_lang(lang_code)

    data = prices.copy()
    if end_dt is not None:
        data = data[data["date"] <= pd.Timestamp(end_dt)]
    if data.empty:
        raise ValueError("No price data available inside the selected date range.")

    if mode == "fixed":
        snapshots, vis_dates, vis_close, start_used = series_fixed_investment(
            data, start_dt, amount
        )
        title = lang["title_fixed"].format(company=company)
        subtitle = lang["subtitle_fixed"].format(
            amount=_format_currency(amount, lang_code),
            date=start_used.date(),
        )
        outfile = f"{company.replace(' ', '_')}_fixed_{start_used.date()}"
    else:
        if freq is None:
            raise ValueError("Choose a dollar-cost averaging frequency.")
        snapshots, vis_dates, vis_close, start_used = series_dca(
            data, start_dt, amount, freq, end_dt
        )
        freq_disp = display_freq(lang, freq)
        title = lang["title_dca"].format(company=company, freq_disp=freq_disp)
        subtitle = lang["subtitle_dca"].format(
            amount=_format_currency(amount, lang_code),
            freq=display_freq_inline(lang, freq),
            date=start_used.date(),
        )
        outfile = f"{company.replace(' ', '_')}_dca_{freq}_{start_used.date()}"

    if len(vis_dates) == 0:
        raise ValueError(lang["no_frames"])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        output_path = Path(tmpfile.name)

    make_animation(
        vis_dates,
        vis_close,
        snapshots,
        title,
        subtitle,
        output_path,
        fps=fps,
        speed=speed,
        dpi=100,
        lang=lang,
        reveal_sec=reveal_duration,
        freeze_hold_sec=freeze_sec,
    )
    return output_path, start_used


def main() -> None:
    st.set_page_config(page_title="TikTok Stock Video Maker", layout="wide")
    st.title("🎬 TikTok Stock Video Maker")
    st.write(
        "Build eye-catching TikTok-ready stock performance videos by choosing your CSV, "
        "investment mode, and rendering parameters."
    )
    st.markdown(
        """
        <style>
            [data-testid="stVideo"] video {
                width: min(420px, 100%) !important;
                border-radius: 12px;
                box-shadow: 0 12px 24px rgba(0, 0, 0, 0.25);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Data source")
    csv_choice = st.sidebar.radio(
        "Select data source",
        options=["Sample dataset", "Upload CSV"],
        help="Use the bundled CSV or upload your own file with Date and Close columns.",
    )
    uploaded_csv = None
    if csv_choice == "Upload CSV":
        uploaded_csv = st.sidebar.file_uploader(
            "Upload CSV", type=["csv"], accept_multiple_files=False
        )

    date_col = st.sidebar.text_input("Date column", value="Date")
    price_col = st.sidebar.text_input("Price column", value="Close")

    csv_path = _resolve_csv_path(csv_choice, uploaded_csv)

    prices: Optional[pd.DataFrame] = None
    if csv_path:
        try:
            prices = _load_prices(csv_path, date_col, price_col)
            st.success(f"Loaded {Path(csv_path).name} with {len(prices):,} rows.")
        except Exception as exc:
            st.error(f"Failed to load CSV: {exc}")

    st.sidebar.header("Video settings")

    if prices is not None and not prices.empty:
        min_date = prices["date"].min().date()
        max_date = prices["date"].max().date()
    else:
        min_date = datetime.today().date()
    max_date = datetime.today().date()

    company = st.sidebar.text_input("Company/Ticker", value="Company Y")
    mode = st.sidebar.selectbox(
        "Mode",
        options=["fixed", "dca", "compare"],
        format_func=lambda x: x.upper(),
    )

    investment_mode = mode
    company_compare = ""
    if mode == "compare":
        investment_mode = st.sidebar.selectbox(
            "Comparison investment type",
            options=["fixed", "dca"],
            format_func=lambda x: x.upper(),
        )
        company_compare = st.sidebar.text_input(
            "Second Company/Ticker",
            value="Company Z",
        )

    start_date = st.sidebar.date_input(
        "Start date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
    )
    end_date = st.sidebar.date_input(
        "End date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )
    amount = st.sidebar.number_input(
        "Amount (€)",
        min_value=0.0,
        value=500.0,
        step=50.0,
    )
    freq = None
    if investment_mode == "dca":
        freq = st.sidebar.selectbox(
            "DCA frequency",
            options=["weekly", "monthly", "yearly"],
            format_func=lambda x: x.capitalize(),
        )

    comparison_prices: Optional[pd.DataFrame] = None
    comparison_csv_path: Optional[str] = None
    if mode == "compare":
        st.sidebar.header("Second ticker data")
        csv_choice_cmp = st.sidebar.radio(
            "Select data source for ticker 2",
            options=["Sample dataset", "Upload CSV"],
            key="compare_csv_choice",
        )
        uploaded_csv_cmp = None
        if csv_choice_cmp == "Upload CSV":
            uploaded_csv_cmp = st.sidebar.file_uploader(
                "Upload CSV for ticker 2",
                type=["csv"],
                accept_multiple_files=False,
                key="compare_csv_uploader",
            )
        comparison_csv_path = _resolve_csv_path(csv_choice_cmp, uploaded_csv_cmp)
        if comparison_csv_path:
            try:
                comparison_prices = _load_prices(
                    comparison_csv_path, date_col, price_col
                )
                st.sidebar.success(
                    f"Loaded {Path(comparison_csv_path).name} with {len(comparison_prices):,} rows."
                )
            except Exception as exc:
                comparison_prices = None
                st.sidebar.error(f"Failed to load comparison CSV: {exc}")
    fps = st.sidebar.slider("FPS", min_value=15, max_value=60, value=30)
    speed = st.sidebar.slider(
        "Playback speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1
    )
    freeze_sec = st.sidebar.slider(
        "Hold last chart frame (s)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Pause on the final chart view before showing the end screen.",
    )
    duration_options = {
        "20 seconds": 20.0,
        "30 seconds": 30.0,
        "60 seconds": 60.0,
    }
    duration_choice = st.sidebar.selectbox(
        "Video duration (excludes end screen)",
        options=list(duration_options.keys()),
        index=2,
        help="Select how long the chart animation should last before the end screen.",
    )
    reveal_duration = duration_options[duration_choice]
    lang_code = st.sidebar.selectbox("Language", options=["en", "fr"], index=0)

    generate_label = "Generate videos" if mode == "compare" else "Generate video"
    generate_btn = st.sidebar.button(generate_label, type="primary")

    if prices is not None and not prices.empty:
        primary_label = company or "Ticker 1"
        st.subheader(f"Price preview — {primary_label}")
        st.line_chart(
            prices.set_index("date")["close"], height=300, use_container_width=True
        )

    if mode == "compare" and comparison_prices is not None and not comparison_prices.empty:
        secondary_label = company_compare or "Ticker 2"
        st.subheader(f"Price preview — {secondary_label}")
        st.line_chart(
            comparison_prices.set_index("date")["close"],
            height=300,
            use_container_width=True,
        )

    if prices is not None and generate_btn:
        if start_date > end_date:
            st.error("Start date must be before end date.")
        else:
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.min.time()) if end_date else None

            if mode == "compare" and (
                comparison_prices is None or comparison_prices.empty
            ):
                st.error("Load price data for the second ticker to run a comparison.")
            else:
                try:
                    if mode == "compare":
                        comparison_name = company_compare or "Ticker 2"
                        with st.spinner("Rendering comparison videos..."):
                            video_primary, start_used_primary = _generate_video(
                                prices,
                                company or "Ticker 1",
                                investment_mode,
                                start_dt,
                                end_dt,
                                amount,
                                freq,
                                fps,
                                speed,
                                freeze_sec,
                                lang_code,
                                reveal_duration,
                            )
                            assert comparison_prices is not None
                            video_secondary, start_used_secondary = _generate_video(
                                comparison_prices,
                                comparison_name,
                                investment_mode,
                                start_dt,
                                end_dt,
                                amount,
                                freq,
                                fps,
                                speed,
                                freeze_sec,
                                lang_code,
                                reveal_duration,
                            )

                        st.success("Videos created for both tickers.")
                        col_primary, col_secondary = st.columns(2)
                        ticker_entries = [
                            (col_primary, company or "Ticker 1", video_primary, start_used_primary),
                            (col_secondary, comparison_name, video_secondary, start_used_secondary),
                        ]
                        for column, name, path, start_used_val in ticker_entries:
                            download_filename = (
                                f"{_slugify_name(name)}_{investment_mode}_{start_used_val.date().isoformat()}.mp4"
                            )
                            with column:
                                st.markdown(f"**{name}**")
                                st.caption(
                                    f"Start date used: {start_used_val.date().isoformat()}"
                                )
                                st.video(str(path))
                                st.download_button(
                                    f"Download {name}",
                                    data=path.read_bytes(),
                                    file_name=download_filename,
                                    mime="video/mp4",
                                )
                    else:
                        with st.spinner("Rendering video..."):
                            video_path, start_used = _generate_video(
                                prices,
                                company,
                                investment_mode,
                                start_dt,
                                end_dt,
                                amount,
                                freq,
                                fps,
                                speed,
                                freeze_sec,
                                lang_code,
                                reveal_duration,
                            )
                        download_filename = (
                            f"{_slugify_name(company)}_{investment_mode}_{start_used.date().isoformat()}.mp4"
                        )
                        st.success(f"Video created: {download_filename}")
                        st.video(str(video_path))
                        with video_path.open("rb") as video_file:
                            st.download_button(
                                "Download video",
                                data=video_file.read(),
                                file_name=download_filename,
                                mime="video/mp4",
                            )
                except Exception as exc:
                    st.error(f"Failed to render video: {exc}")


if __name__ == "__main__":
    main()
