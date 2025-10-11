# TikTok Stock Maker

Create TikTok-ready stock performance animations from CSV price histories.

## Command-line usage

Generate a single lump-sum investment video:

```bash
python3 tiktok_stock_animator_5.py --csv data/TSLA.csv --company "Tesla" --mode fixed \
    --start 2017-01-01 --amount 5000 --fps 30 --lang fr
```

Generate a dollar-cost-averaging (DCA) video:

```bash
python3 tiktok_stock_animator_5.py --csv data/NVDA.csv --company "Nvidia" --mode dca \
    --freq monthly --start 2018-01-01 --amount 50 --fps 30 --lang fr
```

## Visual interface

A Streamlit dashboard is included to configure the animation visually, launch the
render, and preview the resulting MP4 inside the browser.

1. Install the Python dependencies (Streamlit plus the existing requirements).
   ```bash
   pip install -r requirements.txt  # if available
   pip install streamlit
   ```
2. Start the app:
   ```bash
   streamlit run app.py
   ```
3. Choose your CSV (use the bundled sample or upload one), configure the
   investment options, and click **Generate video** to render and preview the
   result.
