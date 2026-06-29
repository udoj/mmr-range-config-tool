# RP Range Config Tool

Visualizes effective matchmaking ranges based on a piecewise-linear RP relaxation config.

## What it does

Given a list of `(RP, width)` anchors, the tool computes the **effective** match range for any player RP using mutual-acceptance logic: two players can match only if each falls within the other's range. The result is tighter than the naive half-width interval, especially at the extremes.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## How to use

- **Relaxation Anchors** — edit the `(RP, Width)` table to define your config. Rows are linearly interpolated. Add/remove rows freely.
- **RP Bounds** — set the min/max RP clamps (e.g. 1000–5200).
- **Reference Lines** — add horizontal markers (rank thresholds, etc.).
- **Point Lookup** — enter any RP value to read its effective high/low/width.
- The graph shows **Effective high** (blue) and **Effective low** (brown) curves, with kink annotations at each config anchor.
