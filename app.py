import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RP Range Config Tool", layout="wide")

# rank name, rp_start, rp_end, fill_rgba, label_color
RANKS = [
    ("Copper",   1000, 1500, "rgba(160, 82,  45,  0.08)", "#a0522d"),
    ("Bronze",   1500, 2000, "rgba(180, 130, 50,  0.08)", "#8b6914"),
    ("Silver",   2000, 2500, "rgba(155, 155, 165, 0.08)", "#606878"),
    ("Gold",     2500, 3000, "rgba(210, 170,  0,  0.08)", "#b8860b"),
    ("Platinum", 3000, 3500, "rgba(  0, 160, 160, 0.08)", "#007070"),
    ("Emerald",  3500, 4000, "rgba( 50, 180, 100, 0.08)", "#1a7a3a"),
    ("Diamond",  4000, 4500, "rgba( 30, 144, 255, 0.08)", "#1060c0"),
    ("Champion", 4500, 5200, "rgba(180,  20,  20, 0.08)", "#b01414"),
]

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("RP Range Config Tool")

    st.subheader("RP Bounds")
    c1, c2 = st.columns(2)
    skill_min = c1.number_input("Min RP", value=1000, step=100, min_value=0)
    skill_max = c2.number_input("Max RP", value=5200, step=100, min_value=1)

    st.subheader("Relaxation Anchors")
    st.caption("Total interval width (centered) at each RP anchor. Linearly interpolated between points.")
    points_df = st.data_editor(
        pd.DataFrame({"RP": [1000, 4200, 5200], "Width": [900, 900, 1900]}),
        num_rows="dynamic",
        column_config={
            "RP":    st.column_config.NumberColumn("RP",            min_value=0, step=100, format="%d"),
            "Width": st.column_config.NumberColumn("Width (total)", min_value=0, step=100, format="%d"),
        },
        use_container_width=True,
        key="anchors",
    )

    st.divider()
    st.subheader("Display")
    show_naive      = st.checkbox("Show naive ranges",       value=False)
    show_fill       = st.checkbox("Show accepted-zone fill", value=True)
    show_rank_bands = st.checkbox("Show rank bands",         value=True)

    st.subheader("Reference Lines")
    ref_df = st.data_editor(
        pd.DataFrame({
            "RP":    [3750, 4500, 4900],
            "Label": ["Initial Top", "Champ V", "Champ I"],
            "Show":  [True, True, True],
        }),
        num_rows="dynamic",
        column_config={
            "RP":    st.column_config.NumberColumn("RP", step=100, format="%d"),
            "Label": st.column_config.TextColumn("Label"),
            "Show":  st.column_config.CheckboxColumn("Show"),
        },
        use_container_width=True,
        key="refs",
    )

    st.divider()
    st.subheader("Point Lookup")
    default_lookup = int(np.clip((skill_min + skill_max) / 2, skill_min, skill_max))
    lookup_val = st.number_input(
        "Player RP",
        value=default_lookup,
        min_value=int(skill_min),
        max_value=int(skill_max),
        step=50,
    )

# ── Validation ───────────────────────────────────────────────────────────────
pts = points_df.dropna().sort_values("RP").reset_index(drop=True)
if len(pts) < 2:
    st.error("Need at least 2 anchor points.")
    st.stop()
if skill_min >= skill_max:
    st.error("Min RP must be less than Max RP.")
    st.stop()

anchor_rp  = pts["RP"].astype(float).values
anchor_val = pts["Width"].astype(float).values

# Warn if relax grows faster than 2 RP/RP (inverse function non-monotone)
slopes = np.diff(anchor_val) / np.diff(anchor_rp)
if np.any(slopes > 2):
    st.warning("A relaxation slope > 2 may cause non-monotone inverse. Results approximate.")

# ── Core computation ─────────────────────────────────────────────────────────
N    = 4000
rp   = np.linspace(float(skill_min), float(skill_max), N)
relax = np.interp(rp, anchor_rp, anchor_val)
half  = relax / 2.0

naive_high  = np.clip(rp + half, skill_min, skill_max)
naive_low   = np.clip(rp - half, skill_min, skill_max)
lower_naive = rp - half   # f(y) = y − relax(y)/2   (monotone if slope < 2)
upper_naive = rp + half   # g(y) = y + relax(y)/2   (always monotone)

# eff_high(x) = min(naive_high(x),  largest y where f(y) ≤ x)
hi_idx   = np.clip(np.searchsorted(lower_naive, rp, side="right") - 1, 0, N - 1)
eff_high_raw = np.minimum(naive_high, rp[hi_idx])
eff_high = np.where(eff_high_raw >= skill_max, np.nan, eff_high_raw)

# eff_low(x)  = max(naive_low(x),   smallest y where g(y) ≥ x)
lo_idx  = np.clip(np.searchsorted(upper_naive, rp, side="left"), 0, N - 1)
eff_low_raw = np.maximum(naive_low, rp[lo_idx])
eff_low = np.where(eff_low_raw <= skill_min, np.nan, eff_low_raw)

# ── Kink x-values (analytical) ───────────────────────────────────────────────
# eff_high kinks at anchor_rp and at f(anchor_rp) = anchor − half
# eff_low  kinks at anchor_rp and at g(anchor_rp) = anchor + half
def xs_where_curve_hits(curve_raw, targets, rp_arr):
    """x values where curve passes through each target y (config anchor)."""
    xs = []
    for t in targets:
        # find sign changes in (curve - t)
        diff = curve_raw - t
        idx = np.where(np.diff(np.sign(diff)))[0]
        for i in idx:
            # linear interpolation to find exact crossing
            x0, x1 = rp_arr[i], rp_arr[i + 1]
            d0, d1 = diff[i], diff[i + 1]
            if d1 != d0:
                xs.append(x0 - d0 * (x1 - x0) / (d1 - d0))
    return np.array(xs)

kink_xs_high = np.unique(np.concatenate([
    anchor_rp,
    xs_where_curve_hits(eff_high_raw, anchor_rp, rp),
]))
kink_xs_low = np.unique(np.concatenate([
    anchor_rp,
    xs_where_curve_hits(eff_low_raw, anchor_rp, rp),
]))

def kink_annotations(kink_xs, curve, color, ay_sign, ax_sign=1):
    anns = []
    for x in kink_xs:
        if not (skill_min < x < skill_max):
            continue
        y_val = float(np.interp(x, rp, curve))
        if np.isnan(y_val):
            continue
        y = round(y_val)
        anns.append(dict(
            x=x, y=y,
            text=f"<b>({round(x)}, {y})</b>",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowcolor=color, ax=ax_sign * 38, ay=ay_sign * 26,
            font=dict(size=9, color=color),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)", borderwidth=0,
        ))
    return anns

# ── Lookup metrics in sidebar ─────────────────────────────────────────────────
lkp = int(np.clip(np.searchsorted(rp, float(lookup_val)), 0, N - 1))
with st.sidebar:
    mc1, mc2 = st.columns(2)
    lkp_high = eff_high_raw[lkp]
    lkp_low  = eff_low_raw[lkp]
    mc1.metric("Effective high",  f"{lkp_high:.0f}")
    mc2.metric("Effective low",   f"{lkp_low:.0f}")
    mc1.metric("Config width",    f"{relax[lkp]:.0f}")
    mc2.metric("Effective width", f"{lkp_high - lkp_low:.0f}")

# ── Build figure ──────────────────────────────────────────────────────────────
HIGH_COLOR = "#1a3a6b"
LOW_COLOR  = "#8B4513"
REF_COLORS = ["#00bcd4", "#c62828", "#ef5350", "#f57c00", "#7b1fa2", "#2e7d32"]

fig = go.Figure()

# Rank bands (vertical + horizontal, very faint)
if show_rank_bands:
    y_top = float(skill_max) * 1.05
    for name, r0, r1, fill_col, text_col in RANKS:
        x0c = max(float(r0), float(skill_min))
        x1c = min(float(r1), float(skill_max))
        if x0c >= x1c:
            continue
        # horizontal band only (color)
        fig.add_hrect(
            y0=max(float(r0), 0.0), y1=min(float(r1), y_top),
            fillcolor=fill_col, line_width=0, layer="below",
        )
        # vertical band: no fill, just label + boundary line
        fig.add_vrect(x0=x0c, x1=x1c, fillcolor="rgba(0,0,0,0)", line_width=0, layer="below")
        # rank label at bottom of vertical band, using paper y coords
        fig.add_annotation(
            x=(x0c + x1c) / 2, y=0.012,
            xref="x", yref="paper",
            text=name, showarrow=False,
            font=dict(size=9, color=text_col, family="Arial Black"),
            xanchor="center", yanchor="bottom",
        )
        # subtle boundary line
        if r0 > skill_min:
            fig.add_vline(
                x=float(r0), line_width=0.6,
                line_color=text_col, line_dash="dot", opacity=0.3,
            )

# Accepted zone fill
if show_fill:
    fill_high = np.where(np.isnan(eff_high), np.clip(eff_high_raw, skill_min, skill_max), eff_high)
    fill_low  = np.where(np.isnan(eff_low),  np.clip(eff_low_raw,  skill_min, skill_max), eff_low)
    fig.add_trace(go.Scatter(
        x=np.concatenate([rp, rp[::-1]]),
        y=np.concatenate([fill_high, fill_low[::-1]]),
        fill="toself", fillcolor="rgba(70,130,180,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Accepted zone", hoverinfo="skip",
    ))

# RP identity
fig.add_trace(go.Scatter(
    x=rp, y=rp, name="RP",
    line=dict(color="#888888", width=1.5, dash="dot"),
    hoverinfo="skip",
))

# Naive ranges (optional)
if show_naive:
    fig.add_trace(go.Scatter(x=rp, y=naive_high, name="Naive high",
        line=dict(color="#93b8d4", width=1.2, dash="dash")))
    fig.add_trace(go.Scatter(x=rp, y=naive_low, name="Naive low",
        line=dict(color="#d4a48a", width=1.2, dash="dash")))

# Effective curves
fig.add_trace(go.Scatter(
    x=rp, y=eff_high, name="Effective high",
    line=dict(color=HIGH_COLOR, width=2.5),
    hovertemplate="RP %{x:.0f} → high: %{y:.0f}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=rp, y=eff_low, name="Effective low",
    line=dict(color=LOW_COLOR, width=2.5),
    hovertemplate="RP %{x:.0f} → low: %{y:.0f}<extra></extra>",
))

# Reference lines
for i, (_, row) in enumerate(ref_df.dropna(subset=["RP"]).iterrows()):
    if not row.get("Show", True):
        continue
    color = REF_COLORS[i % len(REF_COLORS)]
    fig.add_hline(
        y=float(row["RP"]), line_dash="dash", line_color=color, line_width=1.2,
        annotation_text=str(row["Label"]),
        annotation_position="top left",
        annotation_font_color=color, annotation_font_size=10,
    )

# Kink annotations on effective curves
for ann in kink_annotations(kink_xs_high, eff_high_raw, HIGH_COLOR, ay_sign=-1, ax_sign=-1):
    fig.add_annotation(**ann)
for ann in kink_annotations(kink_xs_low, eff_low_raw, LOW_COLOR, ay_sign=1, ax_sign=1):
    fig.add_annotation(**ann)

# Min RP annotation
fig.add_annotation(
    x=rp[0], y=eff_high[0],
    text=f"{int(skill_min)} → [{eff_low_raw[0]:.0f}, {eff_high_raw[0]:.0f}]",
    showarrow=True, arrowhead=2, ax=65, ay=-28,
    font=dict(size=9, color=HIGH_COLOR),
    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", borderwidth=0,
)

# Point lookup line
fig.add_vline(
    x=float(lookup_val), line_width=1.5, line_dash="dot", line_color="#333",
    annotation_text=f"  {lookup_val} RP",
    annotation_position="top",
    annotation_font_size=9,
)

fig.update_layout(
    xaxis_title="Player RP",
    yaxis_title="Matched RP",
    xaxis=dict(
        range=[float(skill_min), float(skill_max)],
        tickmode="linear", tick0=0, dtick=500,
        showgrid=True, gridcolor="rgba(0,0,0,0.07)", gridwidth=1,
    ),
    yaxis=dict(
        range=[float(skill_min) - 250, float(skill_max) * 1.05],
        tickmode="linear", tick0=0, dtick=500,
        showgrid=True, gridcolor="rgba(0,0,0,0.07)", gridwidth=1,
    ),
    showlegend=False,
    height=1100,
    hovermode="x unified",
    template="plotly_white",
    margin=dict(l=60, r=40, t=20, b=50),
)

st.plotly_chart(fig, use_container_width=True)
