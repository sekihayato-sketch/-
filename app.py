import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# =====================================================
# Constants
# =====================================================
c = 299792458  # m/s

# =====================================================
# Page
# =====================================================
st.set_page_config(
    page_title="Fiber Delay Adjustment Tool (Manual / ABS target / Cut-only)",
    layout="wide"
)
st.title("Fiber Delay Adjustment Tool")
st.caption("Manual mode only. ch1/ch2 late/early is determined automatically from the SIGN of measured delay.")

# =====================================================
# Session state
# =====================================================
if "log" not in st.session_state:
    st.session_state.log = []

# =====================================================
# Inputs
# =====================================================
colA, colB, colC = st.columns(3)
with colA:
    target_ps = st.number_input("Target |delay| (ps)", value=500.0, step=1.0)
    tol_ps = st.number_input("Tolerance (ps)", value=1.0, step=0.1)
with colB:
    wavelength = st.number_input("Wavelength (nm)", value=1550.0, step=1.0)
    pm_factor = st.number_input("PM correction factor", value=1.0, format="%.6f")
with colC:
    use_temp = st.checkbox("Temperature correction", value=False)
    delta_T = st.number_input("ΔT (°C)", value=0.0, step=0.5)
    record_ok = st.checkbox("Record OK entries into history", value=False)

st.divider()

# =====================================================
# Helper functions
# =====================================================
def ng_dispersion(lambda_nm: float) -> float:
    # Simple group index approximation (placeholder)
    return 1.468 + 1e-5 * (lambda_nm - 1550.0)

def temp_corr(delay_s: float, dT: float) -> float:
    # 40 ps / km / K model -> fractional correction
    return delay_s * (1.0 + (40e-12 / 1000.0) * dT)

def corrected_delay_s(delay_s_raw: float) -> float:
    delay_s = delay_s_raw
    if use_temp:
        delay_s = temp_corr(delay_s, delta_T)
    delay_s *= pm_factor
    return delay_s

def infer_late_early_from_signed_delay(measured_ps_signed: float):
    """
    Define: measured_ps_signed = t(ch2) - t(ch1)
      + -> ch2 is later (late=ch2, early=ch1)
      - -> ch1 is later (late=ch1, early=ch2)
      0 -> arbitrary; keep ch2 as late for display consistency
    """
    if measured_ps_signed > 0:
        return "ch2", "ch1"
    elif measured_ps_signed < 0:
        return "ch1", "ch2"
    else:
        return "ch2", "ch1"

def cut_only_decision_abs_target(delay_s_raw: float, late_side: str, early_side: str):
    """
    ABS target (|delay| = target_ps), Cut-only:
      |delay| > target  -> CUT LATE side
      |delay| < target  -> CUT EARLY side
    """
    ng = ng_dispersion(wavelength)

    delay_corr_s = corrected_delay_s(delay_s_raw)
    measured_ps_raw = delay_corr_s * 1e12  # signed corrected
    measured_abs = abs(measured_ps_raw)

    diff = measured_abs - target_ps
    within = abs(diff) <= tol_ps

    result = {
        "measured_ps_raw": float(measured_ps_raw),
        "measured_ps_abs": float(measured_abs),
        "diff_ps": float(diff),
        "late_side": late_side,
        "early_side": early_side,
        "cut_side": None,
        "delta_mm": 0.0,
        "note": "",
        "delay_s_raw": float(delay_s_raw),
    }

    if within:
        result["note"] = f"Within ±{tol_ps} ps"
        return result

    if diff > 0:
        # too large -> reduce |delay| -> cut LATE side
        delta_tau_s = diff * 1e-12
        cut_side = late_side
        note = "Too long (|delay| > target) -> cut LATE (longer) side"
    else:
        # too small -> increase |delay| -> cut EARLY side
        delta_tau_s = (-diff) * 1e-12
        cut_side = early_side
        note = "Too short (|delay| < target) -> cut EARLY (shorter) side"

    delta_L_m = (delta_tau_s * c) / ng

    result["cut_side"] = cut_side
    result["delta_mm"] = float(delta_L_m * 1000.0)
    result["note"] = note
    return result

# =====================================================
# Visualization: Status band (target ± tol) + ACTION arrow
# =====================================================
def plot_status_band_with_action(measured_abs_ps: float, target_ps: float, tol_ps: float, cut_side: str | None):
    lo = target_ps - tol_ps
    hi = target_ps + tol_ps
    xmax = max(target_ps, measured_abs_ps) * 1.25
    xmax = max(xmax, target_ps + 5 * tol_ps, 10)

    if measured_abs_ps < lo:
        state = "TOO SHORT"
        color = "#2E86FF"
        action_hint = f"Action: CUT EARLY side  →  {cut_side}" if cut_side else "Action: (within tol)"
    elif measured_abs_ps > hi:
        state = "TOO LONG"
        color = "#FF4B4B"
        action_hint = f"Action: CUT LATE side   →  {cut_side}" if cut_side else "Action: (within tol)"
    else:
        state = "OK"
        color = "#2ECC71"
        action_hint = "Action: OK (no cut)"

    fig = go.Figure()

    # tolerance band
    fig.add_shape(type="rect",
                  x0=lo, x1=hi, y0=0.18, y1=0.82,
                  fillcolor="rgba(46,204,113,0.20)", line_width=0)

    # base axis line
    fig.add_trace(go.Scatter(
        x=[0, xmax],
        y=[0.5, 0.5],
        mode="lines",
        line=dict(color="rgba(120,120,120,0.6)", width=8),
        hoverinfo="skip",
        showlegend=False
    ))

    # target line
    fig.add_shape(type="line",
                  x0=target_ps, x1=target_ps, y0=0.12, y1=0.88,
                  line=dict(color="rgba(0,0,0,0.8)", width=2, dash="dash"))

    # measured marker
    fig.add_trace(go.Scatter(
        x=[measured_abs_ps],
        y=[0.5],
        mode="markers+text",
        marker=dict(size=18, color=color, line=dict(color="black", width=1)),
        text=[f"{state}<br>|delay|={measured_abs_ps:.3f} ps"],
        textposition="top center",
        showlegend=False
    ))

    # action text
    fig.add_annotation(
        x=xmax*0.98, y=0.15, xref="x", yref="y",
        text=f"<b>{action_hint}</b>",
        showarrow=False,
        xanchor="right",
        font=dict(size=14)
    )

    fig.update_layout(
        title="Status: |delay| vs target (tolerance band)",
        xaxis_title="|delay| (ps)",
        yaxis=dict(visible=False, range=[0, 1]),
        xaxis=dict(range=[0, xmax]),
        height=260,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Visualization: Timing diagram (MUST match manual signed delay)
# =====================================================
def plot_timing_diagram_signed(delay_ps_signed: float, late_side: str, early_side: str):
    """
    delay_ps_signed = t(ch2) - t(ch1)
      + -> ch2 later
      - -> ch1 later
    """
    t_ch1 = 0.0
    t_ch2 = float(delay_ps_signed)

    # axis span
    span = max(abs(t_ch2), 50.0) * 1.3
    xmin, xmax = -span, span

    fig = go.Figure()

    # Baselines
    fig.add_trace(go.Scatter(
        x=[xmin, xmax], y=[1, 1],
        mode="lines", line=dict(color="rgba(120,120,120,0.7)", width=6),
        showlegend=False, hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=[xmin, xmax], y=[0, 0],
        mode="lines", line=dict(color="rgba(120,120,120,0.7)", width=6),
        showlegend=False, hoverinfo="skip"
    ))

    # Vertical reference lines at events
    fig.add_vline(x=t_ch1, line_width=1, line_dash="dot", line_color="rgba(0,0,0,0.35)")
    fig.add_vline(x=t_ch2, line_width=1, line_dash="dot", line_color="rgba(0,0,0,0.35)")

    # Event markers
    fig.add_trace(go.Scatter(
        x=[t_ch1], y=[1],
        mode="markers+text",
        marker=dict(size=16, color="#333", line=dict(color="black", width=1)),
        text=["ch1"], textposition="top center",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[t_ch2], y=[0],
        mode="markers+text",
        marker=dict(size=16, color="#333", line=dict(color="black", width=1)),
        text=["ch2"], textposition="bottom center",
        showlegend=False
    ))

    # Horizontal delta arrow (center line)
    mid_y = 0.5
    fig.add_annotation(
        x=t_ch2, y=mid_y,
        ax=t_ch1, ay=mid_y,
        xref="x", yref="y", axref="x", ayref="y",
        text=f"Δt = {t_ch2 - t_ch1:.3f} ps (ch2 - ch1)",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.1,
        arrowwidth=2,
        arrowcolor="rgba(0,0,0,0.85)",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1
    )

    # Late/Early labels placed near the corresponding marker (NOT at right edge)
    # Put label slightly to the right of each marker with a small y offset
    fig.add_annotation(
        x=t_ch1, y=1.15, xref="x", yref="y",
        text=("Late: <b>ch1</b>" if late_side == "ch1" else "Early: <b>ch1</b>"),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.15)",
        borderwidth=1
    )
    fig.add_annotation(
        x=t_ch2, y=-0.15, xref="x", yref="y",
        text=("Late: <b>ch2</b>" if late_side == "ch2" else "Early: <b>ch2</b>"),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.15)",
        borderwidth=1
    )

    # Also show summary in a fixed place (paper coords) so it never overlaps data
    fig.add_annotation(
        x=0.99, y=0.98, xref="paper", yref="paper",
        text=f"Late (longer): <b>{late_side}</b> / Early (shorter): <b>{early_side}</b>",
        showarrow=False,
        xanchor="right",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.15)",
        borderwidth=1
    )

    fig.update_layout(
        title="Timing diagram (manual signed delay, readable)",
        xaxis_title="time (ps)",
        yaxis=dict(
            tickmode="array",
            tickvals=[1, 0],
            ticktext=["ch1", "ch2"],
            range=[-0.5, 1.5]
        ),
        xaxis=dict(range=[xmin, xmax], zeroline=True, zerolinewidth=1),
        height=340,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Manual Input (live-consistent)
# =====================================================
st.subheader("Manual input")

measured_ps_input = st.number_input("Measured delay (ps, signed)  =  t(ch2) - t(ch1)", value=500.0, step=10.0)
st.caption("Rule: + means ch2 is later. - means ch1 is later. Late/Early is auto-determined to match this sign.")

# Determine late/early from the SIGN of manual input (IMPORTANT)
late_side, early_side = infer_late_early_from_signed_delay(measured_ps_input)

# Compute using raw delay from manual input (signed)
delay_s_raw = measured_ps_input * 1e-12
res = cut_only_decision_abs_target(delay_s_raw, late_side=late_side, early_side=early_side)

# Result
st.subheader("Result")
st.write(f"Measured (corrected, signed): **{res['measured_ps_raw']:.3f} ps**")
st.write(f"|delay| used for decision: **{res['measured_ps_abs']:.3f} ps**")
st.write(f"|delay| - target = **{res['diff_ps']:.3f} ps**")
st.write(f"Late (longer): **{res['late_side']}** / Early (shorter): **{res['early_side']}**")

if res["cut_side"] is None:
    st.success(res["note"])
else:
    st.success(f"✂ CUT **{res['cut_side']}** by **{res['delta_mm']:.3f} mm**")
    st.info(res["note"])

# Visuals
colV1, colV2 = st.columns(2)
with colV1:
    plot_status_band_with_action(res["measured_ps_abs"], target_ps, tol_ps, res["cut_side"])
with colV2:
    # Use MANUAL INPUT sign for timing diagram, so it always matches what user typed
    plot_timing_diagram_signed(measured_ps_input, res["late_side"], res["early_side"])

# =====================================================
# History (button to append; avoids annoying auto-logging)
# =====================================================
st.divider()
st.subheader("History")

colH1, colH2 = st.columns([1, 1])
with colH1:
    if st.button("Append to history", type="primary"):
        if (res["cut_side"] is None) and (not record_ok):
            st.info("OK entry not recorded (enable 'Record OK entries' to save).")
        else:
            st.session_state.log.append({
                "mode": "Manual",
                "measured_ps_signed_input": float(measured_ps_input),
                "measured_ps_signed_corrected": float(res["measured_ps_raw"]),
                "measured_ps_abs": float(res["measured_ps_abs"]),
                "target_ps": float(target_ps),
                "diff_ps": float(res["diff_ps"]),
                "late_side": res["late_side"],
                "early_side": res["early_side"],
                "cut_side": res["cut_side"] if res["cut_side"] is not None else "OK",
                "delta_mm": float(res["delta_mm"]),
                "note": res["note"],
            })
            st.toast("Appended to history.", icon="✅")

with colH2:
    if st.button("Clear history", type="secondary"):
        st.session_state.log = []
        st.toast("History cleared.", icon="🧹")

if len(st.session_state.log) == 0:
    st.info("No history yet.")
else:
    log_df = pd.DataFrame(st.session_state.log)
    st.dataframe(log_df, use_container_width=True)
    st.download_button(
        "Download history as CSV",
        data=log_df.to_csv(index=False).encode("utf-8"),
        file_name="fiber_delay_history_manual.csv",
        mime="text/csv"
    )
