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
    page_title="Fiber Delay Adjustment Tool (ABS target / Cut-only)",
    layout="wide"
)
st.title("Fiber Delay Adjustment Tool - ABS Target / Cut-only")
st.caption("Decision uses |delay| vs target. Adjustment is CUT-only (no add).")

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
    use_filter = st.checkbox("Enable filtering (moving average)", value=True)

st.divider()
mode = st.radio("Mode", ["Manual input", "CSV"], horizontal=True)

# =====================================================
# Session state
# =====================================================
if "log" not in st.session_state:
    st.session_state.log = []

# =====================================================
# Helper functions
# =====================================================
def ng_dispersion(lambda_nm: float) -> float:
    # Simple group index approximation (placeholder model)
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

def apply_filter(signal: np.ndarray, window: int = 7) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return signal
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(signal, kernel, mode="same")

def safe_standardize(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    if (s == 0) or (not np.isfinite(s)):
        return None
    return (x - np.mean(x)) / s

def find_roi_indices(signal: np.ndarray, threshold: float = 0.3, margin: int = 80):
    signal = np.asarray(signal, dtype=float)
    abs_sig = np.abs(signal)
    mx = np.max(abs_sig) if len(abs_sig) else 0
    if mx == 0:
        return 0, len(signal)
    norm = abs_sig / mx
    idx = np.where(norm > threshold)[0]
    if len(idx) == 0:
        return 0, len(signal)
    start = max(int(idx[0]) - margin, 0)
    end = min(int(idx[-1]) + margin + 1, len(signal))
    return start, end

def refine_peak_parabolic(y: np.ndarray, idx: int) -> float:
    # Parabolic interpolation around peak for sub-sample peak position
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    y1, y2, y3 = y[idx - 1], y[idx], y[idx + 1]
    denom = (y1 - 2.0 * y2 + y3)
    if denom == 0:
        return float(idx)
    return float(idx) + 0.5 * (y1 - y3) / denom

# =====================================================
# Decision logic (ABS target / Cut-only)
# =====================================================
def cut_only_decision_abs_target(delay_s_raw: float, late_side: str):
    """
    ABS target (|delay| = target_ps), Cut-only:
      |delay| > target  -> CUT LATE side (reduce |delay|)
      |delay| < target  -> CUT EARLY side (increase |delay| by cutting early side)
    NOTE:
      You provide late_side as the longer side.
    """
    ng = ng_dispersion(wavelength)

    delay_corr_s = corrected_delay_s(delay_s_raw)
    measured_ps_raw = delay_corr_s * 1e12
    measured_abs = abs(measured_ps_raw)

    diff = measured_abs - target_ps
    within = abs(diff) <= tol_ps

    early_side = "ch1" if late_side == "ch2" else "ch2"

    result = {
        "measured_ps_raw": measured_ps_raw,
        "measured_ps_abs": measured_abs,
        "diff_ps": diff,
        "late_side": late_side,
        "early_side": early_side,
        "cut_side": None,
        "delta_mm": 0.0,
        "note": ""
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
        # too small -> increase |delay| (by cutting early/shorter side)
        delta_tau_s = (-diff) * 1e-12
        cut_side = early_side
        note = "Too short (|delay| < target) -> cut EARLY (shorter) side"

    delta_L_m = (delta_tau_s * c) / ng

    result["cut_side"] = cut_side
    result["delta_mm"] = float(delta_L_m * 1000.0)
    result["note"] = note
    return result

# =====================================================
# Plot helpers (Plotly)
# =====================================================
def plot_waveforms(t_s, ch1, ch2, title="Waveforms", time_unit="s"):
    # time axis in chosen unit for display
    scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9, "ps": 1e12}[time_unit]
    x = np.asarray(t_s) * scale

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ch1, mode="lines", name="ch1"))
    fig.add_trace(go.Scatter(x=x, y=ch2, mode="lines", name="ch2"))
    fig.update_layout(
        title=title,
        xaxis_title=f"time ({time_unit})",
        yaxis_title="amplitude",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation(corr, lag_samples, title="Cross-correlation"):
    corr = np.asarray(corr, dtype=float)
    lags = np.arange(-(len(corr)//2), len(corr) - (len(corr)//2))  # fallback
    # Better lags: when corr = correlate(x,y,'full'), length=2N-1, lags=-(N-1)..(N-1)
    # We'll infer N from corr length (odd), N=(len(corr)+1)/2
    if len(corr) % 2 == 1:
        N = (len(corr) + 1) // 2
        lags = np.arange(-(N - 1), (N - 1) + 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lags, y=corr, mode="lines", name="xcorr"))
    fig.add_vline(x=lag_samples, line_dash="dash", line_color="red")
    fig.update_layout(
        title=title,
        xaxis_title="lag (samples)",
        yaxis_title="correlation",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# Manual Mode
# =====================================================
if mode == "Manual input":
    st.subheader("Manual input")

    col1, col2 = st.columns(2)
    with col1:
        measured_ps_input = st.number_input("Measured delay (ps, signed)", value=500.0, step=1.0)
    with col2:
        late_side_manual = st.radio("Late side (longer)", ["ch1", "ch2"], horizontal=True)

    delay_s_raw = measured_ps_input * 1e-12
    res = cut_only_decision_abs_target(delay_s_raw, late_side_manual)

    st.subheader("Result")
    st.write(f"Measured (corrected, signed): **{res['measured_ps_raw']:.3f} ps**")
    st.write(f"|delay| used for decision: **{res['measured_ps_abs']:.3f} ps**")
    st.write(f"|delay| - target = **{res['diff_ps']:.3f} ps**")
    st.write(f"Late (longer): **{res['late_side']}** / Early (shorter): **{res['early_side']}**")

    if res["cut_side"] is None:
        st.success(res["note"])
    else:
        st.success(f"CUT **{res['cut_side']}** by **{res['delta_mm']:.3f} mm**")
        st.info(res["note"])

    # Log
    st.session_state.log.append({
        "mode": "Manual",
        "measured_ps_signed": float(res["measured_ps_raw"]),
        "measured_ps_abs": float(res["measured_ps_abs"]),
        "target_ps": float(target_ps),
        "diff_ps": float(res["diff_ps"]),
        "late_side": res["late_side"],
        "cut_side": res["cut_side"] if res["cut_side"] is not None else "OK",
        "delta_mm": float(res["delta_mm"]),
        "note": res["note"],
    })

# =====================================================
# CSV Mode
# =====================================================
if mode == "CSV":
    st.subheader("CSV mode")
    st.write("Upload CSV with 3 columns: **time, ch1, ch2** (header may exist).")

    file = st.file_uploader("Upload CSV (time, ch1, ch2)", type=["csv"])

    colx, coly, colz = st.columns(3)
    with colx:
        threshold = st.slider("Auto-window threshold", 0.05, 0.90, 0.30, 0.05)
        margin = st.number_input("ROI margin (samples)", value=80, min_value=0, step=10)
        filter_window = st.number_input("Filter window (samples)", value=7, min_value=1, step=2)
    with coly:
        time_unit = st.selectbox("Time unit", ["s", "ms", "us", "ns", "ps"], index=3)
        time_scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}[time_unit]
        late_rule = st.selectbox("Late side rule from lag sign", ["lag>0 => ch2 late", "lag>0 => ch1 late"], index=0)
    with colz:
        show_plots = st.checkbox("Show plots", value=True)
        show_roi_only = st.checkbox("Plot ROI only", value=True)

    if file is not None and st.button("Quick Measure", type="primary"):
        try:
            df = pd.read_csv(file)

            if df.shape[1] < 3:
                st.error("CSV must have at least 3 columns: time, ch1, ch2.")
                st.stop()

            # Use first 3 columns regardless of header names
            t = df.iloc[:, 0].to_numpy(dtype=float) * time_scale
            ch1 = df.iloc[:, 1].to_numpy(dtype=float)
            ch2 = df.iloc[:, 2].to_numpy(dtype=float)

            if use_filter:
                ch1 = apply_filter(ch1, window=filter_window)
                ch2 = apply_filter(ch2, window=filter_window)

            # Determine ROI
            s1, e1 = find_roi_indices(ch1, threshold=threshold, margin=int(margin))
            s2, e2 = find_roi_indices(ch2, threshold=threshold, margin=int(margin))
            start, end = min(s1, s2), max(e1, e2)

            if show_roi_only:
                t_roi, ch1_roi, ch2_roi = t[start:end], ch1[start:end], ch2[start:end]
            else:
                t_roi, ch1_roi, ch2_roi = t, ch1, ch2

            # Safety checks
            if len(t_roi) < 2:
                st.error("ROI too short (<2 samples). Lower threshold or increase margin.")
                st.stop()

            dt = float(t_roi[1] - t_roi[0])
            if not np.isfinite(dt) or dt <= 0:
                st.error("Invalid time axis (dt <= 0). Check time column and unit.")
                st.stop()

            ch1n = safe_standardize(ch1_roi)
            ch2n = safe_standardize(ch2_roi)
            if ch1n is None or ch2n is None:
                st.error("Signal std is zero/invalid -> cannot compute correlation. Check data.")
                st.stop()

            # Cross-correlation
            corr = np.correlate(ch1n, ch2n, mode="full")
            idx = int(np.argmax(corr))
            idx_ref = refine_peak_parabolic(corr, idx)
            lag = float(idx_ref - (len(ch1n) - 1))  # samples (can be fractional)
            delay_s_raw = lag * dt

            # late side decision from lag sign
            if late_rule == "lag>0 => ch2 late":
                late_side = "ch2" if lag > 0 else "ch1"
            else:
                late_side = "ch1" if lag > 0 else "ch2"

            res = cut_only_decision_abs_target(delay_s_raw, late_side)

            # Results
            st.subheader("Result")
            st.write(f"Lag (samples): **{lag:.3f}**")
            st.write(f"dt: **{dt:.3e} s**  -> raw delay: **{delay_s_raw*1e12:.3f} ps**")
            st.write(f"Measured (corrected, signed): **{res['measured_ps_raw']:.3f} ps**")
            st.write(f"|delay|: **{res['measured_ps_abs']:.3f} ps**")
            st.write(f"|delay| - target = **{res['diff_ps']:.3f} ps**")
            st.write(f"Late (longer): **{res['late_side']}** / Early (shorter): **{res['early_side']}**")

            if res["cut_side"] is None:
                st.success(res["note"])
            else:
                st.success(f"CUT **{res['cut_side']}** by **{res['delta_mm']:.3f} mm**")
                st.info(res["note"])

            # Plots
            if show_plots:
                st.subheader("Plots")
                plot_waveforms(
                    t_roi, ch1_roi, ch2_roi,
                    title=("Waveforms (ROI)" if show_roi_only else "Waveforms (Full)"),
                    time_unit=time_unit
                )
                plot_correlation(corr, lag_samples=lag, title="Cross-correlation (peak = estimated lag)")

            # Log
            st.session_state.log.append({
                "mode": "CSV",
                "lag_samples": float(lag),
                "dt_s": float(dt),
                "measured_ps_signed": float(res["measured_ps_raw"]),
                "measured_ps_abs": float(res["measured_ps_abs"]),
                "target_ps": float(target_ps),
                "diff_ps": float(res["diff_ps"]),
                "late_side": res["late_side"],
                "cut_side": res["cut_side"] if res["cut_side"] is not None else "OK",
                "delta_mm": float(res["delta_mm"]),
                "note": res["note"],
                "roi_start": int(start),
                "roi_end": int(end),
                "time_unit": time_unit,
            })

        except Exception as e:
            st.exception(e)

# =====================================================
# History
# =====================================================
st.divider()
st.subheader("History")

if len(st.session_state.log) == 0:
    st.info("No history yet.")
else:
    log_df = pd.DataFrame(st.session_state.log)
    st.dataframe(log_df, use_container_width=True)

    # Download
    csv_bytes = log_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download history as CSV",
        data=csv_bytes,
        file_name="fiber_delay_history.csv",
        mime="text/csv"
    )
