import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

c = 299792458  # m/s

# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="Fiber Delay Adjustment Tool (Cut-Only)", layout="wide")
st.title("Fiber Delay Adjustment Tool - CUT-ONLY FINAL")

# =========================
# Inputs
# =========================
colA, colB, colC = st.columns(3)
with colA:
    target_ps = st.number_input("Target delay |Δt| (ps)", value=500.0)
    tol_ps = st.number_input("Tolerance (ps)", value=1.0)
with colB:
    wavelength = st.number_input("Wavelength (nm)", value=1550.0)
    pm_factor = st.number_input("PM correction factor", value=1.0)
with colC:
    use_temp = st.checkbox("Temperature correction")
    delta_T = st.number_input("ΔT (°C)", value=0.0)
    use_filter = st.checkbox("Enable filtering", value=True)

st.divider()

mode = st.radio("Mode", ["Manual input", "CSV"], horizontal=True)

# =========================
# State
# =========================
if "log" not in st.session_state:
    st.session_state.log = []

# =========================
# Helpers
# =========================
def ng_dispersion(lambda_nm: float) -> float:
    # Simple approximation (replace if you have a better model)
    return 1.468 + 1e-5 * (lambda_nm - 1550.0)

def temp_corr(delay_s: float, dT: float) -> float:
    # 40 ps/km/K proportional model
    return delay_s * (1.0 + (40e-12 / 1000.0) * dT)

def apply_filter(signal: np.ndarray, window: int = 7) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")

def safe_standardize(x: np.ndarray):
    x = np.asarray(x)
    s = np.std(x)
    if s == 0 or not np.isfinite(s):
        return None
    return (x - np.mean(x)) / s

def find_roi_indices(signal: np.ndarray, threshold: float = 0.3, margin: int = 80):
    if signal is None or len(signal) == 0:
        return 0, 0
    abs_sig = np.abs(signal)
    mx = np.max(abs_sig)
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
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    y1, y2, y3 = y[idx - 1], y[idx], y[idx + 1]
    denom = (y1 - 2.0 * y2 + y3)
    if denom == 0:
        return float(idx)
    return float(idx) + 0.5 * (y1 - y3) / denom

def compute_delay_from_csv(df: pd.DataFrame, threshold: float, time_scale: float, invert_late_rule: bool):
    """
    Returns:
      t, ch1n, ch2n, corr, lags, lag_float, delay_s_raw, confidence, late_side
    """
    if df.shape[1] < 3:
        raise ValueError("CSV must have at least 3 columns: time, ch1, ch2")

    t = df.iloc[:, 0].to_numpy(dtype=float) * time_scale
    ch1 = df.iloc[:, 1].to_numpy(dtype=float)
    ch2 = df.iloc[:, 2].to_numpy(dtype=float)

    # Remove NaNs row-wise
    mask = np.isfinite(t) & np.isfinite(ch1) & np.isfinite(ch2)
    t, ch1, ch2 = t[mask], ch1[mask], ch2[mask]
    if len(t) < 3:
        raise ValueError("Not enough valid samples after removing NaNs.")

    # Time must increase
    if not np.all(np.diff(t) > 0):
        raise ValueError("Time column must be strictly increasing (after unit scaling).")

    # Optional filter
    if use_filter:
        ch1 = apply_filter(ch1, window=7)
        ch2 = apply_filter(ch2, window=7)

    # ROI union to keep t aligned
    s1, e1 = find_roi_indices(ch1, threshold=threshold, margin=80)
    s2, e2 = find_roi_indices(ch2, threshold=threshold, margin=80)
    start = min(s1, s2)
    end = max(e1, e2)

    t = t[start:end]
    ch1 = ch1[start:end]
    ch2 = ch2[start:end]

    if len(t) < 3:
        raise ValueError("ROI too small. Lower threshold or check waveform.")

    dt = t[1] - t[0]
    if dt <= 0:
        raise ValueError("Invalid dt from time vector.")

    # Standardize
    ch1n = safe_standardize(ch1)
    ch2n = safe_standardize(ch2)
    if ch1n is None or ch2n is None:
        raise ValueError("Signal std is zero (flat). Check CSV columns/scaling.")

    # Cross-correlation
    corr = np.correlate(ch1n, ch2n, mode="full")
    idx = int(np.argmax(corr))
    idx_ref = refine_peak_parabolic(corr, idx)

    lag_float = idx_ref - (len(ch1n) - 1)  # lag in samples
    delay_s_raw = lag_float * dt

    # Confidence
    peak = float(np.max(corr))
    noise = float(np.mean(np.abs(corr))) + 1e-12
    confidence = peak / noise

    lags = np.arange(-len(ch1n) + 1, len(ch1n), dtype=float)

    # Determine which side is "late" (needs to be shortened)
    # Default rule: lag_float > 0 => ch2 is late; else ch1 is late
    late_side = "ch2" if lag_float > 0 else "ch1"
    if invert_late_rule:
        late_side = "ch1" if late_side == "ch2" else "ch2"

    return t, ch1n, ch2n, corr, lags, lag_float, delay_s_raw, confidence, late_side

def corrected_delay_s(delay_s_raw: float) -> float:
    delay_s = delay_s_raw
    if use_temp:
        delay_s = temp_corr(delay_s, delta_T)
    delay_s *= pm_factor
    return delay_s

def cut_only_decision(delay_s_raw: float, late_side: str):
    """
    Policy:
      - Compare by |delay|
      - If |delay| > target+tol => CUT late side by ΔL = (|delay|-target) * c / ng
      - If |delay| within tol => OK
      - If |delay| < target-tol => cannot fix by cutting => WARNING
    """
    ng = ng_dispersion(wavelength)

    delay_corr = corrected_delay_s(delay_s_raw)
    measured_ps_raw = delay_corr * 1e12
    measured_ps_abs = abs(measured_ps_raw)

    diff_ps = measured_ps_abs - target_ps  # positive means "too long"
    within = abs(diff_ps) <= tol_ps

    result = {
        "measured_ps_raw": measured_ps_raw,
        "measured_ps_abs": measured_ps_abs,
        "diff_ps_abs_minus_target": diff_ps,
        "late_side": late_side,
        "action": None,
        "delta_mm": 0.0,
        "note": ""
    }

    if within:
        result["action"] = "OK"
        result["note"] = f"Within ±{tol_ps} ps"
        return result

    if diff_ps > tol_ps:
        # Need LESS delay => CUT the late side
        delta_tau_s = diff_ps * 1e-12
        delta_L_m = (delta_tau_s * c) / ng
        result["action"] = "CUT"
        result["delta_mm"] = delta_L_m * 1000.0
        result["note"] = "Too long -> cut late side"
        return result

    # diff_ps < -tol_ps => too short, cannot fix by cutting
    result["action"] = "TOO_SHORT"
    result["note"] = "Too short -> cannot fix by cutting. Need add fiber or rebuild baseline."
    return result

# =========================
# Manual Mode
# =========================
if mode == "Manual input":
    st.subheader("Manual input (Cut-only logic)")
    col1, col2 = st.columns(2)
    with col1:
        measured_ps_input = st.number_input("Measured delay (ps) (can be ±)", value=500.0)
    with col2:
        late_side_manual = st.radio("Late side (to shorten)", ["ch1", "ch2"], horizontal=True)

    delay_s_raw = measured_ps_input * 1e-12
    res = cut_only_decision(delay_s_raw, late_side_manual)

    st.subheader("Result")
    st.write(f"Measured (raw, corrected): {res['measured_ps_raw']:.3f} ps")
    st.write(f"Measured |.| used for decision: {res['measured_ps_abs']:.3f} ps")
    st.write(f"Late side: {res['late_side']}")
    st.write(f"|delay|-target = {res['diff_ps_abs_minus_target']:.3f} ps")

    if res["action"] == "OK":
        st.success(res["note"])
    elif res["action"] == "CUT":
        st.success(f"CUT {res['late_side']} by {res['delta_mm']:.3f} mm  ({res['note']})")
    else:
        st.warning(res["note"])

# =========================
# CSV Mode
# =========================
if mode == "CSV":
    st.subheader("CSV mode (Cross-correlation + Cut-only)")
    file = st.file_uploader("Upload CSV (time, ch1, ch2)", type=["csv"])

    colx, coly, colz = st.columns(3)
    with colx:
        threshold = st.slider("Auto-window threshold", 0.05, 0.90, 0.30, 0.05)
    with coly:
        time_unit = st.selectbox("Time unit in CSV", ["s", "ms", "us", "ns", "ps"], index=0)
        time_scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}[time_unit]
    with colz:
        invert_late_rule = st.checkbox("Invert late-side rule (if ch1/ch2 seems swapped)", value=False)

    if file is not None and st.button("Quick Measure", type="primary"):
        try:
            df = pd.read_csv(file)

            t, ch1n, ch2n, corr, lags, lag_float, delay_s_raw, confidence, late_side = compute_delay_from_csv(
                df, threshold=threshold, time_scale=time_scale, invert_late_rule=invert_late_rule
            )

            res = cut_only_decision(delay_s_raw, late_side)

            # Log
            st.session_state.log.append({
                "measured_ps_raw": res["measured_ps_raw"],
                "measured_ps_abs": res["measured_ps_abs"],
                "late_side": res["late_side"],
                "action": res["action"],
                "delta_mm": res["delta_mm"],
                "confidence": confidence,
                "lag_samples": lag_float,
            })

            # ---- Result display
            st.subheader("Result")
            st.write(f"Measured (raw, corrected): {res['measured_ps_raw']:.3f} ps")
            st.write(f"Measured |.| used for decision: {res['measured_ps_abs']:.3f} ps")
            st.write(f"Late side (to shorten): {res['late_side']}")
            st.write(f"|delay|-target = {res['diff_ps_abs_minus_target']:.3f} ps")
            st.write(f"Confidence: {confidence:.2f}")
            st.write(f"Lag (samples): {lag_float:.3f}")

            # Gauge (shows diff)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res["diff_ps_abs_minus_target"],
                title={'text': "|delay| - target (ps)"},
                gauge={
                    'axis': {'range': [-10, 10]},
                    'steps': [
                        {'range': [-tol_ps, tol_ps], 'color': "green"},
                        {'range': [-10, -tol_ps], 'color': "red"},
                        {'range': [tol_ps, 10], 'color': "red"},
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

            if res["action"] == "OK":
                st.success(res["note"])
            elif res["action"] == "CUT":
                st.success(f"CUT {res['late_side']} by {res['delta_mm']:.3f} mm  ({res['note']})")
            else:
                st.warning(res["note"])

            # ---- Plots
            st.subheader("Correlation")
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(x=lags, y=corr, name="corr"))
            fig_corr.add_vline(x=lag_float, line_width=2, line_dash="dash", line_color="orange")
            fig_corr.update_layout(xaxis_title="Lag (samples)", yaxis_title="Correlation")
            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Waveforms (normalized)")
            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(x=t, y=ch1n, name="ch1"))
            fig_wave.add_trace(go.Scatter(x=t, y=ch2n, name="ch2"))
            fig_wave.update_layout(xaxis_title="Time (scaled to seconds)", yaxis_title="Amplitude (norm)")
            st.plotly_chart(fig_wave, use_container_width=True)

            st.subheader("Aligned (integer shift preview)")
            shift = int(np.round(lag_float))
            ch2_shift = np.roll(ch2n, -shift)
            fig_align = go.Figure()
            fig_align.add_trace(go.Scatter(x=t, y=ch1n, name="ch1"))
            fig_align.add_trace(go.Scatter(x=t, y=ch2_shift, name=f"ch2 aligned (shift={shift})"))
            fig_align.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude (norm)")
            st.plotly_chart(fig_align, use_container_width=True)

        except Exception as e:
            st.exception(e)

# =========================
# History / Export
# =========================
st.divider()
st.subheader("History")
log_df = pd.DataFrame(st.session_state.log)
st.dataframe(log_df, use_container_width=True)

if len(log_df) > 0:
    st.download_button(
        "Download Log CSV",
        log_df.to_csv(index=False),
        file_name="fiber_delay_log.csv",
        mime="text/csv"
    )
``
