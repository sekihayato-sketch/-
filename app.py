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
st.title("Fiber Delay Adjustment Tool - ABS Target 500 ps / Cut-only")

# =====================================================
# Inputs
# =====================================================
colA, colB, colC = st.columns(3)
with colA:
    target_ps = st.number_input("Target |delay| (ps)", value=500.0)
    tol_ps = st.number_input("Tolerance (ps)", value=1.0)
with colB:
    wavelength = st.number_input("Wavelength (nm)", value=1550.0)
    pm_factor = st.number_input("PM correction factor", value=1.0, format="%.6f")
with colC:
    use_temp = st.checkbox("Temperature correction")
    delta_T = st.number_input("ΔT (°C)", value=0.0)
    use_filter = st.checkbox("Enable filtering", value=True)

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
    # Simple group index approximation
    return 1.468 + 1e-5 * (lambda_nm - 1550.0)

def temp_corr(delay_s: float, dT: float) -> float:
    # 40 ps / km / K model
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
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")

def safe_standardize(x: np.ndarray):
    s = np.std(x)
    if s == 0 or not np.isfinite(s):
        return None
    return (x - np.mean(x)) / s

def find_roi_indices(signal: np.ndarray, threshold: float = 0.3, margin: int = 80):
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

# =====================================================
# Decision logic (FINAL)
# =====================================================
def cut_only_decision_abs_target(delay_s_raw: float, late_side: str):
    """
    ABS target (|delay| = target_ps), Cut-only:
      |delay| > target -> CUT LATE side
      |delay| < target -> CUT EARLY side
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
        # too large -> reduce |delay|
        delta_tau_s = diff * 1e-12
        cut_side = late_side
        note = "Too long -> cut LATE (longer) side"
    else:
        # too small -> increase |delay|
        delta_tau_s = (-diff) * 1e-12
        cut_side = early_side
        note = "Too short -> cut EARLY (shorter) side"

    delta_L_m = (delta_tau_s * c) / ng

    result["cut_side"] = cut_side
    result["delta_mm"] = delta_L_m * 1000.0
    result["note"] = note
    return result

# =====================================================
# Manual Mode
# =====================================================
if mode == "Manual input":
    st.subheader("Manual input")
    col1, col2 = st.columns(2)
    with col1:
        measured_ps_input = st.number_input("Measured delay (ps, signed)", value=500.0)
    with col2:
        late_side_manual = st.radio("Late side (longer)", ["ch1", "ch2"], horizontal=True)

    delay_s_raw = measured_ps_input * 1e-12
    res = cut_only_decision_abs_target(delay_s_raw, late_side_manual)

    st.subheader("Result")
    st.write(f"Measured (corrected): {res['measured_ps_raw']:.3f} ps")
    st.write(f"|delay| used for decision: {res['measured_ps_abs']:.3f} ps")
    st.write(f"|delay| - target = {res['diff_ps']:.3f} ps")
    st.write(f"Late (longer): {res['late_side']}")
    st.write(f"Early (shorter): {res['early_side']}")

    if res["cut_side"] is None:
        st.success(res["note"])
    else:
        st.success(f"CUT {res['cut_side']} by {res['delta_mm']:.3f} mm")
        st.info(res["note"])

# =====================================================
# CSV Mode
# =====================================================
if mode == "CSV":
    st.subheader("CSV mode")
    file = st.file_uploader("Upload CSV (time, ch1, ch2)", type=["csv"])

    colx, coly, colz = st.columns(3)
    with colx:
        threshold = st.slider("Auto-window threshold", 0.05, 0.90, 0.30, 0.05)
    with coly:
        time_unit = st.selectbox("Time unit", ["s", "ms", "us", "ns", "ps"])
        time_scale = {"s":1.0, "ms":1e-3, "us":1e-6, "ns":1e-9, "ps":1e-12}[time_unit]
    with colz:
        invert_late_rule = st.checkbox("Invert late/early rule")

    if file is not None and st.button("Quick Measure", type="primary"):
        try:
            df = pd.read_csv(file)

            t = df.iloc[:,0].to_numpy(float) * time_scale
            ch1 = df.iloc[:,1].to_numpy(float)
            ch2 = df.iloc[:,2].to_numpy(float)

            if use_filter:
                ch1 = apply_filter(ch1)
                ch2 = apply_filter(ch2)

            s1,e1 = find_roi_indices(ch1, threshold)
            s2,e2 = find_roi_indices(ch2, threshold)
            start, end = min(s1,s2), max(e1,e2)

            t, ch1, ch2 = t[start:end], ch1[start:end], ch2[start:end]
            dt = t[1] - t[0]

            ch1n = safe_standardize(ch1)
            ch2n = safe_standardize(ch2)

            corr = np.correlate(ch1n, ch2n, mode="full")
            idx = np.argmax(corr)
            idx_ref = refine_peak_parabolic(corr, idx)
            lag = idx_ref - (len(ch1n) - 1)
            delay_s_raw = lag * dt

            late_side = "ch2" if lag > 0 else "ch1"
            if invert_late_rule:
                late_side = "ch1" if late_side == "ch2" else "ch2"

            res = cut_only_decision_abs_target(delay_s_raw, late_side)

            st.subheader("Result")
            st.write(f"Lag (samples): {lag:.3f}")
            st.write(f"Measured (corrected): {res['measured_ps_raw']:.3f} ps")
            st.write(f"|delay|: {res['measured_ps_abs']:.3f} ps")
            st.write(f"|delay| - target = {res['diff_ps']:.3f} ps")
            st.write(f"Late (longer): {res['late_side']}")
            st.write(f"Early (shorter): {res['early_side']}")

            if res["cut_side"] is None:
                st.success(res["note"])
            else:
                st.success(f"CUT {res['cut_side']} by {res['delta_mm']:.3f} mm")
                st.info(res["note"])

        except Exception as e:
            st.exception(e)

# =====================================================
# History
# =====================================================
st.divider()
st.subheader("History")
log_df = pd.DataFrame(st.session_state.log)
st.dataframe(log_df, use_container_width=True)
