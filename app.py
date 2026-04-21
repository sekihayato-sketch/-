import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

c = 299792458  # m/s

# 群屈折率（近似）
NG = {
    "O": 1.467,
    "C": 1.468
}

st.title("Fiber Delay Adjustment Tool (Advanced)")

band = st.selectbox("Band", ["C", "O"])
target_ps = st.number_input("Target delay (ps)", value=500.0)

# 温度設定
use_temp = st.checkbox("Enable temperature correction")
delta_T = st.number_input("Temperature change ΔT (°C)", value=0.0)

# PM補正（簡易）
pm_factor = st.number_input("PM axis correction factor (≈1.0)", value=1.0)

mode = st.radio("Mode", ["Manual input", "CSV"])

# =========================
# サブps補間関数（2次フィット）
# =========================
def refine_peak(corr, lag):
    if lag <= 0 or lag >= len(corr)-1:
        return lag

    y1, y2, y3 = corr[lag-1], corr[lag], corr[lag+1]

    denom = (y1 - 2*y2 + y3)
    if denom == 0:
        return lag

    shift = 0.5 * (y1 - y3) / denom
    return lag + shift

# =========================
# 温度補正
# =========================
def temp_correction(delay_s, delta_T):
    coeff = 40e-12 / 1000  # 40 ps/km/K → s/m/K
    return delay_s + delay_s * coeff * delta_T

# =========================
# 手入力モード
# =========================
if mode == "Manual input":
    measured_ps = st.number_input("Measured delay (ps)", value=500.0)

    ng = NG[band]

    delay_s = measured_ps * 1e-12

    # 温度補正
    if use_temp:
        delay_s = temp_correction(delay_s, delta_T)

    # PM補正
    delay_s *= pm_factor

    target_s = target_ps * 1e-12

    error_s = delay_s - target_s
    error_ps = error_s * 1e12

    delta_L = - error_s * c / ng

    st.subheader("Result")

    if abs(error_ps) <= 1:
        st.success(f"OK ✅ within ±1 ps (error: {error_ps:.3f} ps)")
    else:
        st.warning(f"Out of tolerance (error: {error_ps:.3f} ps)")

        if delta_L > 0:
            st.success(f"Add fiber: {delta_L*1000:.3f} mm")
        else:
            st.error(f"Cut fiber: {abs(delta_L*1000):.3f} mm")

# =========================
# CSVモード
# =========================
file = st.file_uploader("Upload CSV (time, ch1, ch2)", type=["csv"])

def calc_delay_corr(t, ch1, ch2):
    dt = t[1] - t[0]

    ch1 = ch1 - np.mean(ch1)
    ch2 = ch2 - np.mean(ch2)

    corr = np.correlate(ch1, ch2, mode='full')
    lag = np.argmax(corr)

    refined_lag = refine_peak(corr, lag)

    lag_shift = refined_lag - (len(ch1) - 1)

    delay = lag_shift * dt
    return delay, corr

if file and mode == "CSV":
    df = pd.read_csv(file)

    t = df.iloc[:,0].values
    ch1 = df.iloc[:,1].values
    ch2 = df.iloc[:,2].values

    delay_s, corr = calc_delay_corr(t, ch1, ch2)

    # 温度補正
    if use_temp:
        delay_s = temp_correction(delay_s, delta_T)

    # PM補正
    delay_s *= pm_factor

    ng = NG[band]
    target_s = target_ps * 1e-12

    error_s = delay_s - target_s
    error_ps = error_s * 1e12

    delta_L = - error_s * c / ng

    st.subheader("Result")

    st.write(f"Measured delay: {delay_s*1e12:.3f} ps")

    if abs(error_ps) <= 1:
        st.success(f"OK ✅ within ±1 ps (error: {error_ps:.3f} ps)")
    else:
        st.warning(f"Out of tolerance (error: {error_ps:.3f} ps)")

        if delta_L > 0:
            st.success(f"Add fiber: {delta_L*1000:.3f} mm")
        else:
            st.error(f"Cut fiber: {abs(delta_L*1000):.3f} mm")

    fig1, ax1 = plt.subplots()
    ax1.plot(t, ch1, label="ch1")
    ax1.plot(t, ch2, label="ch2")
    ax1.legend()
    ax1.set_title("Waveforms")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(corr)
    ax2.set_title("Cross-correlation")
    st.pyplot(fig2)
