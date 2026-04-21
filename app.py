import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

c = 299792458  # m/s

NG = {
    "O": 1.467,
    "C": 1.468
}

st.title("Fiber Delay Adjustment Tool")

band = st.selectbox("Band", ["C", "O"])
target_ps = st.number_input("Target delay (ps)", value=500.0)

# モード切替
mode = st.radio("Mode", ["Manual input", "CSV"])

# =========================
# 手入力モード
# =========================
if mode == "Manual input":
    measured_ps = st.number_input("Measured delay (ps)", value=500.0)

    ng = NG[band]

    delay_s = measured_ps * 1e-12
    target_s = target_ps * 1e-12

    error_s = delay_s - target_s
    error_ps = error_s * 1e12

    delta_L = - error_s * c / ng  # m → 符号付き

    st.subheader("Result")

    # ✅ 許容範囲チェック（±1 ps）
    if abs(error_ps) <= 1:
        st.success(f"OK ✅ within ±1 ps (error: {error_ps:.2f} ps)")
    else:
        st.warning(f"Out of tolerance (error: {error_ps:.2f} ps)")

        if delta_L > 0:
            st.success(f"Add fiber: {delta_L*1000:.2f} mm")
        else:
            st.error(f"Cut fiber: {abs(delta_L*1000):.2f} mm")

# =========================
# CSVモード
# =========================
file = st.file_uploader("Upload CSV (time, ch1, ch2)", type=["csv"])

def calc_delay_corr(t, ch1, ch2):
    dt = t[1] - t[0]

    ch1 = ch1 - np.mean(ch1)
    ch2 = ch2 - np.mean(ch2)

    corr = np.correlate(ch1, ch2, mode='full')
    lag = np.argmax(corr) - (len(ch1) - 1)

    delay = lag * dt
    return delay, corr

if file and mode == "CSV":
    df = pd.read_csv(file)

    t = df.iloc[:,0].values
    ch1 = df.iloc[:,1].values
    ch2 = df.iloc[:,2].values

    delay_s, corr = calc_delay_corr(t, ch1, ch2)

    ng = NG[band]
    target_s = target_ps * 1e-12

    error_s = delay_s - target_s
    error_ps = error_s * 1e12

    delta_L = - error_s * c / ng

    st.subheader("Result")

    st.write(f"Measured delay: {delay_s*1e12:.2f} ps")

    # ✅ 許容範囲チェック（±1 ps）
    if abs(error_ps) <= 1:
        st.success(f"OK ✅ within ±1 ps (error: {error_ps:.2f} ps)")
    else:
        st.warning(f"Out of tolerance (error: {error_ps:.2f} ps)")

        if delta_L > 0:
            st.success(f"Add fiber: {delta_L*1000:.2f} mm")
        else:
            st.error(f"Cut fiber: {abs(delta_L*1000):.2f} mm")

    # 波形表示
    fig1, ax1 = plt.subplots()
    ax1.plot(t, ch1, label="ch1")
    ax1.plot(t, ch2, label="ch2")
    ax1.legend()
    ax1.set_title("Waveforms")
    st.pyplot(fig1)

    # 相関表示
    fig2, ax2 = plt.subplots()
    ax2.plot(corr)
    ax2.set_title("Cross-correlation")
    st.pyplot(fig2)


