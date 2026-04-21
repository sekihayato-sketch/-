import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

c = 299792458  # m/s

st.title("Fiber Delay Adjustment Tool - FINAL")

# =========================
# 入力
# =========================
target_ps = st.number_input("Target delay (ps)", value=500.0)
wavelength = st.number_input("Wavelength (nm)", value=1550.0)

use_filter = st.checkbox("Enable filtering", value=True)
use_temp = st.checkbox("Temperature correction")
delta_T = st.number_input("ΔT (°C)", value=0.0)

pm_factor = st.number_input("PM correction factor", value=1.0)

mode = st.radio("Mode", ["Manual input", "CSV"])

# =========================
# 状態保存
# =========================
if "log" not in st.session_state:
    st.session_state.log = []

# =========================
# 関数
# =========================
def ng_dispersion(lambda_nm):
    return 1.468 + 1e-5 * (lambda_nm - 1550)

def apply_filter(signal, window=7):
    return np.convolve(signal, np.ones(window)/window, mode='same')

def auto_window(signal, threshold=0.3):
    norm = np.abs(signal) / np.max(np.abs(signal))
    idx = np.where(norm > threshold)[0]
    if len(idx) == 0:
        return signal
    return signal[idx[0]:idx[-1]]

def refine_peak(corr, idx):
    if idx <= 0 or idx >= len(corr)-1:
        return idx
    y1, y2, y3 = corr[idx-1], corr[idx], corr[idx+1]
    denom = (y1 - 2*y2 + y3)
    if denom == 0:
        return idx
    return idx + 0.5*(y1 - y3)/denom

def temp_corr(delay_s, dT):
    return delay_s * (1 + (40e-12/1000) * dT)

def calc_all(delay_s):
    ng = ng_dispersion(wavelength)

    if use_temp:
        delay_s = temp_corr(delay_s, delta_T)

    delay_s *= pm_factor

    target_s = target_ps * 1e-12
    error_s = delay_s - target_s
    error_ps = error_s * 1e12

    delta_L = - error_s * c / ng

    return error_ps, delta_L, delay_s

# =========================
# 手入力
# =========================
if mode == "Manual input":
    measured_ps = st.number_input("Measured delay (ps)", value=500.0)
    delay_s = measured_ps * 1e-12

    error_ps, delta_L, delay_s = calc_all(delay_s)

    st.subheader("Result")

    if abs(error_ps) <= 1:
        st.success(f"OK (±1 ps)  error={error_ps:.3f}")
    else:
        st.warning(f"Error={error_ps:.3f}")

        if delta_L > 0:
            st.success(f"Add {delta_L*1000:.3f} mm")
        else:
            st.error(f"Cut {abs(delta_L*1000):.3f} mm")

# =========================
# CSVモード
# =========================
file = st.file_uploader("Upload CSV (time, ch1, ch2)", type=["csv"])

if file and mode == "CSV":

    if st.button("Quick Measure"):

        df = pd.read_csv(file)

        t = df.iloc[:,0].values
        ch1 = df.iloc[:,1].values
        ch2 = df.iloc[:,2].values

        if use_filter:
            ch1 = apply_filter(ch1)
            ch2 = apply_filter(ch2)

        ch1 = auto_window(ch1)
        ch2 = auto_window(ch2)

        min_len = min(len(ch1), len(ch2))
        ch1 = ch1[:min_len]
        ch2 = ch2[:min_len]
        t = t[:min_len]

        dt = t[1] - t[0]

        ch1 = (ch1 - np.mean(ch1)) / np.std(ch1)
        ch2 = (ch2 - np.mean(ch2)) / np.std(ch2)

        corr = np.correlate(ch1, ch2, mode='full')
        idx = np.argmax(corr)
        idx_refined = refine_peak(corr, idx)

        lag = idx_refined - (len(ch1)-1)
        delay_s = lag * dt

        error_ps, delta_L, delay_s = calc_all(delay_s)

        # =========================
        # ログ保存
        # =========================
        st.session_state.log.append({
            "measured_ps": delay_s*1e12,
            "error_ps": error_ps,
            "delta_mm": delta_L*1000
        })

        # =========================
        # 結果
        # =========================
        st.subheader("Result")
        st.write(f"Measured: {delay_s*1e12:.3f} ps")

        # ゲージ
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=error_ps,
            title={'text': "Error (ps)"},
            gauge={
                'axis': {'range': [-10, 10]},
                'steps': [
                    {'range': [-1, 1], 'color': "green"},
                    {'range': [-10, -1], 'color': "red"},
                    {'range': [1, 10], 'color': "red"},
                ]
            }
        ))
        st.plotly_chart(fig_gauge)

        if abs(error_ps) <= 1:
            st.success("OK (±1 ps)")
        else:
            if delta_L > 0:
                st.success(f"Add {delta_L*1000:.3f} mm")
            else:
                st.error(f"Cut {abs(delta_L*1000):.3f} mm")

        # 信頼度
        peak = np.max(corr)
        noise = np.mean(np.abs(corr))
        confidence = peak / noise
        st.write(f"Confidence: {confidence:.2f}")

        # =========================
        # ズーム相関
        # =========================
        lags = np.arange(-len(ch1)+1, len(ch1))
        center = int(lag + len(ch1) - 1)

        zoom = 200
        start = max(center - zoom, 0)
        end = min(center + zoom, len(corr))

        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(x=lags[start:end], y=corr[start:end]))
        fig_corr.add_vline(x=lag)
        fig_corr.update_layout(title="Correlation (Zoom)")
        st.plotly_chart(fig_corr, use_container_width=True)

        # 波形
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=t, y=ch1, name="ch1"))
        fig_wave.add_trace(go.Scatter(x=t, y=ch2, name="ch2"))
        st.plotly_chart(fig_wave, use_container_width=True)

        # 補正後
        shift = int(lag)
        ch2_shift = np.roll(ch2, -shift)

        fig_align = go.Figure()
        fig_align.add_trace(go.Scatter(x=t, y=ch1, name="ch1"))
        fig_align.add_trace(go.Scatter(x=t, y=ch2_shift, name="aligned"))
        st.plotly_chart(fig_align, use_container_width=True)

# =========================
# ログ表示
# =========================
st.subheader("History")

log_df = pd.DataFrame(st.session_state.log)
st.dataframe(log_df)

if len(log_df) > 0:
    st.download_button(
        "Download Log CSV",
        log_df.to_csv(index=False),
        file_name="fiber_delay_log.csv"
    )
