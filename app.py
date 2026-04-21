import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

c = 299792458  # m/s

# -------------------------
# Page
# -------------------------
st.set_page_config(page_title="Fiber Delay Adjustment Tool", layout="wide")
st.title("Fiber Delay Adjustment Tool - FINAL (Fixed)")

# =========================
# 入力
# =========================
target_ps = st.number_input("Target delay (ps)", value=500.0)
wavelength = st.number_input("Wavelength (nm)", value=1550.0)

use_filter = st.checkbox("Enable filtering", value=True)
use_temp = st.checkbox("Temperature correction")
delta_T = st.number_input("ΔT (°C)", value=0.0)

pm_factor = st.number_input("PM correction factor", value=1.0)

mode = st.radio("Mode", ["Manual input", "CSV"], horizontal=True)

# =========================
# 状態保存
# =========================
if "log" not in st.session_state:
    st.session_state.log = []

# =========================
# 関数
# =========================
def ng_dispersion(lambda_nm: float) -> float:
    """群屈折率の簡易近似（必要ならここを置き換え）"""
    return 1.468 + 1e-5 * (lambda_nm - 1550.0)

def apply_filter(signal: np.ndarray, window: int = 7) -> np.ndarray:
    """移動平均フィルタ"""
    window = max(1, int(window))
    if window == 1:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")

def find_roi_indices(signal: np.ndarray, threshold: float = 0.3, margin: int = 50):
    """
    パルスらしい領域(ROI)のインデックス範囲を返す。
    threshold: |x|/max(|x|) が threshold を超える領域
    margin: ROIの前後に余白を付ける
    """
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
    """
    相関のピークを2次近似でサブサンプル推定
    """
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)

    y1, y2, y3 = y[idx - 1], y[idx], y[idx + 1]
    denom = (y1 - 2.0 * y2 + y3)
    if denom == 0:
        return float(idx)

    return float(idx) + 0.5 * (y1 - y3) / denom

def temp_corr(delay_s: float, dT: float) -> float:
    """
    温度補正（40 ps/km/K を delay_s に比例補正として適用）
    ※モデルは必要に応じて置き換え
    """
    return delay_s * (1.0 + (40e-12 / 1000.0) * dT)

def calc_all(delay_s: float, wavelength_nm: float) -> tuple[float, float, float]:
    """
    delay_s: 測定遅延 [s]
    return: (error_ps, delta_L[m], corrected_delay_s)
    """
    ng = ng_dispersion(wavelength_nm)

    if use_temp:
        delay_s = temp_corr(delay_s, delta_T)

    delay_s *= pm_factor

    target_s = target_ps * 1e-12
    error_s = delay_s - target_s
    error_ps = error_s * 1e12

    # 誤差を打ち消すための長さ修正
    delta_L = -error_s * c / ng

    return error_ps, delta_L, delay_s

def safe_standardize(x: np.ndarray):
    """平均0・分散1。std=0の場合は None を返して止める"""
    x = np.asarray(x)
    s = np.std(x)
    if s == 0 or not np.isfinite(s):
        return None
    return (x - np.mean(x)) / s

def compute_delay_from_csv(df: pd.DataFrame, threshold: float = 0.3):
    """
    CSVから遅延推定（相互相関）
    戻り値:
      t, ch1, ch2, corr, lags, lag_float, delay_s, confidence
    """
    if df.shape[1] < 3:
        raise ValueError("CSV must have at least 3 columns: time, ch1, ch2")

    t = df.iloc[:, 0].to_numpy(dtype=float)
    ch1 = df.iloc[:, 1].to_numpy(dtype=float)
    ch2 = df.iloc[:, 2].to_numpy(dtype=float)

    # NaN除去（行単位で）
    mask = np.isfinite(t) & np.isfinite(ch1) & np.isfinite(ch2)
    t, ch1, ch2 = t[mask], ch1[mask], ch2[mask]
    if len(t) < 3:
        raise ValueError("Not enough valid samples after removing NaNs.")

    # 単調増加チェック（オシロの時間軸前提）
    if not np.all(np.diff(t) > 0):
        raise ValueError("Time column must be strictly increasing.")

    # フィルタ
    if use_filter:
        ch1 = apply_filter(ch1, window=7)
        ch2 = apply_filter(ch2, window=7)

    # ROI（ch1/ch2それぞれ）→ union を使って t も同じ範囲で切る（重要）
    s1, e1 = find_roi_indices(ch1, threshold=threshold, margin=80)
    s2, e2 = find_roi_indices(ch2, threshold=threshold, margin=80)
    start = min(s1, s2)
    end = max(e1, e2)

    t = t[start:end]
    ch1 = ch1[start:end]
    ch2 = ch2[start:end]

    if len(t) < 3:
        raise ValueError("ROI is too small. Try lowering threshold or check waveform.")

    dt = t[1] - t[0]
    if dt <= 0:
        raise ValueError("Invalid dt computed from time vector.")

    # 標準化（std=0対策）
    ch1n = safe_standardize(ch1)
    ch2n = safe_standardize(ch2)
    if ch1n is None or ch2n is None:
        raise ValueError("Signal std is zero (flat). Check CSV columns / scaling.")

    # 相互相関
    corr = np.correlate(ch1n, ch2n, mode="full")
    idx = int(np.argmax(corr))
    idx_ref = refine_peak_parabolic(corr, idx)

    lag_float = idx_ref - (len(ch1n) - 1)  # サンプル数のずれ
    delay_s = lag_float * dt

    # 信頼度（ざっくり）
    peak = float(np.max(corr))
    noise = float(np.mean(np.abs(corr))) + 1e-12
    confidence = peak / noise

    lags = np.arange(-len(ch1n) + 1, len(ch1n), dtype=float)

    return t, ch1n, ch2n, corr, lags, lag_float, delay_s, confidence

# =========================
# 手入力モード
# =========================
if mode == "Manual input":
    measured_ps = st.number_input("Measured delay (ps)", value=500.0)
    delay_s = measured_ps * 1e-12

    error_ps, delta_L, corrected_delay_s = calc_all(delay_s, wavelength)

    st.subheader("Result")
    st.write(f"Measured (raw): {delay_s*1e12:.3f} ps")
    st.write(f"Measured (corrected): {corrected_delay_s*1e12:.3f} ps")

    if abs(error_ps) <= 1:
        st.success(f"OK (±1 ps)  error={error_ps:.3f} ps")
    else:
        st.warning(f"Error={error_ps:.3f} ps")
        if delta_L > 0:
            st.success(f"Add {delta_L*1000:.3f} mm")
        else:
            st.error(f"Cut {abs(delta_L*1000):.3f} mm")

# =========================
# CSVモード
# =========================
if mode == "CSV":
    file = st.file_uploader("Upload CSV (time, ch1, ch2)", type=["csv"])
    threshold = st.slider("Auto-window threshold", 0.05, 0.90, 0.30, 0.05)

    if file is not None:
        if st.button("Quick Measure", type="primary"):
            try:
                df = pd.read_csv(file)

                t, ch1n, ch2n, corr, lags, lag_float, delay_s, confidence = compute_delay_from_csv(
                    df, threshold=threshold
                )

                error_ps, delta_L, corrected_delay_s = calc_all(delay_s, wavelength)

                # ログ保存
                st.session_state.log.append({
                    "measured_ps": corrected_delay_s * 1e12,
                    "error_ps": error_ps,
                    "delta_mm": delta_L * 1000.0,
                    "confidence": confidence
                })

                # =========================
                # 結果表示
                # =========================
                st.subheader("Result")
                st.write(f"Measured (raw): {delay_s*1e12:.3f} ps")
                st.write(f"Measured (corrected): {corrected_delay_s*1e12:.3f} ps")

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
                st.plotly_chart(fig_gauge, use_container_width=True)

                if abs(error_ps) <= 1:
                    st.success("OK (±1 ps)")
                else:
                    if delta_L > 0:
                        st.success(f"Add {delta_L*1000:.3f} mm")
                    else:
                        st.error(f"Cut {abs(delta_L*1000):.3f} mm")

                st.write(f"Confidence: {confidence:.2f}")
                st.write(f"Lag (samples): {lag_float:.3f}")

                # =========================
                # 相関（ズーム）
                # =========================
                fig_corr = go.Figure()
                fig_corr.add_trace(go.Scatter(x=lags, y=corr, name="corr"))
                fig_corr.add_vline(x=lag_float, line_width=2, line_dash="dash", line_color="orange")
                fig_corr.update_layout(title="Correlation", xaxis_title="Lag (samples)", yaxis_title="Corr")
                st.plotly_chart(fig_corr, use_container_width=True)

                # =========================
                # 波形（正規化後）
                # =========================
                fig_wave = go.Figure()
                fig_wave.add_trace(go.Scatter(x=t, y=ch1n, name="ch1 (norm)"))
                fig_wave.add_trace(go.Scatter(x=t, y=ch2n, name="ch2 (norm)"))
                fig_wave.update_layout(title="Waveforms (Normalized)", xaxis_title="Time", yaxis_title="Amplitude")
                st.plotly_chart(fig_wave, use_container_width=True)

                # =========================
                # アライメント表示（整数シフト）
                # =========================
                shift = int(np.round(lag_float))
                ch2_shift = np.roll(ch2n, -shift)

                fig_align = go.Figure()
                fig_align.add_trace(go.Scatter(x=t, y=ch1n, name="ch1"))
                fig_align.add_trace(go.Scatter(x=t, y=ch2_shift, name=f"ch2 aligned (shift={shift})"))
                fig_align.update_layout(title="Aligned (integer shift)", xaxis_title="Time", yaxis_title="Amplitude")
                st.plotly_chart(fig_align, use_container_width=True)

            except Exception as e:
                st.error(f"Failed: {e}")

# =========================
# ログ表示
# =========================
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
    
