import streamlit as st
import numpy as np
import altair as alt
import pandas as pd


def harmonic_with_noise(amplitude, frequency, phase, noise_mean, noise_covariance, show_noise, existing_noise=None):
    t = np.linspace(0, 1, 1000)
    harmonic = [amplitude * np.sin(2 * np.pi * frequency * x + phase) for x in t]

    if show_noise:
        if existing_noise is None:
            noise = [np.random.normal(loc=noise_mean, scale=noise_covariance ** 0.5) for _ in t]
        else:
            noise = existing_noise
        combined = [h + n for h, n in zip(harmonic, noise)]
        return t, combined, harmonic, noise
    return t, None, harmonic, existing_noise


def moving_average(signal, window_size):
    if signal is None or window_size < 1:
        return signal
    result = []
    half = window_size // 2
    for i in range(len(signal)):
        window = signal[max(i - half, 0):min(i + half + 1, len(signal))]
        result.append(sum(window) / len(window))
    return result


def_defaults = {
    'amplitude': 1.0,
    'frequency': 5.0,
    'phase': 0.0,
    'noise_mean': 0.0,
    'noise_covariance': 0.1,
    'show_noise': True,
    'filter_window': 10,
}

st.set_page_config(layout="wide")
st.title("Інтерактивна гармоніка з шумом і фільтром")

if 'noise' not in st.session_state:
    st.session_state.noise = None

if 'params' not in st.session_state:
    st.session_state.params = def_defaults.copy()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Параметри гармоніки")
    amplitude = st.slider("Амплітуда", 0.0, 5.0, st.session_state.params['amplitude'])
    frequency = st.slider("Частота", 0.1, 20.0, st.session_state.params['frequency'])
    phase = st.slider("Фаза", 0.0, 2 * np.pi, st.session_state.params['phase'])

    st.subheader("Параметри шуму")
    noise_mean = st.slider("Середнє шуму", -1.0, 1.0, st.session_state.params['noise_mean'])
    noise_covariance = st.slider("Дисперсія шуму", 0.0, 1.0, st.session_state.params['noise_covariance'])

    show_noise = st.checkbox("Показати шум", value=st.session_state.params['show_noise'])

    st.subheader("Фільтр")
    filter_window = st.slider("Розмір вікна фільтра", 1, 50, st.session_state.params['filter_window'])

    if st.button("Reset"):
        st.session_state.params = def_defaults.copy()
        st.session_state.noise = None
        st.rerun()

noise_params_changed = (
    st.session_state.params['noise_mean'] != noise_mean or
    st.session_state.params['noise_covariance'] != noise_covariance
)

st.session_state.params.update({
    'amplitude': amplitude,
    'frequency': frequency,
    'phase': phase,
    'noise_mean': noise_mean,
    'noise_covariance': noise_covariance,
    'show_noise': show_noise,
    'filter_window': filter_window,
})

t, noisy_signal, base_harmonic, noise = harmonic_with_noise(
    amplitude, frequency, phase,
    noise_mean, noise_covariance,
    show_noise,
    existing_noise=None if noise_params_changed else st.session_state.noise
)

if show_noise and noise_params_changed:
    st.session_state.noise = noise

filtered_signal = moving_average(noisy_signal, filter_window)

chart_data = pd.DataFrame({
    'Time': t,
    'Гармоніка': base_harmonic,
    'Фільтрована': filtered_signal
})
if noisy_signal:
    chart_data['Зашумлена'] = noisy_signal

with col2:
    st.subheader("Візуалізація сигналів")

    signal_types = ['Гармоніка', 'Фільтрована']
    colors = ['red', 'green']
    if noisy_signal:
        signal_types.insert(1, 'Зашумлена')
        colors.insert(1, 'blue')

    chart = alt.Chart(chart_data.reset_index()).transform_fold(
        signal_types,
        as_=['Signal Type', 'Value']
    ).mark_line().encode(
        x='Time:Q',
        y='Value:Q',
        color=alt.Color('Signal Type:N', scale=alt.Scale(domain=signal_types, range=colors))
    ).properties(height=500)

    st.altair_chart(chart, use_container_width=True)
