import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import butter, filtfilt

# Initial parameters
init_amplitude = 1.0
init_frequency = 5.0
init_phase = 0.0
init_noise_mean = 0.0
init_noise_cov = 0.1
init_filter_cutoff = 10.0

t = np.linspace(0, 1, 1000)

persistent_noise = np.random.normal(init_noise_mean, np.sqrt(init_noise_cov), t.shape)


def harmonic_with_noise(amplitude, frequency, phase, noise_mean, noise_covariance, show_noise):
    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    global persistent_noise
    noisy_signal = signal + persistent_noise if show_noise else signal
    return signal, noisy_signal


def apply_filter(signal, cutoff):
    b, a = butter(4, cutoff / (0.5 * 1000), btype='low')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.5)


pure_line, = ax.plot(t, np.zeros_like(t), label='Чиста')
noisy_line, = ax.plot(t, np.zeros_like(t), label='Шумна')
filtered_line, = ax.plot(t, np.zeros_like(t), label='Відфільтрована')
legend = ax.legend()


ax_amp = plt.axes([0.25, 0.4, 0.65, 0.03])
ax_freq = plt.axes([0.25, 0.35, 0.65, 0.03])
ax_phase = plt.axes([0.25, 0.3, 0.65, 0.03])
ax_nmean = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_ncov = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_fcut = plt.axes([0.25, 0.15, 0.65, 0.03])

s_amp = Slider(ax_amp, 'Amplitude', 0.1, 5.0, valinit=init_amplitude)
s_freq = Slider(ax_freq, 'Frequency', 1.0, 20.0, valinit=init_frequency)
s_phase = Slider(ax_phase, 'Phase', 0.0, 2*np.pi, valinit=init_phase)
s_nmean = Slider(ax_nmean, 'Noise Mean', -1.0, 1.0, valinit=init_noise_mean)
s_ncov = Slider(ax_ncov, 'Noise Covariance', 0.001, 1.0, valinit=init_noise_cov)
s_fcut = Slider(ax_fcut, 'Cutoff Frequency', 1.0, 50.0, valinit=init_filter_cutoff)


ax_check = plt.axes([0.025, 0.6, 0.15, 0.15])
check = CheckButtons(ax_check, ['Показати шумну', 'Показати відфільтровану'], [True, True])


ax_reset = plt.axes([0.025, 0.5, 0.1, 0.04])
btn_reset = Button(ax_reset, 'Reset')


show_noise = True
show_filtered = True
prev_noise_mean = init_noise_mean
prev_noise_cov = init_noise_cov


def update(val=None):
    global persistent_noise, prev_noise_mean, prev_noise_cov, show_noise, show_filtered
    amp = s_amp.val
    freq = s_freq.val
    phase = s_phase.val
    noise_mean = s_nmean.val
    noise_cov = s_ncov.val
    cutoff = s_fcut.val

    if noise_mean != prev_noise_mean or noise_cov != prev_noise_cov:
        persistent_noise = np.random.normal(noise_mean, np.sqrt(noise_cov), t.shape)
        prev_noise_mean = noise_mean
        prev_noise_cov = noise_cov

    signal, noisy = harmonic_with_noise(amp, freq, phase, noise_mean, noise_cov, show_noise)
    pure_line.set_ydata(signal)
    noisy_line.set_ydata(noisy if show_noise else [np.nan]*len(t))

    if show_filtered:
        filtered = apply_filter(noisy, cutoff)
        filtered_line.set_ydata(filtered)
    else:
        filtered_line.set_ydata([np.nan]*len(t))

    visible_lines = [pure_line]
    if show_noise:
        visible_lines.append(noisy_line)
    if show_filtered:
        visible_lines.append(filtered_line)
    ax.legend(handles=visible_lines)

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()


def reset(event):
    s_amp.reset()
    s_freq.reset()
    s_phase.reset()
    s_nmean.reset()
    s_ncov.reset()
    s_fcut.reset()


def on_check(label):
    global show_noise, show_filtered
    if label == 'Показати шумну':
        show_noise = not show_noise
    elif label == 'Показати відфільтровану':
        show_filtered = not show_filtered
    update()


s_amp.on_changed(update)
s_freq.on_changed(update)
s_phase.on_changed(update)
s_nmean.on_changed(update)
s_ncov.on_changed(update)
s_fcut.on_changed(update)
btn_reset.on_clicked(reset)
check.on_clicked(on_check)

update()
plt.show()
