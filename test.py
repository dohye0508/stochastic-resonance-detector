# -*- coding: utf-8 -*-
"""
Wildlife Detection System | Real-time Signal Processing Pipeline
Mapped to Research Report Chapters 2, 3, and 4.
"""
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from msr_engine import BistableDoubleWellEngine  # Object-Oriented double-well integration engine

# ---------------------------------------------------------
# Chapter 2: Sensor Hardware Configuration & Buffer Setup
# ---------------------------------------------------------
port = 'COM4'
try:
    ser = serial.Serial(port, 9600, timeout=0.05)
except Exception as e:
    print(f"Port connection failed: {e}")
    exit()

current_zero = 512.0
fs = 100.0  
fft_buffer_size = 128  # Sliding DFT buffer for time-frequency analysis (1.28s window)

data_buffer = [0.0] * fft_buffer_size

half_n = fft_buffer_size // 2
x_freq = np.fft.fftfreq(fft_buffer_size, 1/fs)[:half_n]

# ---------------------------------------------------------
# Chapter 3.1: Spectrum Parameters & Adaptive Noise Tracking
# ---------------------------------------------------------
NOISE_BAND_WIDTH = 5.0    # Soft attenuation bandwidth around dominant peak (Hz)
NOISE_PEAK_COUNT = 1      # Focus on the single most dominant vehicle noise peak
NOISE_MIN_FREQ = 5.0      # Lower cutoff to preserve ultra-low frequency steps
NOISE_MAX_FREQ = 48.0     # Upper boundary for vehicle noise search

# Live tracking array mapping dominant continuous noise: [(low, high, peak), ...]
detected_noise_bands = []

# Master Sensitivity Controller (Scales all potential barriers & trigger thresholds)
# Increase (>1.0) to decrease sensitivity. Decrease (<1.0) to increase sensitivity.
THRESHOLD_SCALAR = 1.618   

# Fallback visual limit parameter
ALERT_THRESHOLD = 60.0 * THRESHOLD_SCALAR

# Chapter 3.2: Transient Signal Preservation Parameters
ALPHA_EMA = 0.05  # Long-term noise profile EMA adaptation rate
FLUX_K = 2.5      # Spectral flux multiplier bypassing suppression for foot impacts

# Long-term frequency magnitude average (M_avg)
M_avg = np.zeros(half_n)

# ---------------------------------------------------------
# Engineering Extension: Footstep Duration Analyzer (Schmitt Trigger)
# ---------------------------------------------------------
THRESHOLD_START = 15.0 * THRESHOLD_SCALAR
THRESHOLD_END = 8.0 * THRESHOLD_SCALAR
HANG_TIME = 0.1
FILENAME_CSV = "step_duration_data.csv"

is_recording = False
step_start_time = 0.0
end_candidate_time = 0.0
step_durations = []

# ---------------------------------------------------------
# ---------------------------------------------------------
# Chapter 4.1.1: Non-linear Bistable Double-Well Potential Integration Setup
# ---------------------------------------------------------
SIGMA_NOISE = 12.0     # Artificial Gaussian White Noise intensity (σ)

# Instantiate encapsulated integration engine mapped directly to report Chapter 4.1.1
# a=50, b=50 and force_scalar=1.0 drastically increase particle mobility to overcome 100Hz dt inertia.
bistable_engine = BistableDoubleWellEngine(a=50.0, b=50.0, dt=1.0/fs, force_scalar=1.5, bound=2.2)

# ---------------------------------------------------------
# UI Setup: 3-Stage Pipeline Mapping
# ---------------------------------------------------------
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 11))
fig.canvas.manager.set_window_title('Wildlife Detection System | Real-time Analysis')
fig.patch.set_facecolor('#f0f3f7')

x_time = np.arange(fft_buffer_size)

# [Plot 1] Stage 1: SDFT Filtered Signal Output
line_time_clean, = ax1.plot(x_time, data_buffer, color='#0984e3', linewidth=2.0,
                            label='Filtered Output (Transient Preserved)')
ax1.set_title('[Stage 1] SDFT Adaptive Filter — Continuous Noise Suppressed, Footsteps Preserved',
              fontsize=11, fontweight='bold', pad=10, color='#2d3436')
ax1.set_xlim(0, fft_buffer_size - 1)
ax1.set_ylim(-100, 100)
ax1.set_xlabel(f'Sample Index (Buffer: {fft_buffer_size} samples = {fft_buffer_size/fs*1000:.0f}ms)')
ax1.set_ylabel('Amplitude')
ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_facecolor('#ffffff')

alert_text = ax1.text(0.5, 1.13, 'STATUS: MONITORING', transform=ax1.transAxes,
                      fontsize=11, fontweight='bold', color='white', ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='#00b894', edgecolor='none', alpha=0.92))

duration_text = ax1.text(0.02, 0.88, 'Step Event: Waiting...', transform=ax1.transAxes,
                         fontsize=10, fontweight='bold', color='#2d3436',
                         bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffeaa7', edgecolor='#fdcb6e', alpha=0.9))

bypass_label = ax1.text(0.98, 0.12, 'Transient Bypass: OFF', transform=ax1.transAxes,
                        fontsize=9, fontweight='bold', color='#b2bec3', ha='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#dfe6e9'))

# [Plot 2] Stage 2: Stochastic Resonance Event Extractor (Double-Well Integration Output)
line_time_sr, = ax2.plot(x_time, data_buffer, color='#a29bfe', linewidth=1.5,
                         label='Bistable State Output x(t) — Real-time Euler Integration')
ax2.axhline(0.0, color='#d63031', linestyle='--', linewidth=1.5,
            label='Center Potential Barrier (x=0)')
ax2.axhline(1.0, color='#27ae60', linestyle=':', linewidth=1.2, label='Stable Well (+1.0)')
ax2.axhline(-1.0, color='#27ae60', linestyle=':', linewidth=1.2, label='Stable Well (-1.0)')
ax2.set_title('[Stage 2] Stochastic Resonance — N(t): Total Crossings / K(t): Noise Baseline / N-K: Pure Impulses',
              fontsize=11, fontweight='bold', pad=10, color='#2d3436')
ax2.set_xlim(0, fft_buffer_size - 1)
ax2.set_ylim(-2.2, 2.2)
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Amplitude')
ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.set_facecolor('#ffffff')

sr_text = ax2.text(0.02, 0.88, 'N(t): - | K(t): - | Net Events (N-K): Waiting...',
                   transform=ax2.transAxes, fontsize=10, fontweight='bold', color='#2c3e50',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f8f5', edgecolor='#1abc9c', alpha=0.9))

# [Plot 3] Stage 3: Frequency Spectrum & AI Tracking Band
line_freq_raw, = ax3.plot(x_freq, np.zeros(half_n), color='#b2bec3', linewidth=1.5,
                          label='Instantaneous FFT M(t)', alpha=0.7)
line_freq_clean, = ax3.plot(x_freq, np.zeros(half_n), color='#d63031', linewidth=2.0,
                            label='Filtered Spectrum')
line_freq_avg, = ax3.plot(x_freq, np.zeros(half_n), color='#e67e22', linestyle=':', linewidth=2.0,
                          label=f'Long-term Avg M_avg (a={ALPHA_EMA})')

ax3.axvspan(NOISE_MIN_FREQ, NOISE_MAX_FREQ, color='#dfe6e9', alpha=0.2,
            label=f'Search Boundaries ({NOISE_MIN_FREQ}-{NOISE_MAX_FREQ}Hz)')
noise_band_patches = []

ax3.set_title('[Stage 3] Real-time Spectrum — Yellow: Tracked Noise Attenuation Band',
              fontsize=11, fontweight='bold', pad=10, color='#2d3436')
ax3.set_xlim(0, fs / 2)
ax3.set_ylim(0, 50)
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Magnitude')
ax3.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax3.grid(True, linestyle='--', alpha=0.4)
ax3.set_facecolor('#ffffff')

band_info_text = ax3.text(0.02, 0.88, 'Tracked Peak: Initializing...',
                          transform=ax3.transAxes, fontsize=10, fontweight='bold', color='#2d3436',
                          bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3cd', edgecolor='#ffc107', alpha=0.9))

plt.tight_layout(pad=2.0)
print("System Engine Active...")

# ---------------------------------------------------------
# Execution Loop: Sliding Buffer Integration
# ---------------------------------------------------------
last_render_time = time.time()
render_interval = 0.03  

try:
    while plt.fignum_exists(fig.number):
        data_updated = False

        # Read serial buffers
        while ser.in_waiting > 0:
            ln = ser.readline().decode('utf-8', errors='ignore').strip()
            if ln:
                try:
                    raw_val = int(ln)
                    calibrated_val = raw_val - current_zero
                    data_buffer.pop(0)
                    data_buffer.append(calibrated_val)
                    data_updated = True
                except ValueError:
                    pass

        current_time = time.time()
        if data_updated and (current_time - last_render_time >= render_interval):
            raw_signal = np.array(data_buffer)
            signal = raw_signal - np.mean(raw_signal)  # DC suppression

            # Acceleration Metric: Differentiate signal capturing physical impact energy
            diff_signal = np.diff(signal)
            diff_rms = np.sqrt(np.mean(diff_signal**2)) if len(diff_signal) > 0 else 0.0
            activity_index = diff_rms * 1.5

            # Footstep time-domain group analyzer
            if not is_recording:
                if activity_index > THRESHOLD_START:
                    is_recording = True
                    step_start_time = current_time
                    duration_text.set_text('Step Event: [ Recording... ]')
                    duration_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#55efc4', edgecolor='#00b894', alpha=0.9))
            else:
                if activity_index > THRESHOLD_END:
                    end_candidate_time = current_time
                else:
                    if current_time - end_candidate_time > HANG_TIME:
                        duration = end_candidate_time - step_start_time
                        if duration > 0.01:
                            step_durations.append(duration)
                            avg_duration = sum(step_durations) / len(step_durations)
                            print(f"[Step Event] Duration: {duration:.3f}s | Avg: {avg_duration:.3f}s | Count: {len(step_durations)}")
                            try:
                                with open(FILENAME_CSV, 'a', newline='', encoding='utf-8') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), f"{duration:.3f}"])
                            except Exception:
                                pass
                            duration_text.set_text(f'Step Event: {duration:.3f}s (Avg: {avg_duration:.3f}s | N={len(step_durations)})')
                            duration_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#74b9ff', edgecolor='#0984e3', alpha=0.9))
                        is_recording = False

            # Chapter 3.2: SDFT Real-time Analysis & Attenuation Logic
            raw_fft_complex = np.fft.fft(signal)
            M_t = np.abs(raw_fft_complex[:half_n]) / half_n
            freqs = np.fft.fftfreq(fft_buffer_size, 1/fs)

            # Update long-term noise profile via Exponential Moving Average
            if np.sum(M_avg) == 0:
                M_avg = np.copy(M_t)
            else:
                M_avg = (1.0 - ALPHA_EMA) * M_avg + ALPHA_EMA * M_t

            line_freq_avg.set_ydata(M_avg)

            # Identify dominant noise band boundaries dynamically
            valid_mask = (x_freq >= NOISE_MIN_FREQ) & (x_freq <= NOISE_MAX_FREQ)
            valid_indices = np.where(valid_mask)[0]

            detected_noise_bands.clear()
            if len(valid_indices) > 0:
                M_avg_valid = M_avg.copy()
                for _ in range(NOISE_PEAK_COUNT):
                    if np.sum(M_avg_valid[valid_indices]) == 0:
                        break
                    peak_idx = valid_indices[np.argmax(M_avg_valid[valid_indices])]
                    peak_freq = x_freq[peak_idx]
                    band_low = max(NOISE_MIN_FREQ, peak_freq - NOISE_BAND_WIDTH)
                    band_high = min(NOISE_MAX_FREQ, peak_freq + NOISE_BAND_WIDTH)
                    detected_noise_bands.append((band_low, band_high, peak_freq))
                    suppress_mask = (x_freq >= band_low) & (x_freq <= band_high)
                    M_avg_valid[suppress_mask] = 0.0

            clean_fft_complex = np.copy(raw_fft_complex)
            transient_detected = False

            # Apply continuous soft attenuation ratio protecting transients
            for i in range(half_n):
                abs_f = abs(freqs[i])
                in_noise_band = any(lo <= abs_f <= hi for lo, hi, _ in detected_noise_bands)
                if in_noise_band:
                    # Spectral flux validation bypasses step components from suppression
                    if M_t[i] <= M_avg[i] * FLUX_K:
                        ratio = M_t[i] / (M_avg[i] * FLUX_K) if M_avg[i] > 0 else 0.0
                        dynamic_gain = 0.02 + 0.48 * (ratio**2)  # Soft curve suppression
                        clean_fft_complex[i] *= dynamic_gain
                        if i > 0:
                            clean_fft_complex[fft_buffer_size - i] *= dynamic_gain
                    else:
                        transient_detected = True

            filtered_signal = np.fft.ifft(clean_fft_complex).real
            line_time_clean.set_ydata(filtered_signal)

            # Chapter 3.2 & Chapter 4: Matched Stochastic Resonance Net Event Separation
            white_noise = np.random.normal(0, SIGMA_NOISE, fft_buffer_size)
            noisy_signal = filtered_signal + white_noise
            
            # Delegate buffer integration to object-oriented double-well engine module
            x_arr_total, x_arr_noise, N_t, K_t = bistable_engine.process_buffer(filtered_signal, white_noise)
            line_time_sr.set_ydata(x_arr_total)
            net_events = max(0, N_t - K_t)  # Pure biological well hops

            sr_text.set_text(f'N(t) Total: {N_t} | K(t) Noise: {K_t} | Net Events (N-K): {net_events}')
            if net_events > 0:
                sr_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#a3e4d7', edgecolor='#1abc9c', alpha=0.95))
            else:
                sr_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#e8f8f5', edgecolor='#1abc9c', alpha=0.85))

            # UI Status update mapping
            clean_fft_mag = np.abs(clean_fft_complex[:half_n]) / half_n

            if transient_detected:
                bypass_label.set_text('Transient Bypass: ON')
                bypass_label.set_color('#27ae60')
            else:
                bypass_label.set_text('Transient Bypass: OFF')
                bypass_label.set_color('#b2bec3')

            # Chapter 3.3 Report Compliance: Alert strictly triggered via pure SR net events
            if net_events >= 2:
                alert_text.set_text(f'WILD ANIMAL DETECTED! (SR Net Events: {net_events})')
                alert_text.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='#d63031', edgecolor='none', alpha=0.97))
                fig.patch.set_facecolor('#fff0f0')
            else:
                alert_text.set_text('STATUS: MONITORING')
                alert_text.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='#00b894', edgecolor='none', alpha=0.92))
                fig.patch.set_facecolor('#f0f3f7')

            # Real-time boundary rescalers
            max_clean_amp = max(np.max(np.abs(filtered_signal)), 20.0)
            ax1.set_ylim(-max_clean_amp * 1.3, max_clean_amp * 1.3)

            # Double-well potential output bounds fixed nicely showing well-hopping physics
            ax2.set_ylim(-2.2, 2.2)

            line_freq_raw.set_ydata(M_t)
            line_freq_clean.set_ydata(clean_fft_mag)

            # Redraw real-time tracking polygons
            for p in noise_band_patches:
                p.remove()
            noise_band_patches.clear()
            band_info_parts = []
            for (blo, bhi, bpeak) in detected_noise_bands:
                patch = ax3.axvspan(blo, bhi, color='#ffeaa7', alpha=0.6)
                noise_band_patches.append(patch)
                band_info_parts.append(f'{bpeak:.1f}Hz (+/-{NOISE_BAND_WIDTH}Hz)')
            if band_info_parts:
                band_info_text.set_text('Tracked Peak: ' + ' | '.join(band_info_parts))
            else:
                band_info_text.set_text('Tracked Peak: None')

            max_mag = max(np.max(M_t), np.max(M_avg), 10.0)
            ax3.set_ylim(0, max_mag * 1.3)

            fig.canvas.draw()
            fig.canvas.flush_events()
            last_render_time = current_time

        time.sleep(0.001)

except KeyboardInterrupt:
    pass
finally:
    ser.close()
    plt.ioff()
    plt.close()