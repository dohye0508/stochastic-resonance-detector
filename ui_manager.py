import matplotlib.pyplot as plt
import numpy as np
import config

class RealTimeUI:
    """
    [보고서 제 3.3절]: 경고 출력 및 실시간 시각화 매니저
    파이프라인의 3단계 상태를 Matplotlib을 통해 시각적으로 렌더링하고,
    분석 결과에 따라 시각적인 경고 배너를 실시간으로 업데이트합니다.
    (폰트 깨짐 방지를 위해 UI 텍스트는 영문으로 유지합니다.)
    """
    def __init__(self):
        self.fs = config.FS
        self.buffer_size = config.BUFFER_SIZE
        self.half_n = self.buffer_size // 2
        
        self.x_time = np.arange(self.buffer_size)
        self.x_freq = np.fft.fftfreq(self.buffer_size, 1/self.fs)[:self.half_n]
        self.noise_band_width = config.NOISE_BAND_WIDTH
        
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(13, 11))
        self.fig.canvas.manager.set_window_title('Wildlife Detection System | Real-time Analysis')
        self.fig.patch.set_facecolor('#f0f3f7')
        
        # Stage 1 Setup
        self.line_time_clean, = self.ax1.plot(self.x_time, np.zeros(self.buffer_size), color='#0984e3', linewidth=2.0, label='Filtered Output')
        self.ax1.set_title('[Stage 1] SDFT Adaptive Filter — Continuous Noise Suppressed, Footsteps Preserved', fontsize=11, fontweight='bold', pad=10, color='#2d3436')
        self.ax1.set_xlim(0, self.buffer_size - 1)
        self.ax1.set_xlabel(f'Sample Index (Buffer: {self.buffer_size} samples = {self.buffer_size/self.fs*1000:.0f}ms)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)
        self.ax1.grid(True, linestyle='--', alpha=0.4)
        self.ax1.set_facecolor('#ffffff')
        
        self.alert_text = self.ax1.text(0.5, 1.13, 'STATUS: MONITORING', transform=self.ax1.transAxes, fontsize=11, fontweight='bold', color='white', ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#00b894', edgecolor='none', alpha=0.92))
        self.duration_text = self.ax1.text(0.02, 0.88, 'Step Event: Waiting...', transform=self.ax1.transAxes, fontsize=10, fontweight='bold', color='#2d3436', bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffeaa7', edgecolor='#fdcb6e', alpha=0.9))
        self.bypass_label = self.ax1.text(0.98, 0.12, 'Transient Bypass: OFF', transform=self.ax1.transAxes, fontsize=9, fontweight='bold', color='#b2bec3', ha='right', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#dfe6e9'))
        
        # Stage 2 Setup
        self.line_time_sr, = self.ax2.plot(self.x_time, np.zeros(self.buffer_size), color='#a29bfe', linewidth=1.5, label='Bistable Binary State: sgn(x(t))')
        self.ax2.axhline(0.0, color='#d63031', linestyle='--', linewidth=1.5, label='Center Potential Barrier (x=0)')
        self.ax2.axhline(1.0, color='#27ae60', linestyle=':', linewidth=1.2, label='Stable Well (+1.0)')
        self.ax2.axhline(-1.0, color='#27ae60', linestyle=':', linewidth=1.2, label='Stable Well (-1.0)')
        self.ax2.set_title(f'[Stage 2] Stochastic Resonance & ACF — N(t): Total Crossings / K(t): Noise Baseline / N-K: Pure Impulses (R >= {config.ACF_R_THRESHOLD})', fontsize=11, fontweight='bold', pad=10, color='#2d3436')
        self.ax2.set_xlim(0, self.buffer_size - 1)
        self.ax2.set_ylim(-2.2, 2.2)
        self.ax2.set_xlabel('Sample Index')
        self.ax2.set_ylabel('Amplitude')
        self.ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
        self.ax2.grid(True, linestyle='--', alpha=0.4)
        self.ax2.set_facecolor('#ffffff')
        
        self.sr_text = self.ax2.text(0.02, 0.88, 'N(t): - | K(t): - | Net Events (N-K): Waiting...', transform=self.ax2.transAxes, fontsize=10, fontweight='bold', color='#2c3e50', bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f8f5', edgecolor='#1abc9c', alpha=0.9))
        self.acf_text = self.ax2.text(0.02, 0.74, 'ACF Rhythm (R): Waiting...', transform=self.ax2.transAxes, fontsize=10, fontweight='bold', color='#2c3e50', ha='left', bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5eef8', edgecolor='#9b59b6', alpha=0.9))
        
        # Stage 3 Setup
        self.line_freq_raw, = self.ax3.plot(self.x_freq, np.zeros(self.half_n), color='#b2bec3', linewidth=1.5, label='Instantaneous FFT M(t)', alpha=0.7)
        self.line_freq_clean, = self.ax3.plot(self.x_freq, np.zeros(self.half_n), color='#d63031', linewidth=2.0, label='Filtered Spectrum')
        self.line_freq_avg, = self.ax3.plot(self.x_freq, np.zeros(self.half_n), color='#e67e22', linestyle=':', linewidth=2.0, label=f'Long-term Avg M_avg (a={config.ALPHA_EMA})')
        self.ax3.axvspan(config.NOISE_MIN_FREQ, config.NOISE_MAX_FREQ, color='#dfe6e9', alpha=0.2, label=f'Search Range ({config.NOISE_MIN_FREQ}-{config.NOISE_MAX_FREQ}Hz)')
        
        self.noise_band_patches = []
        self.ax3.set_title('[Stage 3] Real-time Spectrum — Yellow: Tracked Noise Attenuation Band', fontsize=11, fontweight='bold', pad=10, color='#2d3436')
        self.ax3.set_xlim(0, self.fs / 2)
        self.ax3.set_xlabel('Frequency (Hz)')
        self.ax3.set_ylabel('Magnitude')
        self.ax3.legend(loc='upper right', framealpha=0.9, fontsize=9)
        self.ax3.grid(True, linestyle='--', alpha=0.4)
        self.ax3.set_facecolor('#ffffff')
        
        self.band_info_text = self.ax3.text(0.02, 0.88, 'Tracked Peak: Initializing...', transform=self.ax3.transAxes, fontsize=10, fontweight='bold', color='#2d3436', bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3cd', edgecolor='#ffc107', alpha=0.9))
        
        plt.tight_layout(pad=2.0)
        
    def update(self, filtered_signal, x_arr_total, M_t, clean_fft_mag, M_avg, detected_noise_bands, 
               is_recording, step_completed, duration, avg_duration, step_count, transient_detected, 
               N_t, K_t, net_events, acf_r, cadence):
        
        self.line_time_clean.set_ydata(filtered_signal)
        max_clean_amp = max(np.max(np.abs(filtered_signal)), 20.0)
        self.ax1.set_ylim(-max_clean_amp * 1.3, max_clean_amp * 1.3)
        
        if is_recording:
            self.duration_text.set_text('Step Event: [ Recording... ]')
            self.duration_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#55efc4', edgecolor='#00b894', alpha=0.9))
        elif step_completed:
            self.duration_text.set_text(f'Step Event: {duration:.3f}s (Avg: {avg_duration:.3f}s | N={step_count})')
            self.duration_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#74b9ff', edgecolor='#0984e3', alpha=0.9))
            
        if transient_detected:
            self.bypass_label.set_text('Transient Bypass: ON')
            self.bypass_label.set_color('#27ae60')
        else:
            self.bypass_label.set_text('Transient Bypass: OFF')
            self.bypass_label.set_color('#b2bec3')
            
        # 최종 경보(Alert) 조건 검증 (보고서 흐름 완벽 매핑)
        # 1. 1차 판정: 공명 전이(N-K) 횟수가 기준치 이상인가?
        if net_events >= config.ALERT_NET_EVENTS:
            # 2. 최종 판정: ACF 주기가 동물의 리듬과 일치하는가? (R값이 임계치(0.75) 이상인가?)
            if acf_r >= config.ACF_R_THRESHOLD:
                self.alert_text.set_text(f'CONFIRMED WILDLIFE! (N-K: {net_events} | ACF R={acf_r:.2f} | {cadence:.2f}s Rhythm)')
                self.alert_text.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='#d63031', edgecolor='none', alpha=0.97))
                self.fig.patch.set_facecolor('#fff0f0')
            else:
                self.alert_text.set_text(f'WARNING: IMPACTS DETECTED (N-K: {net_events}) - Verifying Rhythm...')
                self.alert_text.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='#e67e22', edgecolor='none', alpha=0.97))
                self.fig.patch.set_facecolor('#fdf2e9')
        else:
            self.alert_text.set_text('STATUS: MONITORING')
            self.alert_text.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='#00b894', edgecolor='none', alpha=0.92))
            self.fig.patch.set_facecolor('#f0f3f7')
            
        self.line_time_sr.set_ydata(np.sign(x_arr_total))
        self.sr_text.set_text(f'N(t) Total: {N_t} | K(t) Noise: {K_t} | Net Events (N-K): {net_events}')
        if net_events > 0:
            self.sr_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#a3e4d7', edgecolor='#1abc9c', alpha=0.95))
        else:
            self.sr_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#e8f8f5', edgecolor='#1abc9c', alpha=0.85))
            
        if acf_r >= config.ACF_R_THRESHOLD:
            self.acf_text.set_text(f'ACF Rhythm (R): {acf_r:.2f} (Cadence: {cadence:.2f}s)')
            self.acf_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#d7bde2', edgecolor='#8e44ad', alpha=0.95))
        else:
            self.acf_text.set_text(f'ACF Rhythm (R): {acf_r:.2f} (No periodic rhythm)')
            self.acf_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#f5eef8', edgecolor='#9b59b6', alpha=0.85))
            
        self.line_freq_raw.set_ydata(M_t)
        self.line_freq_clean.set_ydata(clean_fft_mag)
        self.line_freq_avg.set_ydata(M_avg)
        
        for p in self.noise_band_patches:
            p.remove()
        self.noise_band_patches.clear()
        
        band_info_parts = []
        for (blo, bhi, bpeak) in detected_noise_bands:
            patch = self.ax3.axvspan(blo, bhi, color='#ffeaa7', alpha=0.6)
            self.noise_band_patches.append(patch)
            band_info_parts.append(f'{bpeak:.1f}Hz (+/-{self.noise_band_width}Hz)')
            
        if band_info_parts:
            self.band_info_text.set_text('Tracked Peak: ' + ' | '.join(band_info_parts))
        else:
            self.band_info_text.set_text('Tracked Peak: None')
            
        max_mag = max(np.max(M_t), np.max(M_avg), 10.0)
        self.ax3.set_ylim(0, max_mag * 1.3)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def close(self):
        plt.ioff()
        plt.close(self.fig)
