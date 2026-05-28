# -*- coding: utf-8 -*-
"""
Wildlife Detection System — Tkinter + Matplotlib 통합 UI
"""
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rc('axes', unicode_minus=False)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
import numpy as np
import config

# ─── 색상 팔레트 ────────────────────────────────────
BG       = '#0f0f1a'
PANEL_BG = '#13131f'
CARD_BG  = '#1a1a2e'
ACCENT   = '#e94560'
TEXT     = '#e2e2e2'
MUTED    = '#636e72'
PLOT_BG  = '#090912'

MENU_BUTTONS = [
    ('1', '[환경 셋업]',     '배경 진동 측정 및 파일 저장 (30분)', '#0984e3'),
    ('2', '[파라미터 튜닝]', '발걸음 실시간 측정 → SR 최적화',    '#00b894'),
    ('3', '[파라미터 튜닝]', '저장된 파일 불러와 SR 최적화',       '#6c5ce7'),
    ('4', '[신호 녹화]',     '센서 신호를 폴더에 CSV 저장',         '#e17055'),
    ('5', '[실시간 감지]',   '설정된 파라미터로 즉시 가동',         '#fdcb6e'),
]

SENSOR_FOLDERS = {
    '1': ('Geophone',          'geophone'),
    '2': ('surround noise',    'surround_noise'),
}


class WildlifeUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Wildlife Detection System')
        self.root.configure(bg=BG)
        self.root.geometry('1280x800')
        self.root.resizable(True, True)

        self.choice = None
        self._quit = False
        self.manual_bands = []
        self.on_manual_band_change = None
        self.noise_band_patches = []

        self._build_header()
        self.content = None
        self._show_menu()

    # ══════════════════════════════════════════════
    # 헤더 (항상 표시)
    # ══════════════════════════════════════════════
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=PANEL_BG, height=52)
        hdr.pack(fill='x', side='top')
        hdr.pack_propagate(False)

        tk.Label(hdr, text='🐾 Wildlife Detection System',
                 font=('Arial', 13, 'bold'), fg=ACCENT, bg=PANEL_BG
                 ).pack(side='left', padx=18, pady=12)

        tk.Button(hdr, text=' ✕  종료 ', command=self._on_quit,
                  font=('Arial', 10, 'bold'), fg='white', bg=ACCENT,
                  relief='flat', padx=10, pady=4, cursor='hand2',
                  activebackground='#c0392b', activeforeground='white'
                  ).pack(side='right', padx=18, pady=10)

    # ══════════════════════════════════════════════
    # 콘텐츠 영역 교체
    # ══════════════════════════════════════════════
    def _clear(self):
        if self.content:
            self.content.destroy()
        self.content = tk.Frame(self.root, bg=BG)
        self.content.pack(fill='both', expand=True)

    # ══════════════════════════════════════════════
    # 메인 메뉴
    # ══════════════════════════════════════════════
    def _show_menu(self):
        self._clear()
        f = self.content

        tk.Label(f, text='시스템 가동 모드 선택',
                 font=('Arial', 20, 'bold'), fg=TEXT, bg=BG
                 ).pack(pady=(36, 4))
        tk.Label(f, text='수행할 작업을 선택하세요',
                 font=('Arial', 11), fg=MUTED, bg=BG
                 ).pack(pady=(0, 24))

        for key, title, desc, color in MENU_BUTTONS:
            outer = tk.Frame(f, bg=color)
            outer.pack(fill='x', padx=100, pady=5)

            inner = tk.Frame(outer, bg=CARD_BG, pady=14, padx=22)
            inner.pack(fill='x', padx=2, pady=2)

            row = tk.Frame(inner, bg=CARD_BG)
            row.pack(fill='x')

            num_lbl  = tk.Label(row, text=f'{key}.',
                                font=('Arial', 14, 'bold'), fg=color, bg=CARD_BG, width=3)
            ttl_lbl  = tk.Label(row, text=title,
                                font=('Arial', 12, 'bold'), fg=color, bg=CARD_BG, width=18, anchor='w')
            desc_lbl = tk.Label(row, text=desc,
                                font=('Arial', 10), fg='#b2bec3', bg=CARD_BG)
            num_lbl.pack(side='left')
            ttl_lbl.pack(side='left')
            desc_lbl.pack(side='left', padx=8)

            def _cb(k=key):
                self.choice = k
                self.root.quit()

            def _enter(e, w=inner): w.configure(bg='#252540')
            def _leave(e, w=inner): w.configure(bg=CARD_BG)

            for w in (outer, inner, row, num_lbl, ttl_lbl, desc_lbl):
                w.bind('<Button-1>', lambda e, cb=_cb: cb())
                w.bind('<Enter>', _enter)
                w.bind('<Leave>', _leave)

    def wait_for_choice(self):
        """메뉴를 보여주고 사용자 선택을 기다린다."""
        self._show_menu()
        self.choice = None
        self.root.mainloop()
        if self._quit:
            return None
        return self.choice

    # ══════════════════════════════════════════════
    # 모드 1/4: 녹화 / 배경 소음 수집 그래프 (2패널)
    # ══════════════════════════════════════════════
    def show_recording_graph(self, title, subtitle):
        self._clear()
        f = self.content

        tk.Label(f, text=title,    font=('Arial', 14, 'bold'), fg=TEXT,  bg=BG).pack(pady=(10, 2))
        tk.Label(f, text=subtitle, font=('Arial', 10),         fg=MUTED, bg=BG).pack()

        self._status = tk.Label(f, text='⏳ 준비 중...', font=('Arial', 11, 'bold'),
                                fg='#fdcb6e', bg=BG)
        self._status.pack(pady=4)

        fig = Figure(figsize=(11, 5.2), facecolor=BG)
        ax1 = fig.add_subplot(2, 1, 1, facecolor=PLOT_BG)
        ax2 = fig.add_subplot(2, 1, 2, facecolor=PLOT_BG)

        for ax in (ax1, ax2):
            ax.tick_params(colors=MUTED)
            for sp in ax.spines.values(): sp.set_color('#2d2d4e')
            ax.grid(True, linestyle='--', alpha=0.2, color='#333')
        ax1.set_title('실시간 센서 신호', color='#b2bec3', fontsize=10)
        ax1.set_ylabel('Amplitude', color=MUTED)
        
        self._rline, = ax1.plot([], [], color='#00cec9', linewidth=1.2)
            
        self._hline, = ax2.plot([], [], color='#a29bfe', linewidth=1.5)
        ax2.set_title('진폭 분포 (Histogram)', color='#b2bec3', fontsize=10)
        fig.tight_layout(pad=2.0)

        self._rec_ax1, self._rec_ax2 = ax1, ax2
        cv = FigureCanvasTkAgg(fig, master=f)
        cv.draw()
        cv.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=6)
        self._rec_canvas = cv

    @staticmethod
    def _remove_powerline_noise(signal_arr, fs=100.0):
        """
        적응형 FFT 노치 필터 — 지배적 고주파 간섭을 자동으로 찾아 제거.
        발걸음 신호 대역(1~25Hz)은 완전히 보존. 저장 데이터에는 영향 없음.
        """
        n = len(signal_arr)
        if n < 30:
            return signal_arr

        fft_vals = np.fft.rfft(signal_arr)
        freqs    = np.fft.rfftfreq(n, 1.0 / fs)
        mags     = np.abs(fft_vals)

        # 발걸음 저주파 대역(1~20Hz) 평균 파워
        lo_mask = (freqs >= 1.0) & (freqs <= 20.0)
        lo_mean = np.mean(mags[lo_mask]) if lo_mask.any() else 1.0

        # 고주파 간섭 대역(25Hz 이상)에서 지배적 피크 자동 탐색
        hi_mask = freqs >= 25.0
        if hi_mask.any():
            hi_mags = mags.copy()
            hi_mags[~hi_mask] = 0.0
            peak_idx  = np.argmax(hi_mags)
            peak_freq = freqs[peak_idx]
            peak_pow  = mags[peak_idx]

            # 저주파 평균보다 3배 이상 강한 피크가 있으면 → 노치 적용
            if peak_pow > lo_mean * 3.0:
                bw = 6.0  # 노치 대역폭 ±6Hz
                notch = (freqs >= peak_freq - bw) & (freqs <= peak_freq + bw)
                fft_vals[notch] = 0.0

        return np.fft.irfft(fft_vals, n)

    def update_recording_graph(self, samples, status_text, color='#fdcb6e'):
        if not hasattr(self, '_rec_canvas'): return
        try:
            arr     = np.array(samples[-800:], dtype=float)
            centered = arr - np.mean(arr) if len(arr) > 0 else arr

            # 전력선 노이즈(60Hz → 앨리어싱 40Hz)만 제거, 나머지 신호 보존
            cleaned = self._remove_powerline_noise(centered, fs=config.FS) \
                      if len(centered) > 30 else centered

            self._rline.set_data(np.arange(len(cleaned)), cleaned)

            # y축 범위: 최소 -30 ~ 30
            self._rec_ax1.relim()
            self._rec_ax1.autoscale_view()
            ymin, ymax = self._rec_ax1.get_ylim()
            if (ymax - ymin) < 60:
                self._rec_ax1.set_ylim(-30, 30)

            if len(cleaned) > 20:
                counts, edges = np.histogram(cleaned, bins=40)
                centers = (edges[:-1] + edges[1:]) / 2
                self._hline.set_data(centers, counts)
                self._rec_ax2.relim()
                self._rec_ax2.autoscale_view()
                xmin, xmax = self._rec_ax2.get_xlim()
                if (xmax - xmin) < 60:
                    self._rec_ax2.set_xlim(-30, 30)

            self._status.configure(text=status_text, fg=color)
            self._rec_canvas.draw()
            self.root.update()
        except Exception:
            pass

    # ══════════════════════════════════════════════
    # 모드 2/3: 파라미터 최적화 그래프 (3패널 - 실시간 감지형 레이아웃)
    # ══════════════════════════════════════════════
    def show_optimization_graph(self, env_noise, animal_signal=None):
        self._clear()
        f = self.content

        tk.Label(f, text='📊 SR 파라미터 최적화 및 신호 검증 대시보드', font=('Arial', 14, 'bold'), fg=TEXT, bg=BG).pack(pady=(10, 2))
        self._opt_status = tk.Label(f, text='⏳ 준비 중...', font=('Arial', 11, 'bold'), fg='#fdcb6e', bg=BG)
        self._opt_status.pack(pady=4)

        self.fs          = config.FS
        self.buffer_size = config.BUFFER_SIZE
        self.half_n      = self.buffer_size // 2
        self.x_time      = np.arange(self.buffer_size)
        self.x_freq      = np.fft.fftfreq(self.buffer_size, 1/self.fs)[:self.half_n]

        fig = Figure(figsize=(11.5, 6.5), facecolor=BG)
        ax1 = fig.add_subplot(3, 1, 1, facecolor=PLOT_BG)
        ax2 = fig.add_subplot(3, 1, 2, facecolor=PLOT_BG)
        ax3 = fig.add_subplot(3, 1, 3, facecolor=PLOT_BG)
        self._opt_axes = (ax1, ax2, ax3)

        def _style(ax):
            ax.tick_params(colors=MUTED)
            for sp in ax.spines.values(): sp.set_color('#2d2d4e')
            ax.grid(True, linestyle='--', alpha=0.2, color='#333')

        # Subplot 1: Stage 1 (Filtered Output)
        _style(ax1)
        self._opt_line_raw, = ax1.plot(self.x_time, np.zeros(self.buffer_size),
                                       color='#e2e2e2', linewidth=1.3, alpha=0.8, label='Raw Input')
        self._opt_line_clean, = ax1.plot(self.x_time, np.zeros(self.buffer_size),
                                         color='#0984e3', linewidth=1.6, label='SDFT Filtered')
        ax1.set_title('[Stage 1] SDFT Adaptive Filter — Raw vs. Denoised Signal', fontsize=9, fontweight='bold', color='#b2bec3')
        ax1.set_xlim(0, self.buffer_size - 1)
        ax1.set_ylabel('Amplitude', color=MUTED)
        ax1.legend(loc='upper right', fontsize=8, framealpha=0.4)

        # Subplot 2: Stage 2 (Stochastic Resonance & ACF)
        _style(ax2)
        self._opt_line_sr, = ax2.plot(self.x_time, np.zeros(self.buffer_size),
                                      color='#a29bfe', linewidth=1.5, label='Bistable Binary State: sgn(x(t))')
        ax2.axhline(0.0,  color='#d63031', linestyle='--', linewidth=1.5)
        ax2.axhline( 1.0, color='#27ae60', linestyle=':',  linewidth=1.2)
        ax2.axhline(-1.0, color='#27ae60', linestyle=':',  linewidth=1.2)
        ax2.set_title('[Stage 2] Stochastic Resonance & ACF — N(t): Total Crossings / K(t): Noise Baseline / N-K: Pure Impulses (R >= 0.75)',
                      fontsize=9, fontweight='bold', color='#b2bec3')
        ax2.set_xlim(0, self.buffer_size - 1); ax2.set_ylim(-2.2, 2.2)
        ax2.set_ylabel('State', color=MUTED)
        ax2.legend(loc='upper right', fontsize=8, framealpha=0.4)

        self._opt_sr_text = ax2.text(
            0.02, 0.88, 'N(t): - | K(t): - | Net Events: Calculating...', transform=ax2.transAxes,
            fontsize=8, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f8f5', edgecolor='#1abc9c', alpha=0.9))
        self._opt_acf_text = ax2.text(
            0.02, 0.72, 'ACF Rhythm (R): Calculating...', transform=ax2.transAxes,
            fontsize=8, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5eef8', edgecolor='#9b59b6', alpha=0.9))

        # Subplot 3: Stage 3 (Real-time Spectrum)
        _style(ax3)
        self._opt_line_freq_raw, = ax3.plot(self.x_freq, np.zeros(self.half_n),
                                            color='#636e72', linewidth=1.2, label='Raw Spectrum', alpha=0.6)
        self._opt_line_freq_clean, = ax3.plot(self.x_freq, np.zeros(self.half_n),
                                              color='#d63031', linewidth=1.8, label='Filtered Spectrum')
        self._opt_line_freq_avg, = ax3.plot(self.x_freq, np.zeros(self.half_n),
                                            color='#e67e22', linestyle=':', linewidth=1.8, label='Long-term Avg')
        ax3.set_title('[Stage 3] Real-time Spectrum — Yellow: Tracked Noise Attenuation Band',
                      fontsize=9, fontweight='bold', color='#b2bec3')
        ax3.set_xlim(0, 50.0)
        ax3.set_xlabel('Frequency (Hz)', color=MUTED); ax3.set_ylabel('Magnitude', color=MUTED)
        ax3.legend(loc='upper right', fontsize=8, framealpha=0.4)

        self._opt_peak_text = ax3.text(
            0.02, 0.88, 'Tracked Peak: Calculating...', transform=ax3.transAxes,
            fontsize=8, fontweight='bold', color='#7f8c8d',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fcf3cf', edgecolor='#f1c40f', alpha=0.9))

        self._opt_noise_span = None

        fig.tight_layout(pad=1.5)
        cv = FigureCanvasTkAgg(fig, master=f)
        cv.draw()
        cv.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=6)
        self._opt_canvas = cv

    def update_optimization_graph(self, status, raw_signal=None, filtered_signal=None, 
                                   sr_signal=None, N_t=0, K_t=0, net_events=0, 
                                   acf_r=0.0, cadence=0.0, raw_fft=None, clean_fft=None, 
                                   avg_fft=None, tracked_peak=0.0, noise_bands=None, color='#fdcb6e'):
        if not hasattr(self, '_opt_canvas'): return
        try:
            self._opt_status.configure(text=status, fg=color)
            ax1, ax2, ax3 = self._opt_axes

            # 1단계 그리기 (Raw & SDFT Filtered)
            if raw_signal is not None:
                n_len = min(len(raw_signal), self.buffer_size)
                self._opt_line_raw.set_data(np.arange(n_len), raw_signal[-n_len:])
            if filtered_signal is not None:
                n_len = min(len(filtered_signal), self.buffer_size)
                self._opt_line_clean.set_data(np.arange(n_len), filtered_signal[-n_len:])
            else:
                # 필터 신호가 없을 때는 0으로 고정된 파란 선이 Raw 신호를 덮지 않도록 숨김
                self._opt_line_clean.set_data([], [])
            ax1.relim(); ax1.autoscale_view()
            ax1.set_xlim(0, self.buffer_size - 1)
            
            # y축 범위가 너무 극도로 축소되어 미세 진동이 일자로 보이지 않도록 최소 -30 ~ 30 범위 보장
            ymin, ymax = ax1.get_ylim()
            if (ymax - ymin) < 60:
                ax1.set_ylim(-30, 30)

            # 2단계 그리기 (Stochastic Resonance & ACF)
            if sr_signal is not None:
                n_len = min(len(sr_signal), self.buffer_size)
                self._opt_line_sr.set_data(np.arange(n_len), sr_signal[:n_len])
                
                # 텍스트 박스 업데이트
                self._opt_sr_text.configure(
                    text=f'N(t) Total: {N_t} | K(t) Noise: {K_t} | Net Events (N-K): {net_events}')
                
                acf_status = "No periodic rhythm" if acf_r < config.ACF_R_THRESHOLD else f"Rhythm detected! Cadence={cadence:.2f}s"
                self._opt_acf_text.configure(
                    text=f'ACF Rhythm (R): {acf_r:.2f} ({acf_status})')

            # 3단계 그리기 (Spectrum)
            if raw_fft is not None:
                self._opt_line_freq_raw.set_data(self.x_freq, raw_fft[:self.half_n])
            if clean_fft is not None:
                self._opt_line_freq_clean.set_data(self.x_freq, clean_fft[:self.half_n])
            if avg_fft is not None:
                self._opt_line_freq_avg.set_data(self.x_freq, avg_fft[:self.half_n])
            ax3.relim(); ax3.autoscale_view()
            ax3.set_xlim(0, 50.0)

            # 노이즈 피크 텍스트 박스 업데이트
            if tracked_peak > 0:
                self._opt_peak_text.configure(
                    text=f'Tracked Peak: {tracked_peak:.1f}Hz (+/-{config.NOISE_BAND_WIDTH/2:.1f}Hz)')
            else:
                self._opt_peak_text.configure(text='Tracked Peak: -- Hz')
            
            # 노이즈 감쇄 밴드 음영 표시
            if noise_bands:
                if getattr(self, '_opt_noise_span', None) is not None:
                    try: self._opt_noise_span.remove()
                    except: pass
                # 노이즈 밴드 중 첫 번째 밴드를 하이라이트 표시
                b = noise_bands[0]
                self._opt_noise_span = ax3.axvspan(b[0], b[1], color='#f1c40f', alpha=0.15, label='Tracked Noise Band')

            self._opt_canvas.draw()
            self.root.update()
        except Exception:
            pass

    def setup_live_detection(self, on_manual_band_change=None):
        self.on_manual_band_change = on_manual_band_change
        self.manual_bands = []
        self._clear()
        f = self.content

        self.fs          = config.FS
        self.buffer_size = config.BUFFER_SIZE
        self.half_n      = self.buffer_size // 2
        self.x_time      = np.arange(self.buffer_size)
        self.x_freq      = np.fft.fftfreq(self.buffer_size, 1/self.fs)[:self.half_n]
        self.noise_band_width = config.NOISE_BAND_WIDTH

        fig = Figure(figsize=(12, 7.8), facecolor=BG)
        ax1 = fig.add_subplot(3, 1, 1, facecolor=PLOT_BG)
        ax2 = fig.add_subplot(3, 1, 2, facecolor=PLOT_BG)
        ax3 = fig.add_subplot(3, 1, 3, facecolor=PLOT_BG)
        self.ax1, self.ax2, self.ax3 = ax1, ax2, ax3

        def _style(ax):
            ax.tick_params(colors=MUTED)
            for sp in ax.spines.values(): sp.set_color('#2d2d4e')
            ax.grid(True, linestyle='--', alpha=0.25, color='#333')

        # Stage 1
        _style(ax1)
        self.line_time_clean_geo, = ax1.plot(self.x_time, np.zeros(self.buffer_size),
                                              color='#00cec9', linewidth=1.5, label='Geophone (A0)')
        ax1.set_title('[Stage 1] SDFT Adaptive Filter — Noise Suppressed, Footsteps Preserved',
                      fontsize=10, fontweight='bold', color='#b2bec3')
        ax1.set_xlim(0, self.buffer_size - 1)
        ax1.set_xlabel(f'Sample Index (Buffer: {self.buffer_size})', color=MUTED)
        ax1.set_ylabel('Amplitude', color=MUTED)
        ax1.legend(loc='upper right', fontsize=8, framealpha=0.4)

        self.alert_text = ax1.text(
            0.5, 1.13, 'STATUS: MONITORING GEOPHONE', transform=ax1.transAxes,
            fontsize=10, fontweight='bold', color='white', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#00b894', edgecolor='none', alpha=0.92))
        self.duration_text = ax1.text(
            0.02, 0.88, 'Step Event: Waiting...', transform=ax1.transAxes,
            fontsize=9, fontweight='bold', color='#2d3436',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffeaa7', edgecolor='#fdcb6e', alpha=0.9))
        self.bypass_label = ax1.text(
            0.98, 0.12, 'Transient Geo: OFF', transform=ax1.transAxes,
            fontsize=8, fontweight='bold', color='#b2bec3', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL_BG, alpha=0.8, edgecolor='#333'))

        # Stage 2
        _style(ax2)
        self.line_time_sr_geo, = ax2.plot(self.x_time, np.zeros(self.buffer_size),
                                           color='#00cec9', linewidth=1.3, label='Geophone (A0) State')
        ax2.axhline(0.0,  color='#636e72', linestyle='--', linewidth=1.0)
        ax2.axhline( 1.0, color='#27ae60', linestyle=':',  linewidth=1.2)
        ax2.axhline(-1.0, color='#27ae60', linestyle=':',  linewidth=1.2)
        ax2.set_title(f'[Stage 2] Stochastic Resonance & ACF  (R>={config.ACF_R_THRESHOLD})',
                      fontsize=10, fontweight='bold', color='#b2bec3')
        ax2.set_xlim(0, self.buffer_size - 1); ax2.set_ylim(-2.2, 2.2)
        ax2.set_xlabel('Sample Index', color=MUTED); ax2.set_ylabel('State', color=MUTED)
        ax2.legend(loc='upper right', fontsize=8, framealpha=0.4)

        self.sr_text = ax2.text(
            0.02, 0.88, 'Geo N(t): - | Net Events: Waiting...', transform=ax2.transAxes,
            fontsize=9, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f8f5', edgecolor='#1abc9c', alpha=0.9))
        self.acf_text = ax2.text(
            0.02, 0.72, 'ACF Rhythm (R): Waiting...', transform=ax2.transAxes,
            fontsize=9, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5eef8', edgecolor='#9b59b6', alpha=0.9))

        # Stage 3
        _style(ax3)
        self.line_freq_raw_geo,   = ax3.plot(self.x_freq, np.zeros(self.half_n),
                                               color='#00cec9', linewidth=1.0, label='Geo FFT', alpha=0.4)
        self.line_freq_clean_geo, = ax3.plot(self.x_freq, np.zeros(self.half_n),
                                               color='#00b894', linewidth=1.6, label='Geo Filtered')
        self.line_freq_avg,   = ax3.plot(self.x_freq, np.zeros(self.half_n),
                                           color='#e67e22', linestyle=':', linewidth=1.8, label='Geo Long-term Avg')
        ax3.axvspan(config.NOISE_MIN_FREQ, config.NOISE_MAX_FREQ, color='#dfe6e9', alpha=0.06)
        self.noise_band_patches = []
        ax3.set_title('[Stage 3] Real-time Spectrum — Noise Attenuation Bands',
                      fontsize=10, fontweight='bold', color='#b2bec3')
        ax3.set_xlim(0, self.fs / 2)
        ax3.set_xlabel('Frequency (Hz)', color=MUTED); ax3.set_ylabel('Magnitude', color=MUTED)
        ax3.legend(loc='upper right', fontsize=8, framealpha=0.4)

        self.band_info_text = ax3.text(
            0.02, 0.88, 'Tracked Peak: Initializing...', transform=ax3.transAxes,
            fontsize=9, fontweight='bold', color='#2d3436',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3cd', edgecolor='#ffc107', alpha=0.9))

        fig.tight_layout(pad=2.5)

        cv = FigureCanvasTkAgg(fig, master=f)
        cv.draw()
        cv.get_tk_widget().pack(fill='both', expand=True)
        self.live_canvas = cv
        self.live_fig    = fig

        self.span = SpanSelector(
            ax3, self._onselect, 'horizontal', useblit=True,
            props=dict(alpha=0.5, facecolor='#fab1a0'),
            interactive=True, drag_from_anywhere=True)
        fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.root.update()

    def _onselect(self, xmin, xmax):
        if abs(xmax - xmin) < 0.2: return
        self.manual_bands.append((xmin, xmax))
        if self.on_manual_band_change:
            self.on_manual_band_change(self.manual_bands)

    def _on_key(self, event):
        if event.key == 'c':
            self.manual_bands.clear()
            if self.on_manual_band_change:
                self.on_manual_band_change(self.manual_bands)

    def update(self, filtered_signal_geo,
               x_arr_total_geo,
               M_t_geo,
               clean_fft_mag_geo,
               M_avg_geo,
               detected_noise_bands_geo,
               is_recording_geo,
               step_completed_geo,
               duration_geo,
               avg_duration_geo,
               step_count_geo,
               transient_detected_geo,
               N_t_geo, K_t_geo, net_events_geo, acf_r_geo, cadence_geo):

        self.line_time_clean_geo.set_ydata(filtered_signal_geo)
        mx_geo = np.max(np.abs(filtered_signal_geo))
        mx = max(mx_geo, 20.0)
        self.ax1.set_ylim(-mx * 1.3, mx * 1.3)

        # Step Event Text
        geo_status = "Rec..." if is_recording_geo else (f"{duration_geo:.2f}s" if step_completed_geo else "Wait")
        self.duration_text.set_text(
            f'Step Event | Geo: {geo_status} (Avg: {avg_duration_geo:.2f}s, N={step_count_geo})'
        )

        if is_recording_geo:
            self.duration_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#55efc4', edgecolor='#00b894', alpha=0.9))
        elif step_completed_geo:
            self.duration_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#74b9ff', edgecolor='#0984e3', alpha=0.9))
        else:
            self.duration_text.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#ffeaa7', edgecolor='#fdcb6e', alpha=0.9))

        self.bypass_label.set_text(f'Transient Geo: {"ON" if transient_detected_geo else "OFF"}')
        self.bypass_label.set_color('#27ae60' if transient_detected_geo else '#b2bec3')

        # Alarm triggering via logical-OR
        is_wildlife_geo = (net_events_geo >= config.ALERT_NET_EVENTS) and (acf_r_geo >= config.ACF_R_THRESHOLD)
        is_warning_geo = (net_events_geo >= config.ALERT_NET_EVENTS)

        if is_wildlife_geo:
            self.alert_text.set_text(f'🚨 GEOPHONE CONFIRMED WILDLIFE! (Geo: R={acf_r_geo:.2f})')
            self.alert_text.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='#d63031', edgecolor='none', alpha=0.97))
        elif is_warning_geo:
            self.alert_text.set_text(f'⚠️ WARNING: IMPACTS DETECTED (Geo) - Verifying Rhythm...')
            self.alert_text.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='#e67e22', edgecolor='none', alpha=0.97))
        else:
            self.alert_text.set_text('📡 STATUS: MONITORING GEOPHONE')
            self.alert_text.set_bbox(dict(boxstyle='round,pad=0.5', facecolor='#00b894', edgecolor='none', alpha=0.92))

        # SR potential plots
        self.line_time_sr_geo.set_ydata(np.sign(x_arr_total_geo))

        self.sr_text.set_text(f'Geo N-K: {net_events_geo} (N:{N_t_geo}, K:{K_t_geo})')
        self.sr_text.set_bbox(dict(boxstyle='round,pad=0.4',
                                   facecolor='#a3e4d7' if net_events_geo > 0 else '#e8f8f5',
                                   edgecolor='#1abc9c', alpha=0.9))

        geo_acf_status = f"Geo R:{acf_r_geo:.2f}" + (f" ({cadence_geo:.2f}s)" if acf_r_geo >= config.ACF_R_THRESHOLD else "")
        self.acf_text.set_text(f'ACF Rhythm | {geo_acf_status}')
        self.acf_text.set_bbox(dict(boxstyle='round,pad=0.4',
                                    facecolor='#d7bde2' if acf_r_geo >= config.ACF_R_THRESHOLD else '#f5eef8',
                                    edgecolor='#8e44ad', alpha=0.9))

        self.line_freq_raw_geo.set_ydata(M_t_geo)
        self.line_freq_clean_geo.set_ydata(clean_fft_mag_geo)
        self.line_freq_avg.set_ydata(M_avg_geo)

        for p in self.noise_band_patches: p.remove()
        self.noise_band_patches.clear()
        parts = []
        for (blo, bhi, bpeak) in detected_noise_bands_geo:
            self.noise_band_patches.append(self.ax3.axvspan(blo, bhi, color='#00cec9', alpha=0.2))
            parts.append(f'Geo:{bpeak:.1f}Hz')
        self.band_info_text.set_text('Tracked Peak: ' + (', '.join(parts) if parts else 'None'))

        mx_mag = max(np.max(M_t_geo), np.max(M_avg_geo), 10.0)
        self.ax3.set_ylim(0, mx_mag * 1.3)

        self.live_canvas.draw()
        self.live_canvas.flush_events()
        self.root.update()

    # ── 공통 유틸 ──────────────────────────────────
    def is_alive(self):
        try:
            return self.root.winfo_exists() and not self._quit
        except Exception:
            return False

    def _on_quit(self):
        self._quit = True
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def close(self):
        self._on_quit()
