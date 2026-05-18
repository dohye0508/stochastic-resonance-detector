# -*- coding: utf-8 -*-
"""
[보고서 4.2 실측 데이터 수집] 직접 측정한 동물 발자국 신호 기록 스크립트 (Tkinter GUI 듀얼 채널 수집기) - 듀얼 채널 백업본
================================================================================
GUI 창에서 동물의 이름을 적고 [시작] 버튼을 누른 후, [중지 및 저장] 버튼을 누르면
원하는 시간만큼 지오폰과 마이크 데이터를 정확하게 녹음하고 저장할 수 있습니다.
================================================================================
"""
import tkinter as tk
from tkinter import messagebox
import threading
import serial
import time
import numpy as np
import csv
import sounddevice as sd
import config

class StepRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("야생동물 발걸음 측정기 (듀얼 채널) - 백업본")
        self.root.geometry("450x380")
        self.root.resizable(False, False)
        
        # 테마 색상 (Sleek Dark Mode)
        self.bg_color = "#1e1e2e"
        self.card_color = "#252538"
        self.text_color = "#cdd6f4"
        self.accent_green = "#a6e3a1"
        self.accent_red = "#f38ba8"
        self.accent_blue = "#89b4fa"
        
        self.root.configure(bg=self.bg_color)
        
        # 상태 변수
        self.is_recording = False
        self.geo_values = []
        self.audio_rec = None
        self.start_time = None
        self.ser = None
        self.record_thread = None
        
        self._build_ui()

    def _build_ui(self):
        # 타이틀 영역
        title_frame = tk.Frame(self.root, bg=self.bg_color, pady=15)
        title_frame.pack(fill="x")
        
        title_lbl = tk.Label(
            title_frame, 
            text="🐾 야생동물 발걸음 측정 시스템 🐾", 
            font=("Segoe UI", 16, "bold"), 
            bg=self.bg_color, 
            fg=self.accent_blue
        )
        title_lbl.pack()
        
        subtitle_lbl = tk.Label(
            title_frame, 
            text="지오폰(진동) & 마이크(음향) 실시간 동시 측정", 
            font=("Segoe UI", 10), 
            bg=self.bg_color, 
            fg="#a6adc8"
        )
        subtitle_lbl.pack()

        # 입력/설정 카드
        card_frame = tk.Frame(self.root, bg=self.card_color, bd=0, padx=20, pady=20)
        card_frame.pack(fill="both", expand=True, padx=25, pady=5)
        
        # 동물 이름 입력
        name_lbl = tk.Label(
            card_frame, 
            text="측정할 동물/대상 이름:", 
            font=("Segoe UI", 11, "bold"), 
            bg=self.card_color, 
            fg=self.text_color
        )
        name_lbl.grid(row=0, column=0, sticky="w", pady=5)
        
        self.name_entry = tk.Entry(
            card_frame, 
            font=("Segoe UI", 11), 
            bg=self.bg_color, 
            fg=self.text_color, 
            insertbackground=self.text_color,
            relief="flat",
            width=18
        )
        self.name_entry.insert(0, "animal")
        self.name_entry.grid(row=0, column=1, sticky="w", padx=10, pady=5)
        
        # 상태 표시기
        self.status_lbl = tk.Label(
            card_frame, 
            text="● 대기 중", 
            font=("Segoe UI", 13, "bold"), 
            bg=self.card_color, 
            fg="#94e2d5"
        )
        self.status_lbl.grid(row=1, column=0, columnspan=2, pady=20)
        
        # 시간 표시기
        self.time_lbl = tk.Label(
            card_frame, 
            text="0.0초 녹음됨", 
            font=("Segoe UI", 11), 
            bg=self.card_color, 
            fg="#a6adc8"
        )
        self.time_lbl.grid(row=2, column=0, columnspan=2, pady=2)

        # 버튼 영역
        btn_frame = tk.Frame(self.root, bg=self.bg_color, pady=15)
        btn_frame.pack(fill="x")
        
        self.start_btn = tk.Button(
            btn_frame, 
            text="측정 시작 (Start)", 
            font=("Segoe UI", 11, "bold"),
            bg=self.accent_green, 
            fg="#11111b",
            activebackground="#cba6f7",
            relief="flat",
            padx=15, 
            pady=8,
            command=self.start_recording
        )
        self.start_btn.pack(side="left", expand=True, padx=20)
        
        self.stop_btn = tk.Button(
            btn_frame, 
            text="중지 및 저장 (Stop)", 
            font=("Segoe UI", 11, "bold"),
            bg=self.accent_red, 
            fg="#11111b",
            activebackground="#f9e2af",
            relief="flat",
            padx=15, 
            pady=8,
            state="disabled",
            command=self.stop_recording
        )
        self.stop_btn.pack(side="right", expand=True, padx=20)

    def start_recording(self):
        if self.is_recording:
            return
            
        self.is_recording = True
        self.geo_values = []
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.name_entry.config(state="disabled")
        
        self.status_lbl.config(text="● 녹음 진행 중...", fg=self.accent_red)
        
        # 레코딩 루프 스레드 기동
        self.record_thread = threading.Thread(target=self._recording_worker, daemon=True)
        self.record_thread.start()

    def _recording_worker(self):
        # 1. 마이크 녹음 시작 (충분히 긴 5분 한계 설정)
        fs_audio = 16000
        max_duration = 300
        try:
            self.audio_rec = sd.rec(int(max_duration * fs_audio), samplerate=fs_audio, channels=1, dtype='float32')
        except Exception as e:
            self.root.after(0, lambda: self._handle_error(f"마이크 장치 오류: {e}"))
            return

        # 2. 지오폰 시리얼 연결
        try:
            self.ser = serial.Serial(config.PORT, config.BAUD_RATE, timeout=0.1)
        except Exception as e:
            sd.stop()
            self.root.after(0, lambda: self._handle_error(f"지오폰 시리얼 연결 오류 ({config.PORT}): {e}"))
            return

        self.start_time = time.time()
        
        # 3. 실시간 동시 수집 루프
        while self.is_recording:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    try:
                        val = int(line)
                        self.geo_values.append(val)
                    except ValueError:
                        pass
                        
            # UI 타이머 및 진행 표시 업데이트
            elapsed = time.time() - self.start_time
            self.root.after(0, lambda t=elapsed: self.time_lbl.config(text=f"{t:.1f}초 녹음됨"))
            time.sleep(0.001)

    def stop_recording(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        duration = time.time() - self.start_time
        
        # 하드웨어 종료
        sd.stop()
        if self.ser:
            self.ser.close()
            
        self.status_lbl.config(text="● 데이터 분석 및 저장 중...", fg=self.accent_blue)
        self.root.update()
        
        # 데이터 정규화 및 저장
        self._process_and_save(duration)
        
        # UI 리셋
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.name_entry.config(state="normal")
        self.status_lbl.config(text="● 대기 중", fg="#94e2d5")

    def _process_and_save(self, duration):
        label = self.name_entry.get().strip() or "animal"
        fs_audio = 16000
        downsample_factor = int(fs_audio / config.FS)
        target_count = int(duration * config.FS)
        
        if target_count < 10:
            messagebox.showwarning("경고", "녹음 시간이 너무 짧습니다 (최소 0.1초 필요).")
            return

        # 1. 오디오 처리
        audio_flat = self.audio_rec.flatten()
        required_audio_samples = target_count * downsample_factor
        
        if len(audio_flat) > required_audio_samples:
            audio_flat = audio_flat[:required_audio_samples]
        elif len(audio_flat) < required_audio_samples:
            audio_flat = np.pad(audio_flat, (0, required_audio_samples - len(audio_flat)))
            
        audio_downsampled = np.mean(audio_flat.reshape(-1, downsample_factor), axis=1)
        mic_centered = audio_downsampled * 512.0
        mic_raw = mic_centered + config.CURRENT_ZERO

        # 2. 지오폰 처리
        geo_data = np.array(self.geo_values)
        if len(geo_data) > target_count:
            geo_data = geo_data[:target_count]
        elif len(geo_data) < target_count:
            geo_data = np.pad(geo_data, (0, target_count - len(geo_data)), constant_values=config.CURRENT_ZERO)
            
        geo_centered = geo_data - config.CURRENT_ZERO

        # 3. CSV 저장
        timestamp = int(time.time())
        geo_filename = f"animal_step_geophone_{label}_{timestamp}.csv"
        mic_filename = f"animal_step_microphone_{label}_{timestamp}.csv"
        
        try:
            # 지오폰 저장
            with open(geo_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["SampleIndex", "RawValue", "CenteredValue"])
                for i in range(target_count):
                    writer.writerow([i, int(geo_data[i]), int(geo_centered[i])])
                    
            # 마이크 저장
            with open(mic_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["SampleIndex", "RawValue", "CenteredValue"])
                for i in range(target_count):
                    writer.writerow([i, int(mic_raw[i]), int(mic_centered[i])])
                    
            messagebox.showinfo(
                "성공", 
                f"데이터 수집 및 저장이 성공적으로 완료되었습니다!\n\n"
                f"📂 지오폰 파일:\n{geo_filename}\n\n"
                f"📂 마이크 파일:\n{mic_filename}\n\n"
                f"⏱️ 총 기록 시간: {duration:.1f}초 ({target_count} 샘플)"
            )
        except Exception as e:
            messagebox.showerror("오류", f"데이터 저장에 실패했습니다: {e}")

    def _handle_error(self, err_msg):
        self.is_recording = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.name_entry.config(state="normal")
        self.status_lbl.config(text="● 대기 중", fg="#94e2d5")
        messagebox.showerror("하드웨어 에러", err_msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = StepRecorderGUI(root)
    root.mainloop()
