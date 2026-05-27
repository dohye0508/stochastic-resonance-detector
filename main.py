# -*- coding: utf-8 -*-
"""
야생동물 실시간 감지 시스템 | 메인 오케스트레이터 (Main Orchestrator)
보고서의 제 2장, 3장, 4장을 통합하여 실행하는 마스터 스크립트입니다.
기존 test.py를 대체하며, 단원별로 쪼개진 모듈들을 조립해 실시간 파이프라인을 가동합니다.
"""
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import config

# 각 장(Chapter)별 핵심 모듈 불러오기
from step_analyzer import StepDurationAnalyzer
from sdft_filter import SDFTAdaptiveFilter
from msr_engine import BistableDoubleWellEngine
from acf_analyzer import ACFPeriodicityAnalyzer
from ui_manager import WildlifeUI

import os
import glob
import csv
import time as _time_module

def parse_serial_line(ln, channel):
    """아두이노가 보낸 시리얼 데이터(단일값 또는 쉼표 구분값)에서 지정된 채널 값을 추출."""
    if ',' in ln:
        parts = ln.split(',')
        if len(parts) > channel:
            try: return int(parts[channel])
            except ValueError: pass
        try: return int(parts[0])
        except ValueError: return None
    try: return int(ln)
    except ValueError: return None


def auto_calibrate_zero(ser, channel):
    """2초간 센서 신호를 읽어 실제 영점(DC Bias)을 자동으로 계산합니다."""
    print(f"\n⏳ 센서 영점(Baseline) 자동 보정 중... (아두이노 채널 {channel}번)")
    samples = []
    start = _time_module.time()
    ser.reset_input_buffer()
    while _time_module.time() - start < 2.0:
        if ser.in_waiting > 0:
            ln = ser.readline().decode('utf-8', errors='ignore').strip()
            if ln:
                val = parse_serial_line(ln, channel)
                if val is not None:
                    samples.append(val)
        _time_module.sleep(0.001)
    if len(samples) > 20:
        zero = float(np.mean(samples))
        print(f"✅ 영점 보정 완료: {zero:.2f} (기본에서 최적화됨)")
        return zero
    
    # 캘리브레이션 실패 시 센서 채널별 물리적 기본 바이어스로 안전하게 후퇴(Fallback)
    default_zero = 250.0 if channel == 1 else 512.0
    print(f"⚠️ 영점 보정 실패 (데이터 부족) — 센서 채널 {channel}번 기본값 {default_zero:.1f}을 사용합니다.")
    return default_zero


def auto_calibrate_dual(ser):
    """2초간 지오폰(A0)과 마이크(A1) 신호를 동시에 읽어 각각의 영점(DC Bias)을 계산합니다."""
    print("\n⏳ 듀얼 채널(지오폰 및 마이크) 영점 자동 보정 중...")
    samples_geo = []
    samples_mic = []
    start = _time_module.time()
    ser.reset_input_buffer()
    while _time_module.time() - start < 2.0:
        if ser.in_waiting > 0:
            ln = ser.readline().decode('utf-8', errors='ignore').strip()
            if ln and ',' in ln:
                parts = ln.split(',')
                if len(parts) >= 2:
                    try:
                        val0 = int(parts[0])
                        val1 = int(parts[1])
                        samples_geo.append(val0)
                        samples_mic.append(val1)
                    except ValueError: pass
        _time_module.sleep(0.001)
    
    zero_geo = float(np.mean(samples_geo)) if len(samples_geo) > 20 else 512.0
    zero_mic = float(np.mean(samples_mic)) if len(samples_mic) > 20 else 250.0
    print(f"✅ 듀얼 영점 보정 완료: 지오폰={zero_geo:.1f}, 마이크={zero_mic:.1f}")
    return zero_geo, zero_mic



# 센서 종류 → 저장 폴더 매핑
SENSOR_FOLDERS = {
    '1': ('Geophone',          'geophone'),
    '2': ('MEMS microphone',   'microphone'),
    '3': ('surround noise',    'surround_noise'),
    '4': ('Dual channel',      'dual_sensor'),
}

def record_sensor_data(ser, label=None):
    """센서 신호를 선택한 폴더에 CSV로 저장."""
    print("\n" + "="*50)
    print("📡 센서 신호 녹화")
    print("  저장할 센서/폴더를 선택하세요:")
    for k, (folder, _) in SENSOR_FOLDERS.items():
        print(f"  {k}. {folder}")
    print("  0. 취소")
    sel = input("👉 선택: ").strip()
    if sel not in SENSOR_FOLDERS:
        print("취소합니다.")
        return

    folder_name, prefix = SENSOR_FOLDERS[sel]
    os.makedirs(folder_name, exist_ok=True)

    if label is None:
        label = input("📝 파일 레이블 (예: animal, noise, test, 기본값=raw): ").strip() or 'raw'

    dur_str = input("⏱️  녹화 시간(초, 기본값=30): ").strip()
    try:
        duration = float(dur_str) if dur_str else 30.0
    except ValueError:
        duration = 30.0

    timestamp = int(_time_module.time())
    filename = os.path.join(folder_name, f"{prefix}_{label}_{timestamp}.csv")

    print(f"\n🔴 녹화 시작! ({duration:.0f}초, 저장 경로: {filename})")
    samples = []
    t_start = _time_module.time()
    last_print = 0.0
    while _time_module.time() - t_start < duration:
        if ser.in_waiting > 0:
            ln = ser.readline().decode('utf-8', errors='ignore').strip()
            if ln:
                try:
                    samples.append(int(ln))
                except ValueError:
                    pass
        elapsed = _time_module.time() - t_start
        if elapsed - last_print >= 5.0:
            pct = elapsed / duration * 100
            print(f"  ⏳ {pct:.0f}% 완료 ({len(samples)} 샘플)")
            last_print = elapsed
        _time_module.sleep(0.001)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_index', 'raw_value'])
        for i, v in enumerate(samples):
            writer.writerow([i, v])

    print(f"✅ 녹화 완료! {len(samples)} 샘플 → {filename}")
    return filename


def _pick_file_dialog(title, initialdir=None, filetypes=None, parent=None):
    """tkinter 파일 탐색기 창을 열어 파일 경로를 반환합니다. 취소 시 None 반환."""
    import tkinter as _tk
    from tkinter import filedialog as _fd
    
    # parent(기존 ui.root)가 제공되면 별도의 임시 root를 만들지 않고 안전하게 파일 탐색기만 실행
    if parent:
        if filetypes is None:
            filetypes = [('모든 파일', '*.*')]
        if initialdir and not os.path.isabs(initialdir):
            initialdir = os.path.abspath(initialdir)
        path = _fd.askopenfilename(title=title, initialdir=initialdir, filetypes=filetypes, parent=parent)
        return path if path else None

    try:
        root = _tk.Tk()
        root.withdraw()         # 메인 윈도우 숨김
        root.attributes('-topmost', True)   # 파일창이 항상 앞으로
    except Exception:
        root = None
    if filetypes is None:
        filetypes = [('모든 파일', '*.*')]
    if initialdir and not os.path.isabs(initialdir):
        initialdir = os.path.abspath(initialdir)
    path = _fd.askopenfilename(title=title, initialdir=initialdir, filetypes=filetypes)
    if root:
        try: root.destroy()
        except Exception: pass
    return path if path else None


def load_animal_data_from_file(parent=None):
    """파일 탐색기로 지오폰 CSV를 선택하면, 동일 타임스탬프의 마이크 CSV를 자동 매칭하여 듀얼 로드합니다."""
    folder_geo = os.path.abspath('Geophone')
    folder_mic = os.path.abspath('MEMS microphone')
    os.makedirs(folder_geo, exist_ok=True)
    os.makedirs(folder_mic, exist_ok=True)

    print("\n📂 [지오폰 발걸음 CSV] 파일 선택 창을 열고 있습니다...")
    chosen_geo = _pick_file_dialog(
        title='[지오폰] 발걸음 CSV 파일을 선택하세요',
        initialdir=folder_geo,
        filetypes=[('CSV 파일', '*.csv'), ('모든 파일', '*.*')],
        parent=parent
    )
    if not chosen_geo:
        print("⚠️ 취소되었습니다.")
        return None, None

    print(f" ✅ 지오폰 파일 선택: {os.path.basename(chosen_geo)}")

    # 지오폰 파일에서 타임스탬프 추출하여 마이크 파일 자동 매칭
    base_name = os.path.basename(chosen_geo)
    parts = base_name.replace('.csv', '').split('_')
    timestamp = parts[-1] if len(parts) >= 2 else None

    chosen_mic = None
    if timestamp:
        mic_candidates = glob.glob(os.path.join(folder_mic, f"*{timestamp}.csv"))
        if mic_candidates:
            chosen_mic = mic_candidates[0]
            print(f" ✅ 마이크 파일 자동 매칭: {os.path.basename(chosen_mic)}")

    if not chosen_mic:
        print("\n📂 [MEMS 마이크] 자동 매칭 실패 — 파일 선택 창을 열고 있습니다...")
        chosen_mic = _pick_file_dialog(
            title='[MEMS 마이크] 발걸음 CSV 파일을 선택하세요',
            initialdir=folder_mic,
            filetypes=[('CSV 파일', '*.csv'), ('모든 파일', '*.*')],
            parent=parent
        )
        if not chosen_mic:
            print("⚠️ 마이크 파일 선택이 취소되었습니다.")
            return None, None
        print(f" ✅ 마이크 파일 선택: {os.path.basename(chosen_mic)}")

    # 지오폰 로드
    raw = []
    with open(chosen_geo, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try: raw.append(int(row['raw_value']))
            except: pass

    # 마이크 로드
    raw_mic = []
    with open(chosen_mic, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try: raw_mic.append(int(row['raw_value']))
            except: pass

    if len(raw) < 100 or len(raw_mic) < 100:
        print("❌ 파일에 데이터가 너무 적습니다 (최소 100샘플 필요).")
        return None, None

    centered_geo = np.array(raw) - config.CURRENT_ZERO_GEO
    centered_mic = np.array(raw_mic) - config.CURRENT_ZERO_MIC
    print(f"✅ 파일 듀얼 로드 성공!")
    print(f" 👉 지오폰: {os.path.basename(chosen_geo)} ({len(centered_geo)} 샘플)")
    print(f" 👉 마이크: {os.path.basename(chosen_mic)} ({len(centered_mic)} 샘플)")
    return centered_geo, centered_mic


def load_env_noise_from_file(parent=None):
    """파일 탐색기로 배경 소음 파일(.npy 또는 .csv)을 선택하여 듀얼 채널 기준선을 로드합니다."""
    # 초기 탐색 경로: 'surround noise' 폴더 우선, 없으면 현재 폴더
    init_dir = os.path.abspath('surround noise') if os.path.exists('surround noise') else os.path.abspath('.')

    print("\n📂 [지오폰 배경 소음] 파일 선택 창을 열고 있습니다 (npy 또는 csv)...")
    chosen_geo = _pick_file_dialog(
        title='[지오폰] 배경 소음 파일을 선택하세요 (.npy 또는 .csv)',
        initialdir=init_dir,
        filetypes=[
            ('NumPy 배열 (npy)', '*.npy'),
            ('CSV 데이터', '*.csv'),
            ('모든 파일', '*.*')
        ],
        parent=parent
    )
    if not chosen_geo:
        print("⚠️ 지오폰 배경 소음 파일 선택이 취소되었습니다.")
        return None, None
    print(f" ✅ 지오폰 소음 파일 선택: {os.path.basename(chosen_geo)}")

    print("\n📂 [MEMS 마이크 배경 소음] 파일 선택 창을 열고 있습니다 (npy 또는 csv)...")
    chosen_mic = _pick_file_dialog(
        title='[MEMS 마이크] 배경 소음 파일을 선택하세요 (.npy 또는 .csv)',
        initialdir=os.path.dirname(chosen_geo),
        filetypes=[
            ('NumPy 배열 (npy)', '*.npy'),
            ('CSV 데이터', '*.csv'),
            ('모든 파일', '*.*')
        ],
        parent=parent
    )
    if not chosen_mic:
        print("⚠️ 마이크 배경 소음 파일 선택이 취소되었습니다.")
        return None, None
    print(f" ✅ 마이크 소음 파일 선택: {os.path.basename(chosen_mic)}")

    def _load_noise_file(path):
        """확장자에 따라 npy 또는 csv를 로드하고 DC 성분을 제거한 배열을 반환."""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            arr = np.load(path)
        elif ext == '.csv':
            import csv as _csv
            raw = []
            with open(path, newline='', encoding='utf-8-sig') as f:
                reader = _csv.DictReader(f)
                col_candidates = ['raw_value', 'geophone_raw', 'microphone_raw']
                for row in reader:
                    for col in col_candidates:
                        if col in row:
                            try: raw.append(int(row[col])); break
                            except: pass
            if len(raw) < 100:
                return None
            arr = np.array(raw, dtype=float)
        else:
            print(f"❌ 지원하지 않는 파일 형식입니다: {ext}")
            return None
        return arr - np.mean(arr)   # DC Bias 제거

    env_geo = _load_noise_file(chosen_geo)
    env_mic = _load_noise_file(chosen_mic)

    if env_geo is None or env_mic is None:
        print("❌ 파일 로드에 실패했습니다. 데이터가 너무 적거나 형식이 맞지 않습니다.")
        return None, None

    print(f"✅ 배경 소음 듀얼 로드 완료! 지오폰 {len(env_geo)} 샘플, 마이크 {len(env_mic)} 샘플")
    return env_geo, env_mic

def run_env_noise_calibration(ser):
    print("\n" + "="*60)
    print("🐾 야생동물 감지 시스템 - [1단계] 배경 소음 수집 및 파일 저장 🐾")
    print("="*60)
    
    print("\n[1단계/1] 배경 소음(배경 진동) 수집 시작 (10분)")
    print("💡 안내: 센서 주변을 완전히 조용하게 유지해 주십시오.")
    time.sleep(1.0)
    
    env_noise_raw = []
    start_time = time.time()
    last_print = 0.0
    while time.time() - start_time < 600.0:
        if ser.in_waiting > 0:
            ln = ser.readline().decode('utf-8', errors='ignore').strip()
            if ln:
                try:
                    val = int(ln)
                    env_noise_raw.append(val)
                except ValueError:
                    pass
        
        # 5초마다 진행 상황 표시
        elapsed = time.time() - start_time
        if elapsed - last_print >= 5.0:
            percent = (elapsed / 600.0) * 100
            print(f"⏳ 소음 수집 중... {percent:.1f}% 완료 ({len(env_noise_raw)} 샘플)")
            last_print = elapsed
        time.sleep(0.001)
        
    if len(env_noise_raw) < 100:
        print("❌ 경고: 데이터가 충분히 수집되지 않았습니다. 아두이노 전원이나 포트를 확인하세요.")
        return False
        
    env_noise_centered = np.array(env_noise_raw) - config.CURRENT_ZERO
    print(f"✅ 배경 소음 수집 완료! (평균 진폭: {np.mean(np.abs(env_noise_centered)):.2f})")
    
    # 넘파이 배열로 디스크에 영구 저장
    np.save('env_noise_baseline.npy', env_noise_centered)
    print("💾 배경 소음 데이터가 'env_noise_baseline.npy' 파일에 성공적으로 저장되었습니다!")
    return True

def _optimize_from_data(animal_signal_centered, env_noise_centered, ui=None):
    """SDFT 필터 학습 → 발걸음 추출 → a, b, sigma 최적화 공통 로직."""
    print("\n⏳ 최적화 연산 가동 중 (SDFT 필터 연동)...")
    buffer_size = config.BUFFER_SIZE
    calib_sdft = SDFTAdaptiveFilter()

    print(" ⏳ SDFT 필터에 배경 소음 학습 중...")
    for idx in range(0, len(env_noise_centered) - buffer_size, buffer_size):
        chunk = env_noise_centered[idx : idx + buffer_size]
        calib_sdft.process(chunk)

    print(" ⏳ SDFT 기반 발걸음(Transient) 조각 정밀 스캔 중...")
    valid_footstep_chunks = []
    if len(animal_signal_centered) > buffer_size:
        for idx in range(0, len(animal_signal_centered) - buffer_size, int(buffer_size/2)):
            chunk = animal_signal_centered[idx : idx + buffer_size]
            filtered_chunk, _, _, transient_detected = calib_sdft.process(chunk)
            if transient_detected:
                valid_footstep_chunks.append(filtered_chunk)

    if valid_footstep_chunks:
        best_chunk = max(valid_footstep_chunks, key=lambda c: np.sum(np.abs(c)))
        combined_input = best_chunk
        print(f"🔬 SDFT가 발걸음으로 판정한 {len(valid_footstep_chunks)}개 구간 중 최상위 조각 추출 완료!")
    else:
        print("⚠️ SDFT 감지 실패 — 진폭 기반으로 우회 추출합니다.")
        max_sum, best_idx = -1, 0
        for idx in range(0, len(animal_signal_centered) - buffer_size, 10):
            s = np.sum(np.abs(animal_signal_centered[idx : idx + buffer_size]))
            if s > max_sum:
                max_sum, best_idx = s, idx
        raw_chunk = animal_signal_centered[best_idx : best_idx + buffer_size]
        combined_input, _, _, _ = calib_sdft.process(raw_chunk)

    # Phase 1: a, b 파라미터 찾기
    a_range = np.linspace(10, 300, 30)
    engine = BistableDoubleWellEngine()
    results_N = []
    for a_val in a_range:
        engine.a = a_val
        engine.b = a_val
        _, _, N_t, _ = engine.process_buffer(combined_input, np.zeros_like(combined_input))
        results_N.append(N_t)
    results_N = np.array(results_N)
    acceptable = np.where(results_N <= 1)[0]
    opt_a = a_range[acceptable[0]] if len(acceptable) > 0 else a_range[-1]

    # Phase 2: sigma 파라미터 찾기
    VOLT_TO_ADC = 204.8
    sigma_v_range = np.arange(0.02, 0.31, 0.01)
    sigma_adc_range = sigma_v_range * VOLT_TO_ADC
    engine.a = opt_a
    engine.b = opt_a
    results_net_N = []
    for sigma in sigma_adc_range:
        white_noise = np.random.normal(0, sigma, buffer_size)
        _, _, N_t, K_t = engine.process_buffer(combined_input, white_noise)
        results_net_N.append(max(0, N_t - K_t))
    opt_sigma_idx = int(np.argmax(results_net_N))
    opt_sigma_adc = sigma_adc_range[opt_sigma_idx]

    print("\n" + "="*60)
    print("🎉 [자동 파라미터 튜닝 완료] 최적화 결과")
    print(f" 👉 역치 파라미터 (a, b) : {opt_a:.1f}")
    print(f" 👉 최적 인공 소음 (sigma) : {sigma_v_range[opt_sigma_idx]:.2f} V ({opt_sigma_adc:.1f} ADC)")
    print("="*60)

    # --- 최종 최적화 결과에 따른 실시간 데이터 시뮬레이션 및 UI 업데이트 ---
    if ui is not None:
        engine.a = opt_a
        engine.b = opt_a
        white_noise = np.random.normal(0, opt_sigma_adc, buffer_size)
        x_arr_tot, x_arr_noi, N_t, K_t = engine.process_buffer(combined_input, white_noise)
        telegraph_signal = np.sign(x_arr_tot)

        # ACF 분석
        acf_analyzer = ACFPeriodicityAnalyzer()
        acf_r, cadence = acf_analyzer.compute(telegraph_signal)

        # 주파수 분석 (FFT)
        half_n = buffer_size // 2
        raw_chunk = animal_signal_centered[:buffer_size] if len(animal_signal_centered) >= buffer_size else np.pad(animal_signal_centered, (0, buffer_size - len(animal_signal_centered)))
        fft_raw = np.abs(np.fft.fft(raw_chunk))[:half_n]
        fft_clean = np.abs(np.fft.fft(combined_input))[:half_n]
        
        # 배경 소음 FFT
        fft_avg = np.abs(np.fft.fft(env_noise_centered[:buffer_size]))[:half_n] if len(env_noise_centered) >= buffer_size else np.zeros(half_n)

        # SDFT 정보
        tracked_peak = calib_sdft.detected_noise_bands[0][0] + config.NOISE_BAND_WIDTH/2 if calib_sdft.detected_noise_bands else 0.0
        noise_bands = calib_sdft.detected_noise_bands

        # UI 업데이트 호출
        ui.update_optimization_graph(
            status=f'✅ 최적화 완료! a={opt_a:.1f}, σ={sigma_v_range[opt_sigma_idx]:.2f}V ({opt_sigma_adc:.1f} ADC)',
            raw_signal=raw_chunk,
            filtered_signal=combined_input,
            sr_signal=telegraph_signal,
            N_t=N_t,
            K_t=K_t,
            net_events=max(0, N_t - K_t),
            acf_r=acf_r,
            cadence=cadence,
            raw_fft=fft_raw,
            clean_fft=fft_clean,
            avg_fft=fft_avg,
            tracked_peak=tracked_peak,
            noise_bands=noise_bands,
            color='#55efc4'
        )

    return opt_a, opt_a, opt_sigma_adc


def run_footstep_calibration(ser, env_noise_centered):
    print("\n" + "="*60)
    print("🐾 야생동물 감지 시스템 - [2단계] 발걸음 수집 및 파라미터 최적화 🐾")
    print("="*60)

    print("\n[1단계/2] 동물 발걸음 실측 수집 시작 (30초)")
    print("💡 안내: 지금 센서 주변 흙바닥을 동물의 발걸음 세기에 맞춰 조심스럽게 밟아주십시오!")
    time.sleep(1.5)

    animal_signal_raw = []
    start_time = time.time()
    last_print = 0.0
    while time.time() - start_time < 30.0:
        if ser.in_waiting > 0:
            ln = ser.readline().decode('utf-8', errors='ignore').strip()
            if ln:
                try:
                    animal_signal_raw.append(int(ln))
                except ValueError:
                    pass
        elapsed = time.time() - start_time
        if elapsed - last_print >= 1.0:
            print(f"👣 발걸음 수집 중... {elapsed/30.0*100:.1f}% 완료 ({len(animal_signal_raw)} 샘플)")
            last_print = elapsed
        time.sleep(0.001)

    if len(animal_signal_raw) < 100:
        print("❌ 경고: 발걸음 데이터가 수집되지 않았습니다.")
        return None, None, None

    animal_signal_centered = np.array(animal_signal_raw) - config.CURRENT_ZERO
    print(f"✅ 발걸음 수집 완료! (최대 진폭: {np.max(np.abs(animal_signal_centered)):.2f})")

    return _optimize_from_data(animal_signal_centered, env_noise_centered)


import threading

def _run_env_noise_ui(ser, ui):
    """배경 소음 수집을 메인 스레드에서 실행하며 UI를 업데이트."""
    # 영점 자동 조절 (듀얼 채널)
    zero_geo, zero_mic = auto_calibrate_dual(ser)
    config.CURRENT_ZERO_GEO = zero_geo
    config.CURRENT_ZERO_MIC = zero_mic

    ui.show_recording_graph(
        '📡 [1단계] 배경 소음 측정',
        '센서 주변을 조용히 유지해 주세요 — 10분 동안 수집합니다',
        dual=True)
    samples = []
    start = _time_module.time()
    last_print = 0.0
    while _time_module.time() - start < 600.0:  # 10분(600초) 동안 수집
        if not ui.is_alive(): return
        if ser.in_waiting > 0:
            ln = ser.readline().decode('utf-8', errors='ignore').strip()
            if ln and ',' in ln:
                parts = ln.split(',')
                if len(parts) >= 2:
                    try:
                        val0 = int(parts[0])
                        val1 = int(parts[1])
                        samples.append((val0, val1))
                    except ValueError: pass
        elapsed = _time_module.time() - start
        if elapsed - last_print >= 0.05:
            pct = elapsed / 600.0 * 100
            ui.update_recording_graph(
                samples,
                f'⏳ 수집 중... {pct:.1f}%  ({len(samples)} 샘플)',
                '#74b9ff')
            last_print = elapsed
        ui.root.update()
        _time_module.sleep(0.001)
    if len(samples) < 100:
        ui.update_recording_graph(samples, '❌ 데이터 부족 — 아두이노 연결을 확인하세요', '#d63031')
        _time_module.sleep(2.0)
        return
    
    # 데이터 처리 및 저장 (듀얼 채널)
    env_geo = np.array([s[0] for s in samples]) - config.CURRENT_ZERO_GEO
    env_mic = np.array([s[1] for s in samples]) - config.CURRENT_ZERO_MIC
    
    # 1. 루트 폴더에 기준선 저장
    np.save('env_noise_baseline_geo.npy', env_geo)
    np.save('env_noise_baseline_mic.npy', env_mic)
    np.save('env_noise_baseline.npy', env_geo)  # 싱글 채널용 하위 호환
    
    # 2. 'surround noise' 폴더에 CSV 백업 저장
    folder_name = 'surround noise'
    os.makedirs(folder_name, exist_ok=True)
    ts = int(_time_module.time())
    
    # CSV 저장 (듀얼)
    import csv as _csv
    csv_fname = os.path.join(folder_name, f'noise_dual_{ts}.csv')
    with open(csv_fname, 'w', newline='') as f:
        w = _csv.writer(f)
        w.writerow(['sample_index', 'geophone_raw', 'microphone_raw'])
        for i, v in enumerate(samples):
            w.writerow([i, v[0], v[1]])
            
    # NPY 저장 백업
    np.save(os.path.join(folder_name, 'env_noise_baseline_geo.npy'), env_geo)
    np.save(os.path.join(folder_name, 'env_noise_baseline_mic.npy'), env_mic)
    
    ui.update_recording_graph(samples, f'✅ 완료! {len(samples)} 샘플 → {csv_fname} 저장됨', '#55efc4')
    ui.root.update()
    _time_module.sleep(2.0)


def _run_footstep_ui(ser, ui, env_noise_geo, env_noise_mic):
    """발걸음 30초 수집 + 최적화를 메인 스레드에서 실행 (듀얼 채널)."""
    # 영점 자동 조절
    zero_geo, zero_mic = auto_calibrate_dual(ser)
    config.CURRENT_ZERO_GEO = zero_geo
    config.CURRENT_ZERO_MIC = zero_mic

    ui.show_recording_graph(
        '👣 [2단계] 발걸음 실측 수집',
        '센서 주변을 힘껏 밟아주세요! (30초)',
        dual=True)
    samples = []
    start = _time_module.time()
    last_print = 0.0
    while _time_module.time() - start < 30.0:
        if not ui.is_alive(): return None, None, None, None, None, None
        if ser.in_waiting > 0:
            ln = ser.readline().decode('utf-8', errors='ignore').strip()
            if ln and ',' in ln:
                parts = ln.split(',')
                if len(parts) >= 2:
                    try:
                        val0 = int(parts[0])
                        val1 = int(parts[1])
                        samples.append((val0, val1))
                    except ValueError: pass
        elapsed = _time_module.time() - start
        if elapsed - last_print >= 0.05:
            pct = elapsed / 30.0 * 100
            ui.update_recording_graph(
                samples,
                f'👣 발걸음 수집 중... {pct:.1f}%  ({len(samples)} 샘플)',
                color='#fdcb6e')
            last_print = elapsed
        ui.root.update()
        _time_module.sleep(0.001)
    if len(samples) < 100:
        ui.update_recording_graph(samples, '❌ 발걸음 데이터 부족', color='#d63031')
        ui.root.update()
        _time_module.sleep(2.0)
        return None, None, None, None, None, None
        
    animal_geo = np.array([s[0] for s in samples]) - config.CURRENT_ZERO_GEO
    animal_mic = np.array([s[1] for s in samples]) - config.CURRENT_ZERO_MIC
    
    # 1. 지오폰 최적화
    ui.show_optimization_graph(env_noise_geo, animal_signal=animal_geo)
    ui.update_optimization_graph('⏳ 지오폰 파라미터 최적화 연산 중...', raw_signal=animal_geo, color='#a29bfe')
    ui.root.update()
    res_a_geo, _, res_sigma_geo = _optimize_from_data(animal_geo, env_noise_geo, ui=ui)
    ui.root.update()
    _time_module.sleep(1.5)
    
    # 2. 마이크 최적화
    ui.show_optimization_graph(env_noise_mic, animal_signal=animal_mic)
    ui.update_optimization_graph('⏳ 마이크 파라미터 최적화 연산 중...', raw_signal=animal_mic, color='#a29bfe')
    ui.root.update()
    res_a_mic, _, res_sigma_mic = _optimize_from_data(animal_mic, env_noise_mic, ui=ui)
    ui.root.update()
    _time_module.sleep(1.5)
    
    return res_a_geo, res_a_geo, res_sigma_geo, res_a_mic, res_a_mic, res_sigma_mic


def _run_file_optimize_ui(animal_geo, animal_mic, env_noise_geo, env_noise_mic, ui):
    """파일 불러오기 최적화를 메인 스레드에서 실행 (듀얼 채널)."""
    # 1. 지오폰 최적화
    ui.show_optimization_graph(env_noise_geo, animal_signal=animal_geo)
    ui.update_optimization_graph('⏳ 지오폰 파라미터 최적화 연산 중...', raw_signal=animal_geo, color='#a29bfe')
    ui.root.update()
    res_a_geo, _, res_sigma_geo = _optimize_from_data(animal_geo, env_noise_geo, ui=ui)
    ui.root.update()
    _time_module.sleep(1.5)
    
    # 2. 마이크 최적화
    ui.show_optimization_graph(env_noise_mic, animal_signal=animal_mic)
    ui.update_optimization_graph('⏳ 마이크 파라미터 최적화 연산 중...', raw_signal=animal_mic, color='#a29bfe')
    ui.root.update()
    res_a_mic, _, res_sigma_mic = _optimize_from_data(animal_mic, env_noise_mic, ui=ui)
    ui.root.update()
    _time_module.sleep(1.5)
    
    return res_a_geo, res_a_geo, res_sigma_geo, res_a_mic, res_a_mic, res_sigma_mic


def _run_record_ui(ser, ui):
    """신호 녹화를 메인 스레드에서 실행."""
    from ui_manager import SENSOR_FOLDERS
    sel = ui._pending_record_sel
    
    is_dual = (sel == '4')
    
    if is_dual:
        # 듀얼 채널용 지오폰 및 마이크 영점 동시 캘리브레이션
        zero_geo, zero_mic = auto_calibrate_dual(ser)
        config.CURRENT_ZERO_GEO = zero_geo
        config.CURRENT_ZERO_MIC = zero_mic
        
        # 각각 저장할 폴더 및 파일 정의
        os.makedirs('Geophone', exist_ok=True)
        os.makedirs('MEMS microphone', exist_ok=True)
        label  = ui._pending_record_label
        dur    = ui._pending_record_dur
        ts     = int(_time_module.time())
        fname_geo = os.path.join('Geophone', f'geophone_{label}_{ts}.csv')
        fname_mic = os.path.join('MEMS microphone', f'microphone_{label}_{ts}.csv')
        title_subtitle_target = f'Geophone & MEMS mic 폴더로 각각 분할 저장'
    else:
        channel = 1 if sel in ('2', '3') else 0
        # 녹화 전 해당 센서 채널의 영점(DC Bias)을 자동으로 2초간 캘리브레이션합니다.
        config.CURRENT_ZERO = auto_calibrate_zero(ser, channel)
        
        folder_name, prefix = SENSOR_FOLDERS[sel]
        os.makedirs(folder_name, exist_ok=True)
        label  = ui._pending_record_label
        dur    = ui._pending_record_dur
        ts     = int(_time_module.time())
        fname  = os.path.join(folder_name, f'{prefix}_{label}_{ts}.csv')
        title_subtitle_target = fname

    ui.show_recording_graph(
        f'🔴 신호 녹화 — 듀얼 모드' if is_dual else f'🔴 신호 녹화 — {folder_name}',
        f'{dur:.0f}초 동안 녹화합니다 → {title_subtitle_target}',
        dual=is_dual)

    samples = []
    start = _time_module.time()
    last_print = 0.0
    while _time_module.time() - start < dur:
        if not ui.is_alive(): return
        if ser.in_waiting > 0:
            ln = ser.readline().decode('utf-8', errors='ignore').strip()
            if ln:
                if is_dual:
                    if ',' in ln:
                        parts = ln.split(',')
                        if len(parts) >= 2:
                            try:
                                val0 = int(parts[0])
                                val1 = int(parts[1])
                                samples.append((val0, val1))
                            except ValueError: pass
                else:
                    val = parse_serial_line(ln, channel)
                    if val is not None:
                        samples.append(val)
        elapsed = _time_module.time() - start
        if elapsed - last_print >= 0.05:
            pct = elapsed / dur * 100
            ui.update_recording_graph(samples, f'⏳ {pct:.0f}%  ({len(samples)} 샘플)', '#e17055')
            last_print = elapsed
        ui.root.update()
        _time_module.sleep(0.001)

    import csv as _csv
    if is_dual:
        # 지오폰 저장
        with open(fname_geo, 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['sample_index', 'raw_value'])
            for i, v in enumerate(samples):
                w.writerow([i, v[0]])
        # MEMS 마이크 저장
        with open(fname_mic, 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['sample_index', 'raw_value'])
            for i, v in enumerate(samples):
                w.writerow([i, v[1]])
        ui.update_recording_graph(samples, f'✅ 저장 완료! 각각의 폴더에 분리 저장됨 (Geophone, MEMS microphone)', '#55efc4')
    else:
        with open(fname, 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['sample_index', 'raw_value'])
            for i, v in enumerate(samples):
                w.writerow([i, v])
        ui.update_recording_graph(samples, f'✅ 저장 완료! {len(samples)} 샘플 → {fname}', '#55efc4')
    ui.root.update()
    _time_module.sleep(2.0)



def main():
    # 하드웨어 설정 및 시리얼 연결
    try:
        ser = serial.Serial(config.PORT, config.BAUD_RATE, timeout=0.05)
        print(f'[{config.PORT}] 센서 연결 성공!')
    except Exception as e:
        print(f'포트 연결 실패: {e}')
        return

    # 기본 파라미터 (지오폰)
    opt_a_geo     = config.POTENTIAL_A
    opt_b_geo     = config.POTENTIAL_B
    opt_sigma_geo = config.SIGMA_NOISE

    # 기본 파라미터 (마이크로폰)
    opt_a_mic     = config.POTENTIAL_A
    opt_b_mic     = config.POTENTIAL_B
    opt_sigma_mic = config.SIGMA_NOISE

    # Tkinter UI 생성 (메인 스레드)
    ui = WildlifeUI()

    # ── 메뉴 루프 ──────────────────────────────────
    while True:
        if not ui.is_alive():
            break

        choice = ui.wait_for_choice()
        if choice is None:   # Quit 버튼
            break

        if choice == '1':
            _run_env_noise_ui(ser, ui)

        elif choice == '2':
            # 파일 선택창으로 듀얼 배경 소음 로드
            env_noise_geo, env_noise_mic = load_env_noise_from_file(parent=ui.root)

            if env_noise_geo is None or env_noise_mic is None:
                print("⚠️ 배경 소음 파일 선택이 취소되었거나 로드에 실패했습니다.")
                continue
                
            res = _run_footstep_ui(ser, ui, env_noise_geo, env_noise_mic)
            if res[0] is not None:
                opt_a_geo, opt_b_geo, opt_sigma_geo, opt_a_mic, opt_b_mic, opt_sigma_mic = res

        elif choice == '3':
            animal_geo, animal_mic = load_animal_data_from_file(parent=ui.root)
            if animal_geo is None or animal_mic is None:
                continue

            # 파일 선택창으로 듀얼 배경 소음 로드
            env_noise_geo, env_noise_mic = load_env_noise_from_file(parent=ui.root)

            if env_noise_geo is None or env_noise_mic is None:
                print("⚠️ 배경 소음 파일 선택이 취소되었거나 로드에 실패했습니다.")
                continue
                
            res = _run_file_optimize_ui(animal_geo, animal_mic, env_noise_geo, env_noise_mic, ui)
            if res[0] is not None:
                opt_a_geo, opt_b_geo, opt_sigma_geo, opt_a_mic, opt_b_mic, opt_sigma_mic = res

        elif choice == '4':
            # 듀얼 채널 동시 녹화 자동 진행
            print('\n📡 듀얼 채널(지오폰 & MEMS 마이크) 동시 녹화를 진행합니다.')
            label = input('👉 레이블 입력 (기본=raw): ').strip() or 'raw'
            dur_s = input('👉 녹화 시간 입력 (초, 기본=30): ').strip()
            try:    dur = float(dur_s) if dur_s else 30.0
            except: dur = 30.0
            ui._pending_record_sel   = '4'
            ui._pending_record_label = label
            ui._pending_record_dur   = dur
            _run_record_ui(ser, ui)

        elif choice == '5':
            break   # 실시간 감지로 진입

    if not ui.is_alive():
        ser.close()
        return

    # ── 실시간 감지 루프 (듀얼 채널 항상 작동) ───────────────────────────
    # 듀얼 채널 영점 동시 보정
    zero_geo, zero_mic = auto_calibrate_dual(ser)
    config.CURRENT_ZERO_GEO = zero_geo
    config.CURRENT_ZERO_MIC = zero_mic

    # 지오폰 파이프라인 컴포넌트
    data_buffer_geo   = [0.0] * config.BUFFER_SIZE
    step_analyzer_geo = StepDurationAnalyzer()
    sdft_filter_geo   = SDFTAdaptiveFilter()
    bistable_engine_geo = BistableDoubleWellEngine()
    bistable_engine_geo.a = opt_a_geo
    bistable_engine_geo.b = opt_b_geo
    acf_analyzer_geo  = ACFPeriodicityAnalyzer()

    # 마이크 파이프라인 컴포넌트
    data_buffer_mic   = [0.0] * config.BUFFER_SIZE
    step_analyzer_mic = StepDurationAnalyzer()
    sdft_filter_mic   = SDFTAdaptiveFilter()
    bistable_engine_mic = BistableDoubleWellEngine()
    bistable_engine_mic.a = opt_a_mic
    bistable_engine_mic.b = opt_b_mic
    acf_analyzer_mic  = ACFPeriodicityAnalyzer()

    def on_band_change(bands):
        sdft_filter_geo.manual_bands = bands
        sdft_filter_mic.manual_bands = bands

    ui.setup_live_detection(on_manual_band_change=on_band_change)
    last_render_time = time.time()
    print('시스템 엔진 가동 중 (지오폰 A0 & 마이크 A1 동시 분석)...')

    try:
        while ui.is_alive():
            data_updated = False
            while ser.in_waiting > 0:
                ln = ser.readline().decode('utf-8', errors='ignore').strip()
                if ln and ',' in ln:
                    parts = ln.split(',')
                    if len(parts) >= 2:
                        try:
                            val0 = int(parts[0])
                            val1 = int(parts[1])
                            data_buffer_geo.pop(0)
                            data_buffer_geo.append(val0 - zero_geo)
                            data_buffer_mic.pop(0)
                            data_buffer_mic.append(val1 - zero_mic)
                            data_updated = True
                        except ValueError:
                            pass

            current_time = time.time()
            if data_updated and (current_time - last_render_time >= config.RENDER_INTERVAL):
                # 1. 지오폰 분석 파이프라인
                raw_signal_geo = np.array(data_buffer_geo)
                signal_geo = raw_signal_geo - np.mean(raw_signal_geo)
                signal_geo = signal_geo * (1.0 - np.exp(-np.abs(signal_geo) / 2.5))
                
                act_idx_geo, is_rec_geo, step_comp_geo, dur_geo, durations_geo = step_analyzer_geo.analyze(signal_geo, current_time)
                avg_dur_geo = sum(durations_geo)/len(durations_geo) if durations_geo else 0.0
                filtered_signal_geo, M_t_geo, clean_fft_mag_geo, transient_detected_geo = sdft_filter_geo.process(signal_geo)
                
                white_noise_geo = np.random.normal(0, opt_sigma_geo, config.BUFFER_SIZE)
                x_arr_tot_geo, x_arr_noi_geo, N_t_geo, K_t_geo = bistable_engine_geo.process_buffer(filtered_signal_geo, white_noise_geo)
                net_events_geo = max(0, N_t_geo - K_t_geo)
                
                telegraph_signal_geo = np.sign(x_arr_tot_geo)
                acf_r_geo, cadence_geo = acf_analyzer_geo.compute(telegraph_signal_geo)

                # 2. 마이크 분석 파이프라인
                raw_signal_mic = np.array(data_buffer_mic)
                signal_mic = raw_signal_mic - np.mean(raw_signal_mic)
                signal_mic = signal_mic * (1.0 - np.exp(-np.abs(signal_mic) / 2.5))
                
                act_idx_mic, is_rec_mic, step_comp_mic, dur_mic, durations_mic = step_analyzer_mic.analyze(signal_mic, current_time)
                avg_dur_mic = sum(durations_mic)/len(durations_mic) if durations_mic else 0.0
                filtered_signal_mic, M_t_mic, clean_fft_mag_mic, transient_detected_mic = sdft_filter_mic.process(signal_mic)
                
                white_noise_mic = np.random.normal(0, opt_sigma_mic, config.BUFFER_SIZE)
                x_arr_tot_mic, x_arr_noi_mic, N_t_mic, K_t_mic = bistable_engine_mic.process_buffer(filtered_signal_mic, white_noise_mic)
                net_events_mic = max(0, N_t_mic - K_t_mic)
                
                telegraph_signal_mic = np.sign(x_arr_tot_mic)
                acf_r_mic, cadence_mic = acf_analyzer_mic.compute(telegraph_signal_mic)

                # 3. GUI 업데이트 (듀얼 채널 데이터 전달)
                ui.update(
                    filtered_signal_geo, filtered_signal_mic,
                    x_arr_tot_geo, x_arr_tot_mic,
                    M_t_geo, M_t_mic,
                    clean_fft_mag_geo, clean_fft_mag_mic,
                    sdft_filter_geo.M_avg, sdft_filter_mic.M_avg,
                    sdft_filter_geo.detected_noise_bands, sdft_filter_mic.detected_noise_bands,
                    is_rec_geo, is_rec_mic,
                    step_comp_geo, step_comp_mic,
                    dur_geo, dur_mic,
                    avg_dur_geo, avg_dur_mic,
                    len(durations_geo), len(durations_mic),
                    transient_detected_geo, transient_detected_mic,
                    N_t_geo, K_t_geo, net_events_geo, acf_r_geo, cadence_geo,
                    N_t_mic, K_t_mic, net_events_mic, acf_r_mic, cadence_mic
                )

                last_render_time = current_time

            # 아두이노 데이터가 들어오지 않더라도 GUI 이벤트가 처리되도록 함 (응답없음 방지)
            ui.root.update()
            time.sleep(0.001)

    except KeyboardInterrupt:
        print('파이프라인 종료...')
    finally:
        ser.close()
        ui.close()


if __name__ == '__main__':
    main()
