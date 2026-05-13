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

# 각 장(Chapter)별 핵심 모듈 불러오기
from step_analyzer import StepDurationAnalyzer
from sdft_filter import SDFTAdaptiveFilter
from msr_engine import BistableDoubleWellEngine
from ui_manager import RealTimeUI

# ---------------------------------------------------------
# 마스터 설정 (Master Configuration)
# ---------------------------------------------------------
PORT = 'COM4'               # 아두이노 포트
BAUD_RATE = 9600            # 통신 속도
FS = 100.0                  # 샘플링 주파수 (100Hz)
BUFFER_SIZE = 128           # 버퍼 창 크기 (1.28초 분량의 데이터를 한 번에 관측)
CURRENT_ZERO = 512.0        # 아두이노 아날로그 센서 기준점 (10비트 ADC 중간값)
RENDER_INTERVAL = 0.03      # 화면 주사율 조절용 타이머 (초)

THRESHOLD_SCALAR = 1.618    # 전체 감지 민감도를 조절하는 마스터 상수

def main():
    # [제 2장] 하드웨어 설정 및 시리얼 연결
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=0.05)
        print(f"[{PORT}] 센서 연결이 성공적으로 완료되었습니다.")
    except Exception as e:
        print(f"포트 연결 실패: {e}")
        return

    # 실시간 데이터를 담아둘 메인 버퍼 배열
    data_buffer = [0.0] * BUFFER_SIZE

    # 보고서 단원별 서브 모듈 인스턴스 생성
    # 1. 엔지니어링 궤적 추적기 (발걸음 지속시간 분석)
    step_analyzer = StepDurationAnalyzer(
        threshold_start=15.0 * THRESHOLD_SCALAR,
        threshold_end=8.0 * THRESHOLD_SCALAR
    )
    
    # 2. [제 3.1절 & 3.2절] SDFT 적응형 스펙트럼 필터 (소음 감쇄용)
    sdft_filter = SDFTAdaptiveFilter(
        fs=FS, buffer_size=BUFFER_SIZE, 
        noise_band_width=5.0, alpha_ema=0.05, flux_k=2.5
    )
    
    # 3. [제 4.1.1절] 확률 공명 비선형 이중 우물 엔진
    bistable_engine = BistableDoubleWellEngine(
        a=50.0, b=50.0, dt=1.0/FS, force_scalar=1.5, bound=2.2
    )
    
    # 4. [제 3.3절] UI 및 실시간 경보 출력 매니저
    ui = RealTimeUI(
        fs=FS, buffer_size=BUFFER_SIZE,
        noise_min_freq=5.0, noise_max_freq=48.0, 
        alpha_ema=0.05, noise_band_width=5.0
    )

    print("시스템 엔진 가동 중... 센서 데이터를 기다립니다.")
    last_render_time = time.time()

    try:
        # UI 창이 열려있는 동안 무한 루프
        while plt.fignum_exists(ui.fig.number):
            data_updated = False

            # [제 2장] 버퍼 데이터 흡수 (Buffer Intake)
            while ser.in_waiting > 0:
                ln = ser.readline().decode('utf-8', errors='ignore').strip()
                if ln:
                    try:
                        # 기준점을 빼서 음/양 진동 신호로 변환 (DC 억제)
                        calibrated_val = int(ln) - CURRENT_ZERO
                        data_buffer.pop(0)  # 가장 오래된 데이터 제거
                        data_buffer.append(calibrated_val)  # 최신 데이터 삽입
                        data_updated = True
                    except ValueError:
                        pass

            current_time = time.time()
            # 렌더링 간격(0.03초)이 지났고 데이터가 업데이트 되었을 때만 분석 실행
            if data_updated and (current_time - last_render_time >= RENDER_INTERVAL):
                raw_signal = np.array(data_buffer)
                signal = raw_signal - np.mean(raw_signal)  # 추가적인 직류 편향성(DC Offset) 억제

                # 1. 발걸음 지속시간 분석기 연산
                act_idx, is_rec, step_comp, dur, durations = step_analyzer.analyze(signal, current_time)
                avg_dur = sum(durations)/len(durations) if durations else 0.0

                # 2. [제 3.1절 & 3.2절] SDFT 노치 필터링 수행 (소음 억제 및 파형 정제)
                filtered_signal, M_t, clean_fft_mag, transient_detected = sdft_filter.process(signal)

                # 3. [제 4.1.1절] 이중 우물 퍼텐셜 방정식 수치 적분 수행
                # 비교군으로 사용할 가상의 백색 소음(White Noise) 생성
                white_noise = np.random.normal(0, 12.0, BUFFER_SIZE)
                x_arr_tot, x_arr_noi, N_t, K_t = bistable_engine.process_buffer(filtered_signal, white_noise)
                
                # 순수 전이 횟수(Net Events) = 신호+소음 전이 횟수(N) - 순수 소음 전이 횟수(K)
                net_events = max(0, N_t - K_t)

                # 4. [제 3.3절] UI 화면 업데이트 및 최종 경보(Alert) 상태 판별
                ui.update(
                    filtered_signal, x_arr_tot, M_t, clean_fft_mag, sdft_filter.M_avg, sdft_filter.detected_noise_bands,
                    is_rec, step_comp, dur, avg_dur, len(durations), transient_detected, N_t, K_t, net_events
                )

                last_render_time = current_time

            # CPU 점유율 과부하 방지용 미세 휴식
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("사용자 명령에 의해 파이프라인을 종료합니다...")
    finally:
        # 시스템 안전 종료 로직
        ser.close()
        ui.close()

if __name__ == "__main__":
    main()
