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
# 설정값 모듈 불러오기
import config

# 각 장(Chapter)별 핵심 모듈 불러오기
from step_analyzer import StepDurationAnalyzer
from sdft_filter import SDFTAdaptiveFilter
from msr_engine import BistableDoubleWellEngine
from acf_analyzer import ACFPeriodicityAnalyzer
from ui_manager import RealTimeUI

def main():
    # [제 2장] 하드웨어 설정 및 시리얼 연결
    try:
        ser = serial.Serial(config.PORT, config.BAUD_RATE, timeout=0.05)
        print(f"[{config.PORT}] 센서 연결이 성공적으로 완료되었습니다.")
    except Exception as e:
        print(f"포트 연결 실패: {e}")
        return

    # 실시간 데이터를 담아둘 메인 버퍼 배열
    data_buffer = [0.0] * config.BUFFER_SIZE

    # 보고서 단원별 서브 모듈 인스턴스 생성
    step_analyzer = StepDurationAnalyzer()
    sdft_filter = SDFTAdaptiveFilter()
    bistable_engine = BistableDoubleWellEngine()
    acf_analyzer = ACFPeriodicityAnalyzer()

    # UI 상호작용성 강화: 사용자가 스펙트럼에서 드래그한 영역을 필터에 실시간 반영
    def on_band_change(bands):
        sdft_filter.manual_bands = bands
        
    ui = RealTimeUI(on_manual_band_change=on_band_change)

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
                        calibrated_val = int(ln) - config.CURRENT_ZERO
                        data_buffer.pop(0)  # 가장 오래된 데이터 제거
                        data_buffer.append(calibrated_val)  # 최신 데이터 삽입
                        data_updated = True
                    except ValueError:
                        pass

            current_time = time.time()
            # 렌더링 간격이 지났고 데이터가 업데이트 되었을 때만 분석 실행
            if data_updated and (current_time - last_render_time >= config.RENDER_INTERVAL):
                raw_signal = np.array(data_buffer)
                signal = raw_signal - np.mean(raw_signal)  # 추가적인 직류 편향성(DC Offset) 억제

                # [미세 노이즈 소프트 감쇠 (Soft-Thresholding)]
                # 완전히 0으로 자르지 않고, 진폭이 작을수록 강하게 억제하며 큰 충격은 그대로 보존하는 수학적 비선형 감쇠
                signal = signal * (1.0 - np.exp(-np.abs(signal) / 2.5))

                # 1. 발걸음 지속시간 분석기 연산
                act_idx, is_rec, step_comp, dur, durations = step_analyzer.analyze(signal, current_time)
                avg_dur = sum(durations)/len(durations) if durations else 0.0

                # 2. [제 3.1절 & 3.2절] SDFT 노치 필터링 수행
                filtered_signal, M_t, clean_fft_mag, transient_detected = sdft_filter.process(signal)

                # 3. [제 4.1.1절] 이중 우물 퍼텐셜 방정식 수치 적분 수행
                # config의 SIGMA_NOISE를 기반으로 백색 소음 생성
                white_noise = np.random.normal(0, config.SIGMA_NOISE, config.BUFFER_SIZE)
                x_arr_tot, x_arr_noi, N_t, K_t = bistable_engine.process_buffer(filtered_signal, white_noise)
                
                # 순수 전이 횟수(Net Events) = 신호+소음 전이 횟수(N) - 순수 소음 전이 횟수(K)
                net_events = max(0, N_t - K_t)

                # 4. [제 3.3절 추가] ACF 주기성 분석 (텔레그래프 신호 기반)
                # 이중 우물 상태 궤적(x_arr_tot)을 -1과 +1의 이진화된 부호파동으로 변환하여 완벽한 리듬 판독
                telegraph_signal = np.sign(x_arr_tot)
                acf_r, cadence = acf_analyzer.compute(telegraph_signal)

                # 5. [제 3.3절 최종] UI 화면 업데이트 및 최종 경보(Alert) 상태 판별
                ui.update(
                    filtered_signal, x_arr_tot, M_t, clean_fft_mag, sdft_filter.M_avg, sdft_filter.detected_noise_bands,
                    is_rec, step_comp, dur, avg_dur, len(durations), transient_detected, 
                    N_t, K_t, net_events, acf_r, cadence
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
