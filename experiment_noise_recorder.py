# -*- coding: utf-8 -*-
"""
[보고서 4.3.가] 백운산 환경 소음 측정 및 정량화 스크립트
================================================================================
이 코드는 확률 공명(SR) 모델을 적용하기 전, 순수 배경 잡음의 특성을 파악하기 위한 실험용 도구입니다.
확률 공명이나 SDFT 필터를 거치지 않은 '생데이터(Raw Data)'를 30분간 수집하여 평균 진폭을 산출합니다.
================================================================================
"""
import serial
import time
import numpy as np
import csv
import os
import config

def record_background_noise(duration_minutes=15):
    # 1. 시리얼 포트 연결
    try:
        ser = serial.Serial(config.PORT, config.BAUD_RATE, timeout=0.1)
        print(f"[{config.PORT}] 센서 연결 성공. 측정을 시작합니다.")
    except Exception as e:
        print(f"포트 연결 실패: {e}")
        return

    # 2. 데이터 저장 설정
    output_filename = f"env_noise_raw_{int(time.time())}.csv"
    record_count = duration_minutes * 60 * config.FS  # 총 샘플 수 (30분 * 60초 * 100Hz = 180,000개)
    
    print(f"측정 예정 시간: {duration_minutes}분 (약 {int(record_count)} 샘플)")
    print(f"저장 파일명: {output_filename}")
    
    raw_values = []
    start_time = time.time()
    
    try:
        while len(raw_values) < record_count:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    try:
                        # 0~1023 사이의 Raw ADC 값 읽기
                        val = int(line)
                        raw_values.append(val)
                        
                        # 진행률 표시 (10초마다 한 번씩)
                        elapsed = time.time() - start_time
                        if len(raw_values) % (10 * config.FS) == 0:
                            progress = (len(raw_values) / record_count) * 100
                            print(f"진행 상황: {progress:.1f}% | 시간: {elapsed:.1f}s / {duration_minutes*60}s")
                            
                    except ValueError:
                        continue
            
            # CPU 점유율 방지
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n사용자에 의해 측정이 중단되었습니다. 지금까지의 데이터를 저장합니다.")
    finally:
        ser.close()

    if not raw_values:
        print("수집된 데이터가 없습니다.")
        return

    # 3. 데이터 정규화 및 분석 (보고서 4.3.가.3 단계)
    # 기준점(CURRENT_ZERO)을 뺀 실제 진동 변위 계산
    centered_data = np.array(raw_values) - config.CURRENT_ZERO
    
    # 평균 전압 진폭(Mean Absolute Deviation) 산출
    avg_amplitude = np.mean(np.abs(centered_data))
    peak_amplitude = np.max(np.abs(centered_data))
    std_dev = np.std(centered_data)

    print("\n" + "="*50)
    print("측정 결과 보고서")
    print("-" * 50)
    print(f"총 측정 샘플 수: {len(raw_values)} 개")
    print(f"배경 잡음 평균 진폭: {avg_amplitude:.4f} (ADC scale)")
    print(f"피크 진폭: {peak_amplitude:.4f}")
    print(f"표준 편차(Sigma): {std_dev:.4f}")
    print("="*50)

    # 4. CSV 파일 저장
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["SampleIndex", "RawValue", "CenteredValue"])
        for i, val in enumerate(raw_values):
            writer.writerow([i, val, val - config.CURRENT_ZERO])
            
    print(f"데이터가 성공적으로 {output_filename}에 저장되었습니다.")

if __name__ == "__main__":
    # 실제 실험을 위해 30분간 측정 (테스트를 원하시면 1 등으로 바꾸어 실행하세요)
    # record_background_noise(duration_minutes=30)
    
    # 우선 테스트를 위해 1분만 실행하는 옵션을 기본으로 둡니다.
    record_background_noise(duration_minutes=1)
