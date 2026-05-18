# -*- coding: utf-8 -*-
"""
[보고서 4.3.가] 백운산 환경 소음 측정 및 정량화 스크립트 (지오폰 + 마이크 듀얼 수집)
================================================================================
이 코드는 확률 공명(SR) 모델을 적용하기 전, 순수 배경 잡음의 특성을 파악하기 위한 실험용 도구입니다.
지오폰(진동)과 마이크(음향)의 생데이터(Raw Data)를 동시에 수집하여 각각의 표준 편차(Sigma)와 평균 진폭을 산출합니다.
================================================================================
"""
import serial
import time
import numpy as np
import csv
import os
import sounddevice as sd
import config

def record_background_noise(duration_minutes=30):
    fs_audio = 16000  # 마이크 샘플링 레이트
    downsample_factor = int(fs_audio / config.FS)  # 160배 다운샘플링 (100Hz)
    duration_sec = duration_minutes * 60.0
    record_count = int(duration_sec * config.FS)  # 100Hz 기준 총 샘플 수
    
    print("\n" + "="*50)
    print("백운산 배경 소음 측정 시작 (지오폰 + 마이크 동시 수집)")
    print(f"측정 예정 시간: {duration_minutes}분 (약 {record_count} 샘플)")
    print("="*50)

    # 1. 마이크 녹음 시작 (Non-blocking)
    try:
        audio_rec = sd.rec(int(duration_sec * fs_audio), samplerate=fs_audio, channels=1, dtype='float32')
    except Exception as e:
        print(f"마이크 녹음 시작 실패 (사운드 장치를 확인하세요): {e}")
        return

    # 2. 시리얼 연결 및 지오폰 수집
    try:
        ser = serial.Serial(config.PORT, config.BAUD_RATE, timeout=0.1)
        print(f"[{config.PORT}] 지오폰 연결 성공. 동시 측정 진행 중...")
    except Exception as e:
        print(f"지오폰 포트 연결 실패: {e}")
        sd.stop()
        return

    geo_values = []
    start_time = time.time()
    
    try:
        while len(geo_values) < record_count:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    try:
                        val = int(line)
                        geo_values.append(val)
                        
                        # 10초마다 진행률 표시
                        elapsed = time.time() - start_time
                        if len(geo_values) % (10 * config.FS) == 0:
                            progress = (len(geo_values) / record_count) * 100
                            print(f"진행 상황: {progress:.1f}% | 시간: {elapsed:.1f}s / {duration_sec}s")
                            
                    except ValueError:
                        continue
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n사용자에 의해 측정이 중단되었습니다. 지금까지 수집된 데이터를 저장합니다.")
    finally:
        ser.close()
        sd.wait()  # 마이크 녹음 완료 대기

    if not geo_values:
        print("수집된 데이터가 없습니다.")
        return

    print("\n측정 완료! 데이터 전처리 및 분석을 수행합니다...")

    # 3. 마이크 데이터 다운샘플링 및 가상 10비트 ADC 변환
    audio_flat = audio_rec.flatten()
    required_audio_samples = len(geo_values) * downsample_factor
    if len(audio_flat) > required_audio_samples:
        audio_flat = audio_flat[:required_audio_samples]
    elif len(audio_flat) < required_audio_samples:
        audio_flat = np.pad(audio_flat, (0, required_audio_samples - len(audio_flat)))
        
    audio_downsampled = np.mean(audio_flat.reshape(-1, downsample_factor), axis=1)
    mic_centered = audio_downsampled * 512.0
    mic_raw = mic_centered + config.CURRENT_ZERO

    geo_values = np.array(geo_values[:len(audio_downsampled)])
    geo_centered = geo_values - config.CURRENT_ZERO

    # 4. 각각의 센서 데이터 분석 보고서 작성
    # 지오폰 통계
    geo_avg = np.mean(np.abs(geo_centered))
    geo_peak = np.max(np.abs(geo_centered))
    geo_std = np.std(geo_centered)

    # 마이크 통계
    mic_avg = np.mean(np.abs(mic_centered))
    mic_peak = np.max(np.abs(mic_centered))
    mic_std = np.std(mic_centered)

    print("\n" + "="*50)
    print("배경 소음 통계 분석 보고서")
    print("-" * 50)
    print(f"총 측정 샘플 수: {len(geo_values)} 개 (측정 시간: {len(geo_values)/config.FS:.1f}초)")
    print("\n[지오폰 채널 (지표 진동)]")
    print(f" - 평균 소음 진폭: {geo_avg:.4f} (ADC)")
    print(f" - 피크 소음 진폭: {geo_peak:.4f} (ADC)")
    print(f" - 소음 표준 편차 (Sigma_env): {geo_std:.4f}")
    print("\n[마이크 채널 (공기 음향)]")
    print(f" - 평균 소음 진폭: {mic_avg:.4f} (가상 ADC)")
    print(f" - 피크 소음 진폭: {mic_peak:.4f} (가상 ADC)")
    print(f" - 소음 표준 편차 (Sigma_env): {mic_std:.4f}")
    print("="*50)

    # 5. CSV 파일로 독립 저장
    timestamp = int(time.time())
    geo_filename = f"env_noise_geophone_{timestamp}.csv"
    mic_filename = f"env_noise_microphone_{timestamp}.csv"

    # 지오폰 노이즈 저장
    with open(geo_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["SampleIndex", "RawValue", "CenteredValue"])
        for i, val in enumerate(geo_values):
            writer.writerow([i, val, geo_centered[i]])

    # 마이크 노이즈 저장
    with open(mic_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["SampleIndex", "RawValue", "CenteredValue"])
        for i, val in enumerate(mic_raw):
            writer.writerow([i, int(val), int(mic_centered[i])])

    print(f"지오폰 소음 데이터 저장 완료: {geo_filename}")
    print(f"마이크 소음 데이터 저장 완료: {mic_filename}")

if __name__ == "__main__":
    # 테스트를 위해 1분간 측정 (실제 백운산 연구에는 15~30분을 권장합니다)
    record_background_noise(duration_minutes=1)
