# -*- coding: utf-8 -*-
"""
[보고서 4.2 실측 데이터 수집] 직접 측정한 동물 발자국 신호 기록 스크립트 (지오폰 + 마이크 듀얼 수집)
================================================================================
동물이 지나가는 시점에 실행하여 5초간의 고해상도 지오폰(진동) 및 마이크(음향) 데이터를 추출합니다.
마이크 신호는 10비트 가상 ADC 스케일(기존 512 기준)로 매핑되어 저장되므로 지오폰과 동일한 파이프라인에서 분석 가능합니다.
================================================================================
"""
import serial
import time
import numpy as np
import csv
import os
import sounddevice as sd
import config

def record_footstep_snippet(duration_sec=5.0, label="animal"):
    fs_audio = 16000  # 마이크 샘플링 레이트 (16kHz)
    downsample_factor = int(fs_audio / config.FS)  # 160배 다운샘플링 (100Hz로 맞춤)
    target_count = int(duration_sec * config.FS)  # 100Hz 기준 5초 = 500 샘플
    
    # 1. 마이크 녹음 시작 (Non-blocking)
    print("\n" + "="*50)
    print(f"[{label}] 발자국 기록을 시작합니다!")
    print(f"기록 시간: {duration_sec}초")
    print("지금 바로 센서 근처에서 발걸음(충격 및 소리)을 발생시켜 주세요.")
    print("="*50)
    
    try:
        audio_rec = sd.rec(int(duration_sec * fs_audio), samplerate=fs_audio, channels=1, dtype='float32')
    except Exception as e:
        print(f"마이크 녹음 시작 실패 (사운드 장치를 확인하세요): {e}")
        return

    # 2. 시리얼 연결 및 지오폰 수집
    try:
        ser = serial.Serial(config.PORT, config.BAUD_RATE, timeout=0.1)
        print(f"[{config.PORT}] 지오폰 센서 연결 성공. 동시 수집 중...")
    except Exception as e:
        print(f"지오폰 포트 연결 실패: {e}")
        sd.stop()
        return

    geo_values = []
    start_time = time.time()
    
    try:
        while len(geo_values) < target_count:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    try:
                        val = int(line)
                        geo_values.append(val)
                        
                        # 진행바 표시
                        if len(geo_values) % 10 == 0:
                            progress = len(geo_values) / target_count
                            bar = '#' * int(progress * 20)
                            print(f"\r수집 중: [{bar:<20}] {progress*100:.1f}%", end="")
                    except ValueError:
                        continue
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    finally:
        ser.close()
        sd.wait()  # 마이크 녹음 완료 대기

    print("\n\n녹음 완료! 데이터 변환 및 저장을 시작합니다.")

    # 3. 마이크 데이터 다운샘플링 및 가상 10비트 ADC 스케일 매핑
    # raw_audio는 [-1.0, 1.0] 범위 -> 512를 곱해 [-512, 512] 범위로 만들고 512를 더해 [0, 1024] 범위로 변환
    audio_flat = audio_rec.flatten()
    
    # 지오폰 샘플 수와 맞추기 위해 오디오 길이 조정
    required_audio_samples = target_count * downsample_factor
    if len(audio_flat) > required_audio_samples:
        audio_flat = audio_flat[:required_audio_samples]
    elif len(audio_flat) < required_audio_samples:
        audio_flat = np.pad(audio_flat, (0, required_audio_samples - len(audio_flat)))
        
    # 다운샘플링 수행 (상자필터 평균 데시메이션)
    audio_downsampled = np.mean(audio_flat.reshape(-1, downsample_factor), axis=1)
    
    # 가상 10비트 ADC 매핑 (Center: 512.0)
    mic_centered = audio_downsampled * 512.0
    mic_raw = mic_centered + config.CURRENT_ZERO
    
    # 지오폰 데이터도 최종 개수 맞춤
    if len(geo_values) > target_count:
        geo_values = geo_values[:target_count]
    elif len(geo_values) < target_count:
        geo_values = geo_values + [int(config.CURRENT_ZERO)] * (target_count - len(geo_values))

    geo_values = np.array(geo_values)
    geo_centered = geo_values - config.CURRENT_ZERO

    # 4. 각각의 CSV 파일로 분리 저장 (파라미터 독립 튜닝 지원)
    timestamp = int(time.time())
    geo_filename = f"animal_step_geophone_{label}_{timestamp}.csv"
    mic_filename = f"animal_step_microphone_{label}_{timestamp}.csv"

    # 지오폰 저장
    with open(geo_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["SampleIndex", "RawValue", "CenteredValue"])
        for i in range(target_count):
            writer.writerow([i, geo_values[i], geo_centered[i]])

    # 마이크 저장
    with open(mic_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["SampleIndex", "RawValue", "CenteredValue"])
        for i in range(target_count):
            writer.writerow([i, int(mic_raw[i]), int(mic_centered[i])])

    print("="*50)
    print("저장 완료 보고서")
    print("-" * 50)
    print(f"지오폰 파일: {geo_filename}")
    print(f"마이크 파일: {mic_filename}")
    print(f"총 샘플 수: {target_count}개 (Sampling: {config.FS}Hz)")
    print("="*50)
    print("이제 두 파일을 experiment_sr_optimization.py에서 독립적으로 불러와 최적의 파라미터를 구하세요.")

if __name__ == "__main__":
    animal_name = input("측정할 동물의 이름을 입력하세요 (기본: animal): ").strip() or "animal"
    record_footstep_snippet(duration_sec=5.0, label=animal_name)
