# -*- coding: utf-8 -*-
"""
[보고서 2단계 최적화 실험] SR 파라미터(a, b) 및 최적 노이즈(sigma) 산출 (지오폰 / 마이크 독립 튜닝 지원) - 듀얼 채널 백업본
================================================================================
이 스크립트는 두 센서 채널(지오폰, 마이크) 각각에 대해 독립적으로 실행될 수 있습니다.
1단계 (Phase 1): 인공 노이즈 없이 배경 잡음을 차단할 수 있는 최적의 a, b(역치)를 찾습니다.
2단계 (Phase 2): 1단계에서 찾은 a, b 환경에서 전이 빈도가 최대가 되는 최적 노이즈(sigma)를 찾습니다.
================================================================================
"""
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import config
from msr_engine import BistableDoubleWellEngine

def load_latest_noise(sensor_suffix):
    """지정된 센서 타입(geophone 또는 microphone)의 가장 최근 배경 소음 데이터를 불러옵니다."""
    pattern = f"env_noise_{sensor_suffix}_*.csv"
    noise_files = glob.glob(pattern) or glob.glob("env_noise_raw_*.csv")
    
    if not noise_files:
        return None
    latest_file = max(noise_files, key=os.path.getctime)
    print(f"로드할 [{sensor_suffix}] 배경 소음: {latest_file}")
    
    values = []
    try:
        with open(latest_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                values.append(float(row['CenteredValue']))
        return np.array(values)
    except Exception as e:
        print(f"배경 소음 파일 로드 실패: {e}")
        return None

def load_animal_signal(filename):
    """CSV 파일에서 직접 측정한 동물 발자국 신호를 불러옵니다."""
    centered_values = []
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                centered_values.append(float(row['CenteredValue']))
        
        data = np.array(centered_values)
        if len(data) > config.BUFFER_SIZE:
            return data[:config.BUFFER_SIZE]
        else:
            return np.pad(data, (0, config.BUFFER_SIZE - len(data)))
    except Exception as e:
        print(f"발자국 파일 로드 실패: {e}")
        return None

def run_experiment_phase1(env_noise_full, animal_signal, sensor_suffix):
    """
    [Phase 1] Potential Parameter (a, b) Optimization
    """
    print("\n" + "="*50)
    print(f"[Phase 1] {sensor_suffix.upper()} 역치 파라미터(a, b) 탐색 시작")
    
    buffer_size = config.BUFFER_SIZE
    env_noise = env_noise_full[:buffer_size]
    combined_input = env_noise + animal_signal
    
    a_range = np.linspace(10, 300, 30)
    results_N = []
    engine = BistableDoubleWellEngine()
    
    for a_val in a_range:
        engine.a = a_val
        engine.b = a_val
        x_tot, x_noi, N_t, K_t = engine.process_buffer(combined_input, np.zeros_like(combined_input))
        results_N.append(N_t)
        
    results_N = np.array(results_N)
    acceptable_indices = np.where(results_N <= 1)[0]
    opt_a = a_range[acceptable_indices[0]] if len(acceptable_indices) > 0 else a_range[-1]
        
    print(f"도출된 [{sensor_suffix}] 역치 파라미터: a = {opt_a:.1f}, b = {opt_a:.1f}")
    return opt_a

def run_experiment_phase2(opt_a, env_noise_full, animal_signal, sensor_suffix):
    """
    [Phase 2] Optimal Artificial Noise (sigma) Search
    """
    print("\n" + "="*50)
    print(f"[Phase 2] {sensor_suffix.upper()} 최적 인공 노이즈 세기(sigma) 탐색 시작")
    
    env_noise = env_noise_full[:config.BUFFER_SIZE]
    combined_input = env_noise + animal_signal
    
    VOLT_TO_ADC = 204.8
    sigma_v_range = np.arange(0.02, 0.31, 0.01)
    sigma_adc_range = sigma_v_range * VOLT_TO_ADC
    
    engine = BistableDoubleWellEngine()
    engine.a = opt_a
    engine.b = opt_a
    
    results_net_N = []
    for sigma in sigma_adc_range:
        white_noise = np.random.normal(0, sigma, config.BUFFER_SIZE)
        x_tot, x_noi, N_t, K_t = engine.process_buffer(combined_input, white_noise)
        results_net_N.append(max(0, N_t - K_t))
        
    results_net_N = np.array(results_net_N)
    opt_sigma_idx = np.argmax(results_net_N)
    opt_sigma_v = sigma_v_range[opt_sigma_idx]
    
    print(f"최적 인공 노이즈: {opt_sigma_v:.2f} V ({sigma_adc_range[opt_sigma_idx]:.2f} ADC)")
    print(f"최대 순수 전이 빈도 (Net N): {results_net_N[opt_sigma_idx]}회")
    
    plt.figure(figsize=(10, 5))
    plt.plot(sigma_v_range, results_net_N, 'b-o')
    plt.axvline(opt_sigma_v, color='r', linestyle='--', label=f'Optimal D={opt_sigma_v:.2f}V')
    plt.title(f'[{sensor_suffix.upper()}] SR Peak Optimization (a,b={opt_a:.1f})')
    plt.xlabel('Artificial Noise (V)')
    plt.ylabel('Net Transitions (N-K)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"sr_{sensor_suffix}_optimization_peak.png")
    plt.show()
    return opt_sigma_v, sigma_adc_range[opt_sigma_idx]

if __name__ == "__main__":
    print("\n========================================================")
    print("분석할 센서 채널을 선택하십시오:")
    print("1. [지오폰 채널] 지표 진동 센서")
    print("2. [마이크 채널] 공기 음향 센서")
    print("========================================================")
    sensor_choice = input("선택 (1 또는 2): ").strip()
    
    sensor_suffix = "geophone" if sensor_choice == '1' else "microphone"
    
    print("\n========================================================")
    print(f"[{sensor_suffix.upper()} 채널] 어떤 방식으로 측정된 데이터를 분석하시겠습니까?")
    print("1. [현장 실측 방식] 실제 야외 흙바닥에서 측정한 발걸음 (배경 소음 이미 포함됨)")
    print("2. [대리 신호 합성 방식] 조용한 곳에서 측정한 발걸음 (환경 소음 인공 합성 필요)")
    print("========================================================")
    mode = input("선택 (1 또는 2): ").strip()
    
    step_pattern = f"animal_step_{sensor_suffix}_*.csv"
    step_files = glob.glob(step_pattern) or glob.glob("animal_step_*.csv")
    
    if not step_files:
        print(f"에러: 분석할 [{sensor_suffix}] 동물 발걸음 파일이 없습니다. record_animal_footstep.py를 먼저 실행하세요.")
    else:
        latest_step = max(step_files, key=os.path.getctime)
        print(f"\n로드할 발걸음 신호: {latest_step}")
        animal_signal = load_animal_signal(latest_step)
        
        if animal_signal is not None:
            if mode == '1':
                print(f">> [현장 실측 방식] [{sensor_suffix}] 실측 데이터를 그대로 사용하여 최적화를 수행합니다.")
                noise_data = np.zeros(config.BUFFER_SIZE)
                opt_a = run_experiment_phase1(noise_data, animal_signal, sensor_suffix)
                opt_sigma_v, opt_sigma_adc = run_experiment_phase2(opt_a, noise_data, animal_signal, sensor_suffix)
                
            elif mode == '2':
                print(f">> [대리 신호 합성 방식] [{sensor_suffix}] 대리 신호와 백운산 소음을 합성하여 최적화를 수행합니다.")
                noise_data = load_latest_noise(sensor_suffix)
                
                if noise_data is None:
                    print(f"에러: 합성할 [{sensor_suffix}] 배경 소음 파일이 없습니다. experiment_noise_recorder.py를 먼저 실행하세요.")
                else:
                    opt_a = run_experiment_phase1(noise_data, animal_signal, sensor_suffix)
                    opt_sigma_v, opt_sigma_adc = run_experiment_phase2(opt_a, noise_data, animal_signal, sensor_suffix)
            else:
                print("잘못된 입력입니다. 1 또는 2를 입력해주세요.")
                
            if mode in ['1', '2'] and (mode == '1' or noise_data is not None):
                print("\n" + "="*50)
                print(f"[{sensor_suffix.upper()} 최적 파라미터 도출 완료]")
                print("-" * 50)
                print(f" - 역치 파라미터 (a, b) : {opt_a:.1f}")
                print(f" - 최적 인공 노이즈 (sigma) : {opt_sigma_v:.2f} V ({opt_sigma_adc:.2f} ADC)")
                print("="*50)
                print(f"이 값을 config.py에 반영하여 최종 듀얼 채널 시스템을 기동하십시오.")
