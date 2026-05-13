import time
import csv
import numpy as np

class StepDurationAnalyzer:
    """
    [엔지니어링 모듈]: 1차 차분(가속도)과 슈미트 트리거(Schmitt Trigger)를 활용하여 
    실제 발걸음의 물리적 지속 시간(Duration)을 측정하고 기록하는 모듈입니다.
    """
    def __init__(self, threshold_start=15.0, threshold_end=8.0, hang_time=0.1, csv_filename="step_duration_data.csv"):
        # 발걸음 판정 시작 기준 (이 값을 넘으면 측정을 시작)
        self.threshold_start = threshold_start
        # 발걸음 판정 종료 기준 (이 값 밑으로 떨어지면 측정을 종료할지 고민)
        self.threshold_end = threshold_end
        # 행 타임 (신호가 약해져도 바로 종료하지 않고 기다려주는 여유 시간, 0.1초)
        self.hang_time = hang_time
        self.csv_filename = csv_filename
        
        # 내부 상태 추적 변수들
        self.is_recording = False
        self.step_start_time = 0.0
        self.end_candidate_time = 0.0
        self.step_durations = []
        
    def analyze(self, signal, current_time):
        # 파형의 단순 높이가 아닌, '얼마나 날카롭게 꺾였는가(1차 차분)'를 구함
        diff_signal = np.diff(signal)
        # 차분값의 에너지를 나타내는 RMS(평균 제곱근) 계산
        diff_rms = np.sqrt(np.mean(diff_signal**2)) if len(diff_signal) > 0 else 0.0
        # 시각적 스케일 보정을 위한 배수 (활동성 지표)
        activity_index = diff_rms * 1.5
        
        duration = 0.0
        step_completed = False
        
        # 상태 기계(State Machine) 로직 (슈미트 트리거 기반)
        if not self.is_recording:
            # 대기 상태에서 시작 기준(threshold_start)을 돌파하면 측정 시작!
            if activity_index > self.threshold_start:
                self.is_recording = True
                self.step_start_time = current_time
        else:
            # 측정 중일 때
            if activity_index > self.threshold_end:
                # 에너지가 아직 종료 기준보다 높으면, 종료 후보 시간을 최신으로 갱신
                self.end_candidate_time = current_time
            else:
                # 에너지가 종료 기준 밑으로 떨어졌고, 떨어진 지 여유 시간(hang_time)이 지났다면?
                if current_time - self.end_candidate_time > self.hang_time:
                    # 진짜 발걸음이 끝났다고 확정하고 지속 시간 계산
                    duration = self.end_candidate_time - self.step_start_time
                    
                    # 너무 짧은 노이즈(0.01초 이하)는 무시하고, 진짜 발걸음만 기록
                    if duration > 0.01:
                        self.step_durations.append(duration)
                        step_completed = True
                        try:
                            # CSV 파일에 현재 시각과 지속 시간을 백그라운드에서 저장
                            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
                                writer = csv.writer(f)
                                writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), f"{duration:.3f}"])
                        except Exception:
                            pass
                    
                    # 다시 대기 상태로 복귀
                    self.is_recording = False
                    
        return activity_index, self.is_recording, step_completed, duration, self.step_durations
