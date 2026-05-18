import numpy as np
import config

class ACFPeriodicityAnalyzer:
    """
    [보고서 제 3.3절 연계 (추가 검증)]: 자기상관함수(ACF) 모듈
    확률 공명 엔진을 통과해 나온 이진화된 텔레그래프 신호(sgn(x))를 바탕으로,
    야생동물 특유의 일정한 보행 리듬(주기성)이 존재하는지 수학적으로 판정합니다.
    """
    def __init__(self):
        self.r_threshold = getattr(config, 'ACF_R_THRESHOLD', getattr(config, 'ACF_R_THRESHOLD_GEO', 0.75))
        self.lag_min = config.ACF_LAG_MIN
        self.lag_max = config.ACF_LAG_MAX

    def compute(self, sr_state_signal):
        """
        SR 상태 신호 배열(이중 우물 전이 궤적)을 받아 ACF 상관계수(R)를 계산합니다.
        
        원리:
            신호의 분산(Variance)으로 정규화된 자기상관함수를 계산한 뒤,
            동물의 물리적 발걸음 주기에 해당하는 시간차(Lag: 0.3초 ~ 1.5초) 구간 내에서
            가장 강력하게 일치하는 지점의 피크 값(R)을 찾습니다.
        """
        n = len(sr_state_signal)
        
        # 신호가 완전히 정지해 있거나 우물 바닥에만 머물러 있는 경우 처리
        var = np.var(sr_state_signal)
        if var == 0 or n < self.lag_max:
            return 0.0, 0.0
            
        # 평균을 0으로 맞추어 정확한 교류 에너지 상관성만 파악
        signal_centered = sr_state_signal - np.mean(sr_state_signal)
        
        # 전체 지연(Lag)에 대한 ACF 계산
        acf_full = np.correlate(signal_centered, signal_centered, mode='full')
        # 음의 지연(과거)은 버리고 양의 지연(미래) 구간만 사용
        acf = acf_full[n-1:]
        
        # 정규화 (상관계수 R이 -1.0 ~ 1.0 사이에 오도록 스케일링)
        acf_normalized = acf / (var * n)
        
        # 동물의 물리적 한계 주기(0.3초 ~ 1.5초) 구간 내에서 가장 높은 주기성(피크) 탐색
        target_acf_range = acf_normalized[self.lag_min : self.lag_max]
        
        if len(target_acf_range) > 0:
            max_r = np.max(target_acf_range)
            # 피크가 위치한 실제 인덱스(주기 타임스탬프)
            best_lag = self.lag_min + np.argmax(target_acf_range)
            cadence_sec = best_lag / config.FS
            return max_r, cadence_sec
        else:
            return 0.0, 0.0
