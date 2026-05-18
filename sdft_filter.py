import numpy as np
import time
import config

class SDFTAdaptiveFilter:
    """
    [보고서 제 3.1절 & 3.2절]: 적응형 소음 추적 및 SDFT 스펙트럼 필터 모듈 (Spectral Flattening 적용)
    차량 엔진음과 같은 지배 소음 대역(노란색 영역)을 단순 일괄 감쇄하는 것이 아니라,
    초기 5초간 학습한 주변의 '다른 주파수들의 평균 기저 에너지(Baseline)'와 완벽하게 
    동일한 높이가 되도록 주파수 빈(bin)마다 각기 다른 맞춤형 감쇄 비율을 적용합니다.
    """
    def __init__(self):
        self.fs = config.FS
        self.buffer_size = config.BUFFER_SIZE
        self.half_n = self.buffer_size // 2
        # 나이퀴스트 이론에 따른 주파수 X축 배열 생성
        self.x_freq = np.fft.fftfreq(self.buffer_size, 1/self.fs)[:self.half_n]
        
        self.noise_band_width = config.NOISE_BAND_WIDTH
        self.noise_peak_count = config.NOISE_PEAK_COUNT
        self.noise_min_freq = config.NOISE_MIN_FREQ
        self.noise_max_freq = config.NOISE_MAX_FREQ
        self.alpha_ema = config.ALPHA_EMA
        self.flux_k = config.FLUX_K
        
        # 5초간 주변 환경 소음을 파악하기 위한 초기화 타이머
        self.start_time = time.time()
        self.calibration_time = config.CALIBRATION_TIME
        
        # 지속 소음 프로필을 누적 기억하는 장기 평균 배열
        self.M_avg = np.zeros(self.half_n)
        # 현재 화면에서 감지된 타겟 소음 밴드 목록
        self.detected_noise_bands = []
        self.manual_bands = []   # 사용자가 UI에서 수동으로 지정한 감쇄 대역 [(low, high), ...]

    def process(self, signal):
        # 1. 고속 푸리에 변환(FFT)을 통해 시간 도메인 신호를 주파수 도메인으로 변환
        raw_fft_complex = np.fft.fft(signal)
        M_t = np.abs(raw_fft_complex[:self.half_n]) / self.half_n
        freqs = np.fft.fftfreq(self.buffer_size, 1/self.fs)
        
        # 2. 지수이동평균(EMA)을 이용해 장기 소음 프로필 업데이트 (배경 소음 학습)
        if np.sum(self.M_avg) == 0:
            self.M_avg = np.copy(M_t) # 최초 실행 시 초기화
        else:
            self.M_avg = (1.0 - self.alpha_ema) * self.M_avg + self.alpha_ema * M_t
            
        # 3. 실시간으로 가장 강한 지배 소음(Dominant Noise) 대역 경계 식별
        valid_mask = (self.x_freq >= self.noise_min_freq) & (self.x_freq <= self.noise_max_freq)
        valid_indices = np.where(valid_mask)[0]
        
        self.detected_noise_bands.clear()
        noise_mask_global = np.zeros(self.half_n, dtype=bool)
        
        if len(valid_indices) > 0:
            M_avg_valid = self.M_avg.copy()
            for _ in range(self.noise_peak_count):
                if np.sum(M_avg_valid[valid_indices]) == 0:
                    break
                # 가장 강력한 소음 주파수(피크)를 탐색
                peak_idx = valid_indices[np.argmax(M_avg_valid[valid_indices])]
                peak_freq = self.x_freq[peak_idx]
                
                # 피크를 중심으로 좌우 밴드(Band) 폭 설정
                band_low = max(self.noise_min_freq, peak_freq - self.noise_band_width)
                band_high = min(self.noise_max_freq, peak_freq + self.noise_band_width)
                self.detected_noise_bands.append((band_low, band_high, peak_freq))
                
                # 다음 피크 탐색을 위해 현재 찾은 밴드는 임시 마스킹(0 처리)
                suppress_mask = (self.x_freq >= band_low) & (self.x_freq <= band_high)
                M_avg_valid[suppress_mask] = 0.0
                noise_mask_global |= suppress_mask
                
        # [수동 감쇄] 사용자가 드래그한 영역 추가
        for m_low, m_high in self.manual_bands:
            m_mask = (self.x_freq >= m_low) & (self.x_freq <= m_high)
            noise_mask_global |= m_mask
            # UI 표시를 위해 detected 목록에도 추가 (피크값은 중간값으로 표시)
            self.detected_noise_bands.append((m_low, m_high, (m_low + m_high)/2))
                
        # 원본 FFT 복소수 배열 복사 (여기에 필터링을 가함)
        clean_fft_complex = np.copy(raw_fft_complex)
        transient_detected = False
        
        # 프로그램 실행 후 경과 시간 확인
        elapsed = time.time() - self.start_time
        
        # 초기 5초(캘리브레이션 기간) 동안은 주변 환경 주파수를 그저 관측하며 학습합니다.
        # 5초가 지나면 평탄화(Flattening) 감쇄 로직을 가동합니다.
        if elapsed >= self.calibration_time:
            # 타겟 소음 밴드(노란색)를 제외한 나머지 "조용한" 주변 주파수들의 평균 기저 에너지(Baseline)를 계산합니다.
            non_noise_indices = valid_indices[~noise_mask_global[valid_indices]]
            if len(non_noise_indices) > 0:
                baseline_target = np.median(self.M_avg[non_noise_indices])
            else:
                baseline_target = np.mean(self.M_avg)
                
            # 4. 소음 대역에 대한 주파수별 맞춤형 스펙트럼 평탄화(Spectral Flattening) 및 과도 신호 보호
            for i in range(self.half_n):
                if noise_mask_global[i]:
                    # 수동 지정 대역인지 확인 (수동 대역은 보호 로직을 무시하고 강제 감쇄)
                    is_manual = any(low <= self.x_freq[i] <= high for low, high in self.manual_bands)
                    
                    # [과도 신호 보호] 현재 에너지가 평소보다 flux_k 배 이상 크다면 발걸음이므로 보호!
                    if is_manual or M_t[i] <= self.M_avg[i] * self.flux_k:
                        # [안정화된 감쇄] 순간적인 M_t가 아닌 학습된 M_avg를 기준으로 감쇄 배율(gain) 결정
                        # 이를 통해 매 프레임 파형이 요동치는 'Musical Noise' 현상을 억제합니다.
                        if self.M_avg[i] > baseline_target:
                            gain = baseline_target / self.M_avg[i]
                            
                            # 복소수 스펙트럼 값에 독립적인 맞춤 감쇄 적용
                            clean_fft_complex[i] *= gain
                            if i > 0:
                                clean_fft_complex[self.buffer_size - i] *= gain
                    else:
                        transient_detected = True # 발걸음 보호 발동!
                        
        # 5. 역 푸리에 변환(IFFT)을 통해 튀어나온 소음 대역만 정밀하게 깎인 깨끗한 시간 파형 복원
        filtered_signal = np.fft.ifft(clean_fft_complex).real
        
        # UI에 그리기 위한 필터링 후의 스펙트럼 크기(Magnitude) 배열
        clean_fft_mag = np.abs(clean_fft_complex[:self.half_n]) / self.half_n
        
        return filtered_signal, M_t, clean_fft_mag, transient_detected
