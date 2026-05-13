# -*- coding: utf-8 -*-
"""
제 4.1.1절: 비선형 이중 우물 퍼텐셜(Double-Well Potential) 적분 엔진
================================================================================
확률 공명(Stochastic Resonance, SR) 현상을 시뮬레이션하기 위한 수학적 기반 및 수치해석 솔버
================================================================================

[1. 이중 우물 퍼텐셜 함수 공식]
    U(x) = (b / 4) * x^4 - (a / 2) * x^2

    * 퍼텐셜에 의한 복원력(경사면의 힘): F_pot(x) = -dU/dx = a*x - b*x^3
    * 안정된 우물 바닥 (입자가 쉬는 곳): x_m = +/- sqrt(a/b)
    * 가운데 에너지 장벽 (넘어야 할 산): x = 0 (장벽의 높이: dU = a^2 / 4b)

[2. 비선형 랑주뱅 미분방정식 (Langevin Equation)]
    아날로그 센서 입력 S(t)와 백색소음 N(t)가 밀어주는 힘을 받을 때, 
    우물 안에 갇힌 입자의 물리적 움직임을 결정하는 미분방정식입니다.
    
    dx/dt = a*x - b*x^3 + Force(t)
    (여기서 Force(t) = [S(t) + N(t)] * force_scalar)

[3. 미분방정식을 코드로 푸는 원리: 오일러-마루야마(Euler-Maruyama) 수치 적분]
    컴퓨터는 연속적인 시간을 계산할 수 없으므로, 아두이노가 보내는 매우 짧은 시간 간격(dt = 0.01초)마다 
    입자가 얼마나 이동했는지를 단계별로 쪼개서 더해나갑니다.
    
    * 핵심 원리: "다음 위치 = 현재 위치 + (현재의 이동 속도) * (짧은 시간)"
               => 거리는 속도 * 시간이라는 기본 물리 법칙과 완벽히 동일합니다.
    * 수식 적용: x_{t+1} = x_t + (dx/dt) * dt
               => x_{t+1} = x_t + (a*x_t - b*x_t^3 + Force_t) * dt
"""
import numpy as np

class BistableDoubleWellEngine:
    def __init__(self, a=50.0, b=50.0, dt=0.01, force_scalar=1.0, bound=2.2):
        """
        보고서 4.1.1절에 맞춰 수치 시뮬레이터 파라미터를 초기화합니다.
        
        매개변수:
            a (float): 퍼텐셜 우물의 기울기 및 복원력을 결정하는 선형 계수
            b (float): 에너지 장벽의 가파름을 결정하는 비선형 스칼라 계수
            dt (float): 이산적 시간 간격 (1.0 / 샘플링 주파수). 100Hz면 0.01초
            force_scalar (float): 100Hz의 짧은 찰나(dt)에 입자가 충분히 움직이도록 외력을 증폭하는 배수
            bound (float): 프로그램이 수학적 발산(무한대)으로 튕기는 것을 막는 절대적 하드웨어 안전 한계치
        """
        self.a = a
        self.b = b
        self.dt = dt
        self.force_scalar = force_scalar
        self.bound = bound
        
        # 이전 프레임에서 계산된 입자의 마지막 위치를 기억해두는 변수입니다.
        # 이를 통해 버퍼(프레임)가 넘어가도 입자의 움직임이 물리적으로 끊기지 않고 연속되도록 보장합니다.
        self.x_state_total = 1.0
        self.x_state_noise = 1.0

    def process_buffer(self, filtered_signal, white_noise):
        """
        실시간으로 들어오는 센서 데이터 버퍼(배열) 전체에 대해 오일러-마루야마 적분을 실행합니다.
        
        수학적 연산 과정:
            1. 신호+소음 배열에 대한 순간적인 전체 외력(Force)을 계산합니다.
            2. 매 샘플마다 미분방정식을 풀어 입자의 연속적인 위치 x(t)를 업데이트합니다.
            3. 입자가 가운데 장벽(x=0)을 넘나든 횟수(0-교차)를 세어 전이 빈도를 계산합니다.
            
        반환값:
            x_arr_total: 신호+소음의 힘을 받아 움직인 입자의 궤적 (발소리가 포함된 실제 상태)
            x_arr_noise: 순수 배경 소음만으로 움직인 입자의 궤적 (비교를 위한 대조군)
            N_t: 신호+소음 상태에서 입자가 장벽(0)을 넘어 전이한 총 횟수
            K_t: 소음 단독 상태에서 우연히 장벽을 넘은 기준 전이 횟수
        """
        buffer_size = len(filtered_signal)
        x_arr_total = np.zeros(buffer_size)
        x_arr_noise = np.zeros(buffer_size)
        
        # 센서에서 들어온 원본 스케일이 너무 커서 x^3 항이 무한대로 발산하는 것을 막기 위해
        # force_scalar를 곱해 연속 수학 모델에 맞는 안전한 스케일로 외력을 미리 축소/조정합니다.
        force_total = (filtered_signal + white_noise) * self.force_scalar
        force_noise = white_noise * self.force_scalar
        
        # 버퍼 안의 128개 샘플을 하나씩 꺼내며 순차적으로 시간을 0.01초씩 흐르게 합니다.
        for i in range(buffer_size):
            # -----------------------------------------------------------
            # [단계 1] 미분방정식(dx/dt) 계산: 현재 입자가 받는 '총 속도(변화량)' 구하기
            # -----------------------------------------------------------
            # f_tot = (우물 경사면이 밀어내거나 당기는 힘) + (외부에서 센서와 소음이 때리는 힘)
            f_tot = self.a * self.x_state_total - self.b * (self.x_state_total**3) + force_total[i]
            
            # -----------------------------------------------------------
            # [단계 2] 오일러-마루야마 수치 적분 적용: 입자의 위치 업데이트
            # -----------------------------------------------------------
            # 다음 위치 = 현재 위치 + (총 속도) * (0.01초)
            self.x_state_total = self.x_state_total + f_tot * self.dt
            
            # 물리적인 우물 벽이 무한히 높지 않고 센서의 한계가 있음을 반영하여, 
            # 입자가 너무 멀리 날아가지 않도록 최대/최소 위치(bound)를 제한합니다.
            self.x_state_total = max(min(self.x_state_total, self.bound), -self.bound)
            x_arr_total[i] = self.x_state_total
            
            # (대조군 연산) 위와 완전히 동일한 과정을 '순수 백색 소음'에 대해서만 따로 진행합니다.
            f_noi = self.a * self.x_state_noise - self.b * (self.x_state_noise**3) + force_noise[i]
            self.x_state_noise = self.x_state_noise + f_noi * self.dt
            self.x_state_noise = max(min(self.x_state_noise, self.bound), -self.bound)
            x_arr_noise[i] = self.x_state_noise
            
        # -----------------------------------------------------------
        # [단계 3] 전이 빈도 함수 N(t)와 K(t) 계산
        # -----------------------------------------------------------
        # 입자가 다른 우물로 넘어갔다는 것은 가운데 에너지 장벽(x=0)을 교차했다는 뜻입니다.
        # 배열 상에서 이전 샘플과 다음 샘플의 부호(+, -)가 달라진 순간(Zero-crossing)을 
        # 모두 더하여 총 몇 번 넘나들었는지(Well-hopping 횟수)를 셉니다.
        N_t = int(np.sum((x_arr_total[:-1] >= 0) != (x_arr_total[1:] >= 0)))
        K_t = int(np.sum((x_arr_noise[:-1] >= 0) != (x_arr_noise[1:] >= 0)))
        
        return x_arr_total, x_arr_noise, N_t, K_t

    def reset_states(self):
        """강제로 입자의 위치를 평상시 기저 상태(우물 바닥)로 초기화합니다."""
        self.x_state_total = 1.0
        self.x_state_noise = 1.0
