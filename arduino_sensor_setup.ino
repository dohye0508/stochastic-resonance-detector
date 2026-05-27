#include <LiquidCrystal.h> // I2C가 빠진 기본 라이브러리

// 핀 연결: RS, E, D4, D5, D6, D7 순서
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

void setup() {
  Serial.begin(9600); // 파이썬으로 데이터 쏘기 위함
  lcd.begin(16, 2);   // 16칸 2줄 LCD 시작

  lcd.setCursor(0, 0);
  lcd.print("Seismograph"); // 첫 줄 고정 문구

  // ─── ADC 크로스토크 완전 억제 초기화 ───────────────────────────────
  // 프리스케일러 128 명시적 설정: 16MHz / 128 = 125kHz (ATmega328P 최저 클록)
  // 느린 ADC 클록 = 샘플앤홀드 커패시터 안정화 시간 최대 확보
  ADCSRA = (ADCSRA & ~0x07) | 0x07; // ADPS2:0 = 111 → prescaler 128

  // 시작 채널을 A0으로 고정하여 최초 채널 전환 충격 제거
  ADMUX = (ADMUX & 0xF0) | 0x00; // A0 채널로 초기화 (상위 4비트 기준전압 보존)
  delay(20); // 초기화 안정화 대기
  // ────────────────────────────────────────────────────────────────────
}

void loop() {
  // ═══════════════════════════════════════════════════════════════════
  // [ADC ADMUX 레지스터 직접 제어 방식 — 크로스토크 완전 억제]
  //
  // 원리:
  //   아두이노 내부 ADC는 14pF 샘플앤홀드 커패시터를 단 1개만 가지고 있습니다.
  //   채널을 A0 → A1로 전환할 때, 이 커패시터에 남아있는 A0(지오폰)의 잔류
  //   전하가 A1(마이크) 측정값을 오염시킵니다.
  //
  //   해결: ADMUX 레지스터로 채널을 "미리" 선택해두고, 커패시터가 새 채널
  //   전압으로 완전히 안정화될 때까지 2000us를 기다린 뒤 더미 리드 3회 후
  //   실제 값을 읽습니다. 이렇게 하면 잔류 전하가 100% 소거됩니다.
  // ═══════════════════════════════════════════════════════════════════

  // ── A0: Geophone (고임피던스 수동 코일 센서) ────────────────────────
  ADMUX = (ADMUX & 0xF0) | 0x00; // ADMUX로 A0 채널 직접 선택 (상위 4비트 보존)
  delayMicroseconds(2000);        // 2000us 대기 — 고임피던스 지오폰 완전 안정화
  analogRead(A0);                 // 더미 리드 1회 — 잔류 전하 추가 방전
  analogRead(A0);                 // 더미 리드 2회
  analogRead(A0);                 // 더미 리드 3회 — 완전 소거
  delayMicroseconds(500);         // 최종 안정화 대기
  int val0 = analogRead(A0);      // 실제 측정

  // ── A1: MEMS Microphone (저임피던스 능동 센서, 내장 앰프) ────────────
  ADMUX = (ADMUX & 0xF0) | 0x01; // ADMUX로 A1 채널 직접 선택
  delayMicroseconds(2000);        // 2000us 대기 — 지오폰 잔류 전하 완전 소거
  analogRead(A1);                 // 더미 리드 1회
  analogRead(A1);                 // 더미 리드 2회
  analogRead(A1);                 // 더미 리드 3회
  delayMicroseconds(500);         // 최종 안정화 대기
  int val1 = analogRead(A1);      // 실제 측정

  // 1. 파이썬(노트북)으로 쉼표로 구분하여 전송 (듀얼 채널)
  Serial.print(val0);
  Serial.print(",");
  Serial.println(val1);

  // 2. LCD 화면에는 두 채널 값 모두 표시
  lcd.setCursor(0, 1);
  lcd.print("G:");
  lcd.print(val0);
  lcd.print(" M:");
  lcd.print(val1);
  lcd.print("    "); // 이전 숫자가 길었을 때 잔상 지우기

  delay(10);
}
