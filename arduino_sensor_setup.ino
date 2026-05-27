#include <LiquidCrystal.h> // I2C가 빠진 기본 라이브러리

// 핀 연결: RS, E, D4, D5, D6, D7 순서
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

void setup() {
  Serial.begin(9600); // 파이썬으로 데이터 쏘기 위함
  lcd.begin(16, 2);   // 16칸 2줄 LCD 시작

  lcd.setCursor(0, 0);
  lcd.print("Seismograph"); // 첫 줄 고정 문구
}

void loop() {
  // ── A0: Geophone (고임피던스 수동 코일 센서) ────────────────────────
  int val0 = analogRead(A0);      // 실제 측정

  // 1. 파이썬(노트북)으로 단일값 전송 (지오폰)
  Serial.println(val0);

  // 2. LCD 화면에는 지오폰 값만 표시
  lcd.setCursor(0, 1);
  lcd.print("Val: ");
  lcd.print(val0);
  lcd.print("      "); // 이전 숫자가 길었을 때 잔상 지우기

  delay(10); // 100Hz 샘플링 (10ms 대기)
}
