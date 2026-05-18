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
  // [아두이노 ADC 멀티플렉서 간섭(Crosstalk) 방지 공학적 기법]
  // 채널 전환 시 내부 커패시터 충방전 시간을 벌기 위해 
  // 더미 리드(Dummy Read) 후 미세 지연을 주고 진짜 값을 읽습니다.
  
  // A0: Geophone (진동 센서)
  analogRead(A0); // 더미 리드 (이전 잔류 전압 버리기)
  delayMicroseconds(200);
  int val0 = analogRead(A0); // 진짜 측정
  
  // A1: MEMS Microphone (음향 센서)
  analogRead(A1); // 더미 리드 (이전 잔류 전압 버리기)
  delayMicroseconds(200);
  int val1 = analogRead(A1); // 진짜 측정
  
  // 1. 파이썬(노트북)으로 쉼표로 구분하여 전송 (듀얼 채널)
  Serial.print(val0);
  Serial.print(",");
  Serial.println(val1);
  
  // 2. LCD 화면에는 지오폰 값을 기본 표시
  lcd.setCursor(0, 1);
  lcd.print("G:");
  lcd.print(val0);
  lcd.print(" M:");
  lcd.print(val1);
  lcd.print("    "); // 이전 숫자가 길었을 때 잔상 지우기
  
  delay(10);
}
