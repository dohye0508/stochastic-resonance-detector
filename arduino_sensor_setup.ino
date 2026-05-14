#include <LiquidCrystal.h> // I2C가 빠진 기본 라이브러리다!

// 핀 연결: RS, E, D4, D5, D6, D7 순서
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

void setup() {
  Serial.begin(9600); // 파이썬으로 데이터 쏘기 위함
  lcd.begin(16, 2);   // 16칸 2줄 LCD 시작
  
  lcd.setCursor(0, 0);
  lcd.print("Seismograph"); // 첫 줄 고정 문구
}

void loop() {
  int val = analogRead(A0);
  
  // 1. 파이썬(노트북)으로 숫자 전송
  Serial.println(val);
  
  // 2. LCD 화면에 숫자 표시
  lcd.setCursor(0, 1);
  lcd.print("Value: ");
  lcd.print(val);
  lcd.print("    "); // 이전 숫자가 길었을 때 잔상 지우기
  
  delay(10);
}
