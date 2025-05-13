#include <Arduino.h>

// Имена полей для справки
const char* labels[] = {
  "WorkTime","Temperature","Sensor1","Sensor2",
  "Sensor3","Sensor4","Water","Battery","MovedDist"
};

void setup() {
  Serial.begin(115200);
  randomSeed(analogRead(A0));
}

void loop() {
  static uint32_t workTimeSec = 0;

  // Синтетика вместо реальных датчиков
  float temperature = 25.0 + random(-30, 31) / 10.0;           // °C
  int   s1 = random(10, 120);  // мм
  int   s2 = random(10, 120);
  int   s3 = random(10, 120);
  int   s4 = random(10, 120);
  int   water   = random(0, 100); // %
  int   battery = random(20, 100); // %
  int   moved   = workTimeSec * 5; // условно 5 мм/с

  // Отправляем CSV‑строку
  Serial.print(workTimeSec);        Serial.print(',');
  Serial.print(temperature);        Serial.print(',');
  Serial.print(s1);                 Serial.print(',');
  Serial.print(s2);                 Serial.print(',');
  Serial.print(s3);                 Serial.print(',');
  Serial.print(s4);                 Serial.print(',');
  Serial.print(water);              Serial.print(',');
  Serial.print(battery);            Serial.print(',');
  Serial.println(moved);

  workTimeSec++;
  delay(1000); // 1 Гц
}
