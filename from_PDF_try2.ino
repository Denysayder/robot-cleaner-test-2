#include <Arduino.h>

void setup() {
  Serial.begin(115200);
  randomSeed(analogRead(A0));
}

void loop() {
  static uint32_t sec = 0;
  static float temperature = 25.0;
  static int   water       = 100;
  static float battery     = 100.0;  // Изменили на float для плавности

  /*---- динамика температуры чуть «мягче»  ----*/
  temperature = constrain(temperature + random(-10, 11) * 0.04, 20, 60);

  /*---- солнечная мощность с небольшим шумом  ----*/
  float daylight = max(0.0f, sin(sec * 0.0007f));            // плавное "солнце"
  int   s1 = (int)(300 + daylight * 900) + random(-15, 16);  // 280-1220 W
  int   s2 = 30 + (sec % 20);                                // 30→49 за 20 с
  int   s3 = 400 + (sec * 20) % 400;                         // 400→800
  int   s4 = 5 + (sec * 4) % 90;                             // 5→95

  /*---- расход ресурсов/накопление пути  ----*/
  if (sec != 0 && sec % 3 == 0 && water > 90) water--;      // −5 % за 20 с

  // Плавное снижение батареи с небольшим шумом
  if (battery > 5.0) {
    float batteryDecline = 0.4 + random(-2, 3) * 0.05;      // ~0.4% в секунду ± шум
    battery = max(5.0f, battery - batteryDecline);
  }

  long moved = sec * 7;      // расстояние растёт ~линейно

  Serial.print(sec);            Serial.print(',');
  Serial.print(temperature, 1); Serial.print(',');
  Serial.print(s1);             Serial.print(',');
  Serial.print(s2);             Serial.print(',');
  Serial.print(s3);             Serial.print(',');
  Serial.print(s4);             Serial.print(',');
  Serial.print(water);          Serial.print(',');
  Serial.print((int)battery);   Serial.print(',');          // Приводим к int для вывода
  Serial.println(moved);

  sec++;
  delay(1000);                  // 1 сек → 20 строк за демонстрацию
}