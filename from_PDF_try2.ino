#include <Arduino.h>

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

  float temperature = 25.0 + random(-30, 31) / 10.0;
  int   s1 = random(10, 120);
  int   s2 = random(10, 120);
  int   s3 = random(10, 120);
  int   s4 = random(10, 120);
  int   water   = random(0, 100);
  int   battery = random(20, 100);
  int   moved   = workTimeSec * 5;

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
  delay(1000);
}
