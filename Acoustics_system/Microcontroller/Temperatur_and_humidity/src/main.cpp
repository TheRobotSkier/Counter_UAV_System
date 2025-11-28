#include <Arduino.h>
#include <DHT.h>

// DHT22 sensor
#define DHTPIN 2
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

// How long to measure (milliseconds)
const unsigned long MEASURE_TIME_MS = 5000;

void setup() {
  Serial.begin(115200);
  dht.begin();
  delay(2000);
  Serial.println("READY");  // handshake for PC
}

void loop() {

  // Check if a command has arrived
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "MEASURE") {
      Serial.println("START");

      unsigned long t0 = millis();
      while (millis() - t0 < MEASURE_TIME_MS) {
        float h = dht.readHumidity();
        float t = dht.readTemperature();

        if (!isnan(h) && !isnan(t)) {
          Serial.print("th_");
          Serial.print(t, 2);
          Serial.print("_");
          Serial.println(h, 2);
        }

        delay(500);
      }

      Serial.println("DONE");
    }
  }
}
