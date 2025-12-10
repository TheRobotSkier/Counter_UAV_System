#include <Arduino.h>
#include <DHT.h>

// Define pins and DHT sensor type:
#define TEMPPIN A0
#define DHTPIN 2
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

// Speed of sound in air using Cramer (1993)
// Inputs: temperature (°C), relative humidity (%)
// Constants: CO2 = 422.8 ppm, mean pressure = 101349 Pa

const double CO2_MOLE_FRACTION = 0.0004228;  // 422.8 ppm
const double PRESSURE = 101349.0;             // Pa

// Coefficients for Eq. 15 (speed of sound)
const double a0 = 331.5024;
const double a1 = 0.603055;
const double a2 = -0.000528;
const double a3 = 51.471935;
const double a4 = 0.1495874;
const double a5 = -0.000782;
const double a6 = -1.82e-07;
const double a7 = 3.73e-08;
const double a8 = -2.93e-10;
const double a9 = -85.20931;
const double a10 = -0.228525;
const double a11 = 5.91e-05;
const double a12 = -2.835149;
const double a13 = -2.15e-13;
const double a14 = 29.179762;
const double a15 = 0.000486;

double speedOfSound(double t_C, double RH_percent) {
  // Convert to SI units
  double RH = RH_percent / 100.0;
  double T = t_C + 273.15; // Kelvin
  double p = PRESSURE;

  // Eq. A2: enhancement factor
  double f = 1.00062 + 3.14e-8 * p + 5.6e-7 * t_C * t_C;

  // Eq. A3: saturation vapor pressure (Pa)
  double psv = exp(1.2811805e-5 * T * T
                  - 1.9509874e-2 * T
                  + 34.04926034
                  - 6.3536311e3 / T);

  // Eq. A1: mole fraction of water vapor
  double xw = RH * f * (psv / p);

  // CO2 mole fraction
  double Xc = CO2_MOLE_FRACTION;

  // Eq. 15: Speed of sound in m/s
  double c = a0
           + a1 * t_C + a2 * t_C * t_C
           + (a3 + a4 * t_C + a5 * t_C * t_C) * xw
           + (a6 + a7 * t_C + a8 * t_C * t_C) * p
           + (a9 + a10 * t_C + a11 * t_C * t_C) * Xc
           + a12 * xw * xw
           + a13 * p * p
           + a14 * Xc * Xc
           + a15 * xw * p * Xc;

  return c;
}

void setup() {
  Serial.begin(115200);
  dht.begin();
  // Set the reference voltage for analog input to the built-in 1.1 V reference:
  analogReference(INTERNAL);
  
  delay(2000);  // give the port and DHT sensor time to initialize
  Serial.println("Ready");
}

void loop() {
  /*
  // Get reading from the temperature sensor:
  int reading = analogRead(TEMPPIN);
  
  // Convert the reading into voltage:
  //float voltage = reading * (5000 / 1024.0); // in mV with 5V reference
  float voltage = reading * (1100 / 1024.0); // in mV with 1.1V reference

  // Convert the voltage into the temperature in degree Celsius:
  float temperature = voltage / 10;
  */
  // Get humidity and temperature values from DHT22:
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  /*
  if (!isnan(h) && !isnan(t)) {
    Serial.print("Analog Temp Sensor: "); Serial.print(temperature); Serial.println(" °C");
    Serial.print(" T: "); Serial.print(t);
    Serial.print(" °C  RH: "); Serial.print(h); Serial.println(" %");
  }
  delay(1500);*/
  Serial.print(t);
  Serial.print("|");
  Serial.println(h);
  delay(1000);

}

