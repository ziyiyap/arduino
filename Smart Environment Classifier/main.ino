
#include <DHT.h>
#include <U8g2lib.h>
#include <frames.h>

// output: temp, humidity, analogReadValue, sound, distance

//Pins initialization
int redPin = 9;
int greenPin = 10;
int bluePin = 11;
int DHTReader = 3;
const int trigPin = 7;
const int echoPin = 8;

// sound constants
unsigned long previousTime = 0;
unsigned long interval = 50;  // in ms
unsigned int sample;

//OLED Initialization
U8G2_SH1106_128X64_NONAME_1_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE);

DHT dht(DHTReader, DHT11);

double speed_of_sound = 0.0343;
long duration;
float distance;

double wave_dur();


void printData(float temp, float humidity, double analogVoltage, float sound, float distance) {
  Serial.print(temp);
  Serial.print(", ");
  Serial.print(humidity);
  Serial.print(", ");
  Serial.print(analogVoltage);
  Serial.print(", ");
  Serial.print(sound);
  Serial.print(", ");
  Serial.println(distance);
}

void setup() {
  // put your setup code here, to run once:

  //Ultrasensor
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  //Initialize Serial comms
  Serial.begin(9600);
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
  //Initialize DHT11
  dht.begin();
  //Initialize OLED
  u8g2.begin();
}

void loop() {
  // put your main code here, to run repeatedly:
    if (Serial.available() > 0) {
    String incoming = Serial.readStringUntil('\n');

    int firstComma = incoming.indexOf(',');
    String presence = incoming.substring(0, firstComma);

    String rest = incoming.substring(firstComma + 1);
    int secondComma = rest.indexOf(',');
    String env = rest.substring(0, secondComma);
    String sound = rest.substring(secondComma + 1);

    String presenceLine = "Presence: " + presence;
    String envLine = "Env: " + env;
    String soundLine = "Sound: " + sound;

    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_ncenB08_tr);

      u8g2.drawStr(0, 20, presenceLine.c_str());
      u8g2.drawStr(0, 40, envLine.c_str());
      u8g2.drawStr(0, 60, soundLine.c_str());
    } while (u8g2.nextPage());
  } else {
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_ncenB08_tr);

      u8g2.drawStr(0, 40, "Neural Network");
      u8g2.drawStr(0, 55, "Disconnected");
    } while (u8g2.nextPage());

  }
  
  previousTime = millis();
  unsigned int signalMax = 0;
  unsigned int signalMin = 1024;  // analog max is 1023.0 (including 0)
  unsigned int sound = 0;
  double analogVoltage = analogRead(A0);
  float temp = dht.readTemperature();
  float humidity = dht.readHumidity();

  if (isnan(temp) || isnan(humidity)) {
    Serial.println("Failed to read from DHT sensor");
    return;
  }

  while (millis() - previousTime < interval) {
    sample = analogRead(A1);
    if (sample > signalMax) {
      signalMax = sample;
    } else if (sample < signalMin) {
      signalMin = sample;
    }
  }
  sound = signalMax - signalMin;

  duration = wave_dur();
  distance = (duration * speed_of_sound) / 2;
  printData(temp, humidity, analogVoltage, sound, distance);
}

double wave_dur() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  return pulseIn(echoPin, HIGH);
}