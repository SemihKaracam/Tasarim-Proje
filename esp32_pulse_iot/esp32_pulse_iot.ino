#include <WiFi.h>
#include <ESPAsyncWebSrv.h>

const char *ssid = "G301";
const char *password = "semih1306";

AsyncWebServer server(80);

int pulseRate = 0;  // Nabız değeri, sensörden okunan değer burada güncellenecek

int readPulseRate() {
  // Nabız sensörü 36 numaralı GPIO pinine bağlı
  int sensorPin = 36;

  // Analog değeri oku
  int sensorValue = analogRead(sensorPin);

  // Burada sensör değerini uygun bir şekilde işleyebilirsiniz
  // Örneğin, sensör değerini nabız hızına dönüştürmek için gerekli hesaplamaları yapabilirsiniz

  // Aşağıda sadece örnek bir dönüşüm yapılmıştır. Bu değeri sensör ve projenize göre uyarlayın.
  int pulseRate = map(sensorValue, 0, 4095, 60, 95);  // Örnek bir dönüşüm
  return pulseRate;
}

void setup() {
  Serial.begin(115200);

  // WiFi bağlantısını kur
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
  Serial.println(WiFi.localIP());
  DefaultHeaders::Instance().addHeader("Access-Control-Allow-Origin", "*");
  // "/get" endpoint'ine GET isteği geldiğinde handleGet fonksiyonunu çağır
  server.on("/get", HTTP_GET, [](AsyncWebServerRequest *request){
    // JSON verisini oluştur
    String json = "{\"pulse\": " + String(pulseRate) + "}";

    // JSON verisini isteği yanıt olarak gönder
    request->send(200, "application/json", json);
  });

  // Sunucuyu başlat
  server.begin();
}

void loop() {
  // Nabız sensöründen değeri oku ve güncelle
  pulseRate = readPulseRate();  // Bu fonksiyonu sensörünüz ve kullanılan kütüphane için uyarlayın
}

