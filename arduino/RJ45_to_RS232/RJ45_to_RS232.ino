#include <SPI.h>
#include <EEPROM.h>
#include <Ethernet.h>
#include <LiquidCrystal.h>
#include <SoftwareSerial.h>

SoftwareSerial mySerial(A3, A4);  // RX, TX

uint32_t BAUD_RATES[] = { 300, 600, 1200, 2400, 4800, 9600, 14400, 19200, 28800, 31250, 38400, 57600, 115200, 0 };
size_t baudRateIndex = 5;  // default to 9600

// LCD pin to Arduino
const int pin_RS = 8;
const int pin_EN = 9;
const int pin_d4 = 4;
const int pin_d5 = 5;
const int pin_d6 = 6;
const int pin_d7 = 7;
const int pin_BL = 10;

LiquidCrystal lcd(pin_RS, pin_EN, pin_d4, pin_d5, pin_d6, pin_d7);

enum Key {
  RIGHT,
  UP,
  DOWN,
  LEFT,
  SELECT,
  NONE
};

enum Mode {
  SELECTION,
  DISPLAYING
};

enum Selection {
  BAUD_RATE,
  PORT_NUMBER
};

// Enter a MAC address for your controller below.
// Newer Ethernet shields have a MAC address printed on a sticker on the shield
byte mac[] = {
  0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED
};

EthernetServer *server;
uint16_t portNumber;

enum EEPROMAddress {
  BAUD_RATE_EEPROM_ADDRESS = 0,
  PORT_NUMBER_EEPROM_ADDRESS = 4
};

uint16_t readU16FromEEPROM(int idx) {
  uint16_t value = 0;
  for (int i = 0; i < (sizeof value); ++i) {
    value |= uint16_t(EEPROM.read(idx + i)) << (i * 8);
  }
  return value;
}

void writeU16ToEEPROM(int idx, uint16_t value) {
  for (int i = 0; i < (sizeof value); ++i) {
    EEPROM.update(idx + i, (value >> (i * 8)) & 0xff);
  }
}

uint32_t readU32FromEEPROM(int idx) {
  uint32_t value = 0;
  for (int i = 0; i < (sizeof value); ++i) {
    value |= uint32_t(EEPROM.read(idx + i)) << (i * 8);
  }
  return value;
}

void writeU32ToEEPROM(int idx, uint32_t value) {
  for (int i = 0; i < (sizeof value); ++i) {
    EEPROM.update(idx + i, (value >> (i * 8)) & 0xff);
  }
}

String toString(const IPAddress &ip) {
  String s;
  for (uint8_t i = 0; i < 3; ++i) {
    String w = String(ip[i]);
    while (w.length() < 3) {
      w = String('0') + w;
    }
    s += w;
    s += '.';
  }
  s += String(ip[3]);
  return s;
}

void showServerReadyIPAddress() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.write("Server is ready");
  lcd.setCursor(0, 1);
  lcd.write(toString(Ethernet.localIP()).c_str());
}

void showBaudRate(bool valueOnly = false) {
  if (valueOnly) {
    lcd.setCursor(0, 1);
    lcd.print("                ");
  } else {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.write("Baud Rate:");
  }
  lcd.setCursor(0, 1);
  lcd.print(BAUD_RATES[baudRateIndex]);
}

void showPortNumber(bool valueOnly = false) {
  if (valueOnly) {
    lcd.setCursor(0, 1);
    lcd.print("                ");
  } else {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.write("TCP Port:");
  }
  lcd.setCursor(0, 1);
  lcd.print(portNumber);
}

void setup() {
  // Find a baud rate if it's stored in EEPROM correctly:
  {
    uint32_t baud_rate = readU32FromEEPROM(BAUD_RATE_EEPROM_ADDRESS);
    size_t i = 0;
    while (BAUD_RATES[i] && BAUD_RATES[i] != baud_rate) {
      ++i;
    }
    if (BAUD_RATES[i]) {
      baudRateIndex = i;
    }
  }
  // Open serial communications:
  mySerial.begin(BAUD_RATES[baudRateIndex]);

  lcd.begin(16, 2);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.write("Initializing");
  lcd.setCursor(0, 1);
  lcd.write("DHCP connection");

  // start the Ethernet connection:
  if (Ethernet.begin(mac) == 0) {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.write("DHCP failed");
    lcd.setCursor(0, 1);
    if (Ethernet.hardwareStatus() == EthernetNoHardware) {
      lcd.write("Shield missing");
    } else if (Ethernet.linkStatus() == LinkOFF) {
      lcd.write("No cable");
    }
    // no point in carrying on, so do nothing forevermore:
    while (true) {
      delay(1);
    }
  }

  // Initialize the Ethernet server library
  // with the IP address and port you want to use:
  portNumber = readU16FromEEPROM(PORT_NUMBER_EEPROM_ADDRESS);
  if (!portNumber) {
    portNumber = 9111;
  }
  server = new EthernetServer(portNumber);

  // start the server
  server->begin();
  showServerReadyIPAddress();
}

Key getKey() {
  int x = analogRead(0);
  if (x < 60) {
    return RIGHT;
  } else if (x < 200) {
    return UP;
  } else if (x < 400) {
    return DOWN;
  } else if (x < 600) {
    return LEFT;
  } else if (x < 800) {
    return SELECT;
  }
  return NONE;
}

void loop() {
  static Mode mode = DISPLAYING;
  static Selection selection = BAUD_RATE;
  static Key lastKey = NONE;
  static uint64_t sameLastKeyCount = 0;

  Key key = getKey();
  switch (key) {
    case SELECT:
      switch (mode) {
        case SELECTION:
          mode = DISPLAYING;
          if (readU32FromEEPROM(BAUD_RATE_EEPROM_ADDRESS) != BAUD_RATES[baudRateIndex]) {
            writeU32ToEEPROM(BAUD_RATE_EEPROM_ADDRESS, BAUD_RATES[baudRateIndex]);
            mySerial.begin(BAUD_RATES[baudRateIndex]);
          }
          if (readU16FromEEPROM(PORT_NUMBER_EEPROM_ADDRESS) != portNumber) {
            writeU16ToEEPROM(PORT_NUMBER_EEPROM_ADDRESS, portNumber);
            delete server;
            server = new EthernetServer(portNumber);
          }
          showServerReadyIPAddress();
          break;
        case DISPLAYING:
          mode = SELECTION;
          switch (selection) {
            case BAUD_RATE:
              showBaudRate();
              break;
            case PORT_NUMBER:
              showPortNumber();
              break;
          }
          break;
      }
      break;
    case UP:
    case DOWN:
      if (mode == SELECTION) {
        switch (selection) {
          case BAUD_RATE:
            selection = PORT_NUMBER;
            showPortNumber();
            break;
          case PORT_NUMBER:
            selection = BAUD_RATE;
            showBaudRate();
            break;
        }
      }
      break;
    case LEFT:
      if (mode == SELECTION) {
        switch (selection) {
          case BAUD_RATE:
            if (!baudRateIndex) {
              while (BAUD_RATES[++baudRateIndex])
                ;
            }
            --baudRateIndex;
            showBaudRate(key == lastKey);
            break;
          case PORT_NUMBER:
            // accelerate the change
            portNumber -= sameLastKeyCount + 1;
            if (!portNumber) {
              portNumber = -1;
            }
            showPortNumber(key == lastKey);
            break;
        }
      }
      break;
    case RIGHT:
      if (mode == SELECTION) {
        switch (selection) {
          case BAUD_RATE:
            if (!BAUD_RATES[++baudRateIndex]) {
              baudRateIndex = 0;
            }
            showBaudRate(key == lastKey);
            break;
          case PORT_NUMBER:
            // accelerate the change
            portNumber += sameLastKeyCount + 1;
            if (!portNumber) {
              portNumber = 1;
            }
            showPortNumber(key == lastKey);
            break;
        }
      }
      break;
    default:
      // nothing happened
      break;
  }
  if (key != NONE) {
    if (key == lastKey && sameLastKeyCount + 1) {
      ++sameLastKeyCount;
    } else {
      sameLastKeyCount = 0;
    }
    unsigned long ms = 333 / (sameLastKeyCount + 1);
    if (ms) {
      delay(ms);
    }
  }
  lastKey = key;

  switch (Ethernet.maintain()) {
    case 1:
      //renewed fail
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.write("Error:");
      lcd.setCursor(0, 1);
      lcd.write("renewed failed");
      break;

    case 2:
      // renewed success
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.write("Renewed IP:");
      lcd.setCursor(0, 1);
      lcd.write(toString(Ethernet.localIP()).c_str());
      break;

    case 3:
      // rebind fail
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.write("Error:");
      lcd.setCursor(0, 1);
      lcd.write("rebind failed");
      break;

    case 4:
      //rebind success
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.write("Rebind IP:");
      lcd.setCursor(0, 1);
      lcd.write(toString(Ethernet.localIP()).c_str());
      break;

    default:
      // nothing happened
      break;
  }

  // listen for incoming clients
  EthernetClient client = server->available();
  if (client) {
    lcd.clear();
    // lcd.setCursor(0, 0);
    // lcd.write("New client:");
    // lcd.setCursor(0, 1);
    // lcd.write(toString(client.remoteIP()).c_str());

    uint8_t p0 = 0, p1 = 0;
    while (client.connected()) {

      while (client.available()) {
        uint8_t c = client.read();
        if (c == '\n' || c == '\r') {
          lcd.setCursor(p0, 0);
          while (p0 < 16) {
            p0 += lcd.write(' ');
          }
          p0 = 0;
        } else if (p0 < 16) {
          if (!p0) {
            lcd.setCursor(p0, 0);
            while (p0 < 16) {
              p0 += lcd.write(' ');
            }
            p0 = 0;
          }
          lcd.setCursor(p0, 0);
          p0 += lcd.write(c);
        }
        mySerial.write(c);
      }

      while (mySerial.available()) {
        uint8_t c = mySerial.read();
        if (c == '\n' || c == '\r') {
          lcd.setCursor(p1, 0);
          while (p1 < 16) {
            p1 += lcd.write(' ');
          }
          p1 = 0;
        } else if (p1 < 16) {
          if (!p1) {
            lcd.setCursor(p1, 1);
            while (p1 < 16) {
              p1 += lcd.write(' ');
            }
            p1 = 0;
          }
          lcd.setCursor(p1, 1);
          p1 += lcd.write(c);
        }
        client.write(c);
      }
    }
    // close the connection:
    client.stop();
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.write("Client");
    lcd.setCursor(0, 1);
    lcd.write("disconnected");
  }
}
