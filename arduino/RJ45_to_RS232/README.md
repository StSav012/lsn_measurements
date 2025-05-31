# Converter between RJ45 and RS232

The code enables Arduino to relay commands between a TCP socket and a device connected via a serial port.

## Hardware
* Arduino Uno or another 5-V board with the same main pins,
* [Arduino Ethernet Shield](https://docs.arduino.cc/retired/shields/arduino-ethernet-shield-without-poe-module/),
* a 16×2-char [LCD KeyPad Shield](https://wiki.dfrobot.com/LCD_KeyPad_Shield_For_Arduino_SKU__DFR0009) by DFRobot or something compatible.
* a MAX3232-based board to convert TTL into RS232 levels.

## Connections
* Place the Ethernet Shield on top of the Arduino board.
* The RJ45 socket is too high for the LCD KeyPad Shield to connect properly.
  So, solder an extender for the pins.
* Place the LCD KeyPad Shield on top of the Ethernet Shield through the extender.
* Solder additional pins to the LCD KeyPad Shield to connect the MAX3232-based board.
  Here, `A3` and `A4` pins are used to initialize the `SoftwareSerial` instance:
  ```arduino
  SoftwareSerial mySerial(A3, A4);  // RX, TX;
  ```
* Connect the pins (`A3` and `A4` here) and power to the MAX3232-based board.

## Features
* The code relies on DHCP to get and maintain the IP address.
* The data sent is displayed on the LCD.
* The TCP socket port and the serial baud rate are adjustable and are stored in EEPROM.
  Use the SELECT button to enter the settings, the UP/DOWN buttons to navigate within the settings, and the LEFT/RIGHT buttons to change the values.
