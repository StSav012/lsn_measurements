# coding: utf-8
from __future__ import annotations

import time
from math import nan
from socket import AF_INET, SOCK_STREAM, socket
from threading import Thread
from typing import Any

from utils.port_scanner import port_scanner

__all__ = ["Triton"]


class Triton(Thread):
    @staticmethod
    def heater_range(temperature: float) -> str:
        if temperature < 0.025:
            return "0.0316"
        if temperature < 0.04:
            return "0.1"
        if temperature < 0.1:
            return "0.316"
        if temperature < 0.7:
            return "1"
        if temperature < 0.9:
            return "3.16"
        return "10"

    @staticmethod
    def filter_readings(temperature: float) -> bool:
        if temperature < 0.2:
            return False
        return True

    def __init__(self, ip: str | None, port: int) -> None:
        self.socket: socket | None = None
        if ip is None:
            from ipaddress import IPv4Address

            connectable_hosts: list[IPv4Address] = port_scanner(port)
            if not connectable_hosts:
                raise RuntimeError("Triton could not be found automatically. Try specifying an IP address.")
            if len(connectable_hosts) > 1:
                raise RuntimeError(
                    f"There are numerous devices with open port {port}:\n",
                    ",\n".join(map(str, connectable_hosts)),
                    "\nTry specifying an IP address.",
                )
            ip = str(connectable_hosts[0])

        super().__init__()
        self.daemon = True

        self.conversation: dict[str, str] = dict()
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((ip, port))

        self._running: bool = True
        self._issuing: bool = False

        self.start()

    def __del__(self) -> None:
        self._running = False
        if self.socket is not None:
            self.socket.close()

    def communicate(self, command: str) -> str:
        if self.socket is None:
            return ""
        while self._issuing:
            time.sleep(0.1)
        self._issuing = True
        self.socket.send((command.strip() + "\r\n").encode())
        resp: bytes = b""
        while (not resp) or resp[-1] != 10:
            resp += self.socket.recv(1)
            if not resp:
                self._issuing = False
                if command.startswith("READ:"):
                    self.conversation[command] = ""
                return ""
        self._issuing = False
        if command.startswith("READ:"):
            self.conversation[command] = resp.decode().strip()
        return resp.decode().strip()

    def query(self, command: str, blocking: bool = False) -> str:
        if blocking:
            return self.communicate(command)
        if command not in self.conversation:
            self.conversation[command.strip()] = ""
        return self.conversation[command.strip()]

    def query_value(self, command: str, blocking: bool = False) -> tuple[float, str]:
        if not command.startswith("READ:"):
            command = "READ:" + command
        response: str
        if blocking:
            response = self.communicate(command)
        else:
            response = self.query(command)
        if not response:
            return nan, ""
        response_start: str = ":".join(["STAT"] + command.split(":")[1:]) + ":"
        if not response.startswith(response_start):
            print(command, "->", response)
            return nan, ""
        response = response[len(response_start) :]
        unit: str = ""
        while response and response[-1] not in "1234567890-+.":
            unit = response[-1] + unit
            response = response[:-1]
        if not response:
            return nan, unit
        return float(response), unit

    def query_temperature(self, index: int, blocking: bool = False) -> tuple[float, str]:
        return self.query_value(f"READ:DEV:T{index}:TEMP:SIG:TEMP", blocking=blocking)

    def issue_value(self, command: str, value: Any) -> bool:
        if not command.startswith("SET:"):
            command = "SET:" + command
        if isinstance(value, bool):
            value = {False: "OFF", True: "ON"}[value]
        response: str = self.query(command.strip(":") + ":" + str(value), blocking=True)
        if not response:
            return False
        response_start: str = ":".join(["STAT"] + command.split(":")) + ":"
        if not response.startswith(response_start):
            print(command, "->", response)
            return False
        return response.endswith(":VALID")

    def issue_temperature(self, index: int, value: float) -> bool:
        return self.issue_value(f"SET:DEV:T{index}:TEMP:LOOP:T" "SET", value)

    def issue_filter_readings(self, index: int, value: bool) -> bool:
        return self.issue_value(f"SET:DEV:T{index}:TEMP:LOOP:FILT:ENAB", value)

    def issue_heater_range(self, index: int, value: str) -> bool:
        return self.issue_value(f"SET:DEV:T{index}:TEMP:LOOP:RANGE", value)

    def run(self) -> None:
        while self._running:
            for command in list(self.conversation):
                self.communicate(command)
            time.sleep(1)


if __name__ == "__main__":
    t: Triton = Triton(None, 33576)
    # for _ in range(3):
    #     print(_, *t.query_value('READ:DEV:T6:TEMP:SIG:TEMP'))
    #     print(_, *t.query_value('READ:DEV:T6:TEMP:LOOP:TSET'))
    #     print(_, *t.query_value('SET:DEV:T6:TEMP:LOOP:TSET:0.9'))
    #     time.sleep(2)
    print(t.communicate("READ:DEV:T6:TEMP:LOOP:RANGE"))
    # print(t.issue_value('SET:DEV:T6:TEMP:LOOP:RANGE', '0.316'))
