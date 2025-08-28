import time
from collections import OrderedDict
from datetime import UTC, datetime
from math import nan
from socket import AF_INET, SOCK_STREAM, socket
from threading import Condition, Lock, Thread
from typing import Any, AnyStr, ClassVar, Final, assert_type

from astropy.units import Quantity

from utils import error
from utils.port_scanner import port_scanner

__all__ = ["Triton", "TritonScript"]


class Triton(Thread):
    @staticmethod
    def heater_range(temperature: float | Quantity) -> str:
        if isinstance(temperature, float):
            temperature = Quantity(temperature, "K")
        if temperature < Quantity(0.025, "K"):
            return "0.0316"
        if temperature < Quantity(0.04, "K"):
            return "0.1"
        if temperature < Quantity(0.1, "K"):
            return "0.316"
        if temperature < Quantity(0.7, "K"):
            return "1"
        if temperature < Quantity(0.9, "K"):
            return "3.16"
        return "10"

    @staticmethod
    def filter_readings(temperature: float | Quantity) -> bool:
        if isinstance(temperature, float):
            temperature = Quantity(temperature, "K")
        return not temperature < Quantity(0.2, "K")

    def __init__(self, ip: str | None = None, port: int = 33576) -> None:
        self.socket: socket = socket(AF_INET, SOCK_STREAM)
        self.conversation: dict[str, str] = {}

        self._issuing: Lock = Lock()
        self._has_pending_data: Condition = Condition()

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

        super().__init__(daemon=True)

        self.socket.connect((ip, port))

        self.start()

    def __del__(self) -> None:
        self.socket.close()

    def communicate(self, command: str) -> str:
        with self._issuing:
            self.socket.send((command.strip() + "\r\n").encode())
            resp: bytes = b""
            while (not resp) or resp[-1] != 10:
                try:
                    resp += self.socket.recv(1)
                except ConnectionResetError:
                    error(self.socket.getpeername())
                if not resp and command.startswith("READ:"):
                    self.conversation[command] = ""
                    with self._has_pending_data:
                        self._has_pending_data.notify()
                    return ""
        if command.startswith("READ:"):
            self.conversation[command] = resp.decode().strip()
            with self._has_pending_data:
                self._has_pending_data.notify()
        return resp.decode().strip()

    def query(self, command: str, blocking: bool = False) -> str:
        if blocking or command not in self.conversation:
            return self.communicate(command)
        return self.conversation.get(command.strip(), "")

    def query_value(self, command: str, blocking: bool = False) -> bool | Quantity:
        if not command.startswith("READ:"):
            command = "READ:" + command
        response: str = self.communicate(command) if blocking else self.query(command)
        if not response:
            return Quantity(nan)
        response_start: str = ":".join(["STAT"] + command.split(":")[1:]) + ":"
        if not response.startswith(response_start):
            print(command, "->", response)
            return Quantity(nan)
        response = response[len(response_start) :]
        if response == "OFF":
            return False
        if response == "ON":
            return True
        return Quantity(response)

    def query_temperature(self, index: int, blocking: bool = False) -> Quantity:
        return self.query_value(f"READ:DEV:T{index}:TEMP:SIG:TEMP", blocking=blocking)

    def query_heater_power(self, index: int, blocking: bool = False) -> Quantity:
        return self.query_value(f"READ:DEV:H{index}:HTR:SIG:POWR", blocking=blocking)

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

    def issue_temperature(self, index: int, value: float | Quantity) -> bool:
        if isinstance(value, Quantity):
            value = value.to_value("K")
        return self.issue_value(f"SET:DEV:T{index}:TEMP:LOOP:TSET", value)

    def ensure_temperature(self, index: int, value: float | Quantity) -> bool:
        if isinstance(value, Quantity):
            value = value.to_value("K")
        if value == self.query_value(f"READ:DEV:T{index}:TEMP:LOOP:TSET").value:
            return True
        return self.issue_value(f"SET:DEV:T{index}:TEMP:LOOP:TSET", value)

    def issue_filter_readings(self, index: int, value: bool) -> bool:
        return self.issue_value(f"SET:DEV:T{index}:TEMP:LOOP:FILT:ENAB", value)

    def ensure_filter_readings(self, index: int, value: bool) -> bool:
        if value == self.query_value(f"READ:DEV:T{index}:TEMP:LOOP:FILT:ENAB"):
            return True
        return self.issue_value(f"SET:DEV:T{index}:TEMP:LOOP:FILT:ENAB", value)

    def issue_heater_range(self, index: int, value: str) -> bool:
        return self.issue_value(f"SET:DEV:T{index}:TEMP:LOOP:RANGE", value)

    def ensure_heater_range(self, index: int, value: str) -> bool:
        if value == self.query_value(f"READ:DEV:T{index}:TEMP:LOOP:RANGE").value:
            return True
        return self.issue_value(f"SET:DEV:T{index}:TEMP:LOOP:RANGE", value)

    def issue_heater_power(self, index: int, value: float | Quantity) -> bool:
        if isinstance(value, Quantity):
            value = value.to_value("uW")
        return self.issue_value(f"SET:DEV:H{index}:HTR:SIG:POWR", value)

    def run(self) -> None:
        while self.is_alive():
            with self._has_pending_data:
                self._has_pending_data.wait_for(self.conversation.__len__, 0.1)
                for command in list(self.conversation):
                    self.communicate(command)
                time.sleep(1)


class TritonScript(socket):
    _end: ClassVar[bytes] = b"<end>"

    def __init__(self, ip: str | None = None, port: int = 22518) -> None:
        super().__init__(AF_INET, SOCK_STREAM)
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
        self.connect((ip, port))

    def __del__(self) -> None:
        self.close()

    def communicate(self, command: AnyStr) -> list[bytes]:
        if isinstance(command, str):
            command = command.encode()
        assert_type(command, bytes)
        self.send(command.strip() + b"\r\n")
        resp: bytes = b""
        while not resp.endswith(TritonScript._end):
            resp += self.recv(1)
            if not resp:
                return resp
        lines: list[bytes] = resp.splitlines()
        if lines[0] != command:
            raise ValueError(f"Expected the response to start with {command!r}, got {lines[0]!r}")
        if lines[-1] != TritonScript._end:
            raise ValueError(f"Expected the response to end with {TritonScript._end!r}, got {lines[-1]!r}")
        if not lines[1].startswith(b"<"):
            raise ValueError(f"Unexpected response line: {lines[1]!r}")
        return lines[2:-1]

    @property
    def status(self) -> dict[str, str]:
        cmd: Final[bytes] = b"status"
        data: OrderedDict[str, str] = OrderedDict()
        line: bytes
        for line in self.communicate(cmd):
            if b" is " in line:
                key: bytes
                value: bytes
                key, value = line.split(b" is ", maxsplit=1)
                data[key.decode()] = value.decode()
            elif line.endswith(b" not found"):
                data[line.removesuffix(b" not found").decode()] = "not found"
        return data

    @property
    def pressures(self) -> dict[str, Quantity]:
        cmd: Final[bytes] = b"pressures"
        data: OrderedDict[str, Quantity] = OrderedDict()
        line: bytes
        for line in self.communicate(cmd):
            if b": " in line:
                key: bytes
                value: bytes
                key, value = line.split(b": ", maxsplit=1)
                data[key.decode()] = Quantity(value.decode())
        return data

    @property
    def thermometry(self) -> tuple[list[dict[str, float | str | bool | datetime]], dict[str, Quantity]]:
        cmd: Final[bytes] = b"thermometry"
        ch_data: list[dict[str, float | str | bool | datetime]] = []
        data: OrderedDict[str, Quantity] = OrderedDict()
        line: bytes
        for line in self.communicate(cmd):
            if b": " not in line:
                continue
            line = line.rstrip(b";")
            if line.startswith(b"channel"):
                parts: list[str] = line.decode().split("; ")
                ch_data_part: OrderedDict[str, float | str | bool | datetime] = OrderedDict()
                for part in parts:
                    if ":" not in part:
                        continue
                    part_key: str
                    part_value: str
                    part_key, part_value = part.split(": ", maxsplit=1)
                    if part_key == "enabled":
                        ch_data_part[part_key] = bool(int(part_value))
                    elif part_key == "time":
                        ch_data_part[part_key] = datetime.fromtimestamp(float(part_value), tz=UTC)
                    else:
                        try:
                            ch_data_part[part_key] = float(part_value)
                        except ValueError:
                            ch_data_part[part_key] = part_value
                ch_data.append(ch_data_part)
            else:
                key: bytes
                value: bytes
                key, value = line.split(b": ", maxsplit=1)
                data[key.decode()] = Quantity(value.decode())
        return ch_data, data


if __name__ == "__main__":
    from pprint import pp

    t: Triton = Triton()
    for _ in range(30):
        print(_, t.query_value("READ:DEV:T6:TEMP:SIG:TEMP"))
        print(_, t.query_value("READ:DEV:T6:TEMP:LOOP:TSET"))
        print(_, t.communicate("SET:DEV:T6:TEMP:LOOP:TSET:0.9"))
        time.sleep(0.2)
    pp(t.communicate("READ:DEV:T6:TEMP:LOOP:RANGE"))
    pp(t.issue_value("SET:DEV:T6:TEMP:LOOP:RANGE", "0.316"))

    # ts: TritonScript = TritonScript("triton")
    # pp(ts.status)
    # pp(ts.pressures)
    # pp(ts.thermometry)
