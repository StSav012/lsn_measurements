from math import nan
from typing import Final, NamedTuple

from vxi11.vxi11 import Instrument

try:
    from .scpi_device import SCPIDevice
except ImportError:
    from scpi_device import SCPIDevice

__all__ = ["GSM7"]


class ReadResult(NamedTuple):
    voltage: float = nan  # [V]
    current: float = nan  # [A]

    @property
    def resistance(self) -> float:
        return self.voltage / self.current

    @property
    def power(self) -> float:
        return self.voltage * self.current

    @classmethod
    def from_string(cls, string: str) -> "ReadResult":
        parts: list[str] = string.split(",")
        length: int = len(cls())
        if len(parts) < length:
            parts += [nan] * (length - len(parts))
        return cls(*map(float, parts[:length]))


class _Calculate:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent


class _Calculate2:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent


class _Calculate3:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent


class _Display:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent


class _Format:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent


class _Output:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent

    def __bool__(self) -> bool:
        return self.state

    @property
    def state(self) -> bool:
        if getattr(self._parent, "_instr", None) is None:
            return False
        return bool(int(self._parent.query(":output")))

    @state.setter
    def state(self, new_value: bool) -> None:
        self._parent.issue(":output", bool(new_value))


class _Route:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent


class _Source:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent


class _Source2:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent


class _Status:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent


class _System:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent


class _Trace:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent


class _Trigger:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent


class _Arm:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent


class GSM7(SCPIDevice):
    _PORT: Final[int] = 111

    def __init__(self, ip: str | None = None, *, expected: bool = True) -> None:
        super().__init__(ip, GSM7._PORT, terminator=b"\n", expected=expected, reset=False)
        self._instr: Instrument | None = None
        if self.socket is not None:
            self.socket.close()
            self._instr = Instrument(self.socket.getpeername()[0], term_char=self.terminator)
            self.socket = None

        self.calculate: Final[_Calculate] = _Calculate(self)
        self.calculate1: Final[_Calculate] = self.calculate
        self.calculate2: Final[_Calculate2] = _Calculate2(self)
        self.calculate3: Final[_Calculate3] = _Calculate3(self)
        self.display: Final[_Display] = _Display(self)
        self.format: Final[_Format] = _Format(self)
        self.output: Final[_Output] = _Output(self)
        self.route: Final[_Route] = _Route(self)
        self.source: Final[_Source] = _Source(self)
        self.source1: Final[_Source] = self.source
        self.source2: Final[_Source2] = _Source2(self)
        self.status: Final[_Status] = _Status(self)
        self.system: Final[_System] = _System(self)
        self.trace: Final[_Trace] = _Trace(self)
        self.trigger: Final[_Trigger] = _Trigger(self)
        self.arm: Final[_Arm] = _Arm(self)

    def communicate(self, command: str) -> str | None:
        if self._instr is None:
            return ""
        if command.rstrip().endswith("?"):
            return self._instr.ask(command)
        return self._instr.write(command)

    @property
    def read(self) -> ReadResult:
        if not self.output:
            return ReadResult()
        return ReadResult.from_string(self.communicate(":read?"))

    @property
    def current(self) -> float:
        return self.read.current

    @property
    def voltage(self) -> float:
        return self.read.voltage

    @property
    def resistance(self) -> float:
        return self.read.resistance

    @property
    def power(self) -> float:
        return self.read.power


if __name__ == "__main__":
    g: GSM7 = GSM7()
    print(g.idn)
