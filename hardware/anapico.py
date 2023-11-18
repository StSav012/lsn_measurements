# coding: utf-8
from __future__ import annotations

from math import isnan, nan
from typing import Final

from .scpi_device import SCPIDevice

__all__ = ["APUASYN20"]


class _AmplitudeModulation:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent

    def __bool__(self) -> bool:
        return self.state

    @property
    def state(self) -> bool:
        if self._parent.socket is None:
            return False
        return bool(int(self._parent.query(":am:state")))

    @state.setter
    def state(self, new_value: bool) -> None:
        self._parent.issue(":am:state", bool(new_value))

    @property
    def source(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":am:source")

    @source.setter
    def source(self, new_value: str) -> None:
        if new_value.casefold() == "ext":
            new_value = "external"
        elif new_value.casefold() == "int":
            new_value = "internal"
        elif new_value.casefold() not in ("external", "internal"):
            raise ValueError(f"Invalid AM Source: {new_value}")
        self._parent.issue(":am:source", new_value)

    @property
    def depth(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(":am:depth"))

    @depth.setter
    def depth(self, new_value: float) -> None:
        """Sets the amplitude modulation depth [0..0.99] when modulated by the internal source."""
        if not (0 <= new_value <= 0.99):
            raise ValueError(f"Invalid AM Depth: {new_value}")
        self._parent.issue(":am:depth", new_value)

    @property
    def sensitivity(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(":am:sensitivity"))

    @sensitivity.setter
    def sensitivity(self, new_value: float) -> None:
        """Sets the sensitivity of the external signal source for amplitude modulation [0..3/V]."""
        if not (0.0 <= new_value <= 3.0):
            raise ValueError(f"Invalid AM Sensitivity: {new_value}")
        self._parent.issue(":am:sensitivity", new_value)

    @property
    def frequency(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(":am:internal:frequency"))

    @frequency.setter
    def frequency(self, new_value: float) -> None:
        """Sets the internal amplitude modulation rate [10..50000 Hz]."""
        if not (10.0 <= new_value <= 50_000.0):
            raise ValueError(f"Invalid Internal AM Frequency: {new_value}")
        self._parent.issue(":am:internal:frequency", new_value)


class _LFOutput:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent

    def __bool__(self) -> bool:
        return self.state

    @property
    def state(self) -> bool:
        if self._parent.socket is None:
            return False
        return bool(int(self._parent.query(":lfOutput:state")))

    @state.setter
    def state(self, new_value: bool) -> None:
        """Activates LF signal output."""
        self._parent.issue(":lfOutput:state", bool(new_value))

    @property
    def source(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":lfOutput:source")

    @source.setter
    def source(self, new_value: str) -> None:
        """Determines the LF signal to be synchronized, when monitoring is enabled."""
        if new_value.casefold() == "lfg":
            new_value = "lf" "generator"
        elif new_value.casefold() == "trig":
            new_value = "trigger"
        elif new_value.casefold() not in ("lf" "generator", "pul" "m", "trigger"):
            raise ValueError(f"Invalid LF Output Source: {new_value}")
        self._parent.issue(":lfOutput:source", new_value)


class _PulseModulationInternal:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent

    @property
    def frequency(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(":pul" "m:internal:frequency"))

    @frequency.setter
    def frequency(self, new_value: float) -> None:
        if isnan(new_value):
            return
        self._parent.issue(":pul" "m:internal:frequency", new_value)

    @property
    def period(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(":pul" "m:internal:period"))

    @period.setter
    def period(self, new_value: float) -> None:
        if isnan(new_value):
            return
        self._parent.issue(":pul" "m:internal:period", new_value)

    @property
    def pulse_width(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(":pul" "m:internal:p" "width"))

    @pulse_width.setter
    def pulse_width(self, new_value: float) -> None:
        if isnan(new_value):
            return
        self._parent.issue(":pul" "m:internal:p" "width", new_value)


class _PulseModulation:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent
        self.internal: Final[_PulseModulationInternal] = _PulseModulationInternal(parent)

    def __bool__(self) -> bool:
        return self.state

    @property
    def state(self) -> bool:
        if self._parent.socket is None:
            return False
        return bool(int(self._parent.query(":pul" "m:state")))

    @state.setter
    def state(self, new_value: bool) -> None:
        """Activates pulse modulation."""
        self._parent.issue(":pul" "m:state", bool(new_value))

    @property
    def source(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":pul" "m:source")

    @source.setter
    def source(self, new_value: str) -> None:
        """Selects between the internal Pulse Generator or an External pulse signal for the modulation."""
        if new_value.casefold() == "ext":
            new_value = "external"
        elif new_value.casefold() == "int":
            new_value = "internal"
        elif new_value.casefold() == "bits":
            new_value = "bitstream"
        elif new_value.casefold() not in ("internal", "external", "bitstream"):
            raise ValueError(f"Invalid Pulse Modulation Transition Mode: {new_value}")
        self._parent.issue(":pul" "m:source", new_value)

    @property
    def polarity(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":pul" "m:polarity")

    @polarity.setter
    def polarity(self, new_value: str) -> None:
        """Sets the polarity of the externally applied modulation signal."""
        if new_value.casefold() == "norm":
            new_value = "normal"
        elif new_value.casefold() == "inv":
            new_value = "inverted"
        elif new_value.casefold() not in ("normal", "inverted"):
            raise ValueError(f"Invalid Pulse Modulation Polarity: {new_value}")
        self._parent.issue(":pul" "m:polarity", new_value)


class _ALC:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent

    def __bool__(self) -> bool:
        return self.state

    @property
    def state(self) -> bool:
        if self._parent.socket is None:
            return False
        return bool(int(self._parent.query(":power:alc:state")))

    @state.setter
    def state(self, new_value: bool) -> None:
        """Turns the ALC (automatic levelling control) on or off.
        Specified output power is guaranteed only with ALC on."""
        self._parent.issue(":power:alc:state", bool(new_value))

    @property
    def low_noise(self) -> bool:
        if self._parent.socket is None:
            return False
        return bool(int(self._parent.query(":power:alc:low" "noise")))

    @low_noise.setter
    def low_noise(self, new_value: bool) -> None:
        """Enables or disables the low amplitude noise mode providing up to 1/1000 dB output power resolution.
        When enabled, the automatic levelling control will work in a mode similar to hold.
        """
        self._parent.issue(":power:alc:low" "noise", bool(new_value))


class _TriggerOutput:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent

    @property
    def polarity(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":trigger:output:polarity")

    @polarity.setter
    def polarity(self, new_value: str) -> None:
        """Sets the trigger output signal polarity."""
        if new_value.casefold() == "norm":
            new_value = "normal"
        elif new_value.casefold() == "inv":
            new_value = "inverted"
        elif new_value.casefold() not in ("normal", "inverted"):
            raise ValueError(f"Invalid Trigger Output Polarity: {new_value}")
        self._parent.issue(":trigger:output:polarity", new_value)

    @property
    def mode(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":trigger:output:mode")

    @mode.setter
    def mode(self, new_value: str) -> None:
        """Sets the trigger output signal mode."""
        if new_value.casefold() == "norm":
            new_value = "normal"
        elif new_value.casefold() == "point"[:-1]:
            new_value = "point"
        elif new_value.casefold() == "val":
            new_value = "valid"
        elif new_value.casefold() not in ("normal", "gate", "point", "valid"):
            raise ValueError(f"Invalid Trigger Output Mode: {new_value}")
        self._parent.issue(":trigger:output:mode", new_value)

    @property
    def source(self) -> int | str | None:
        if self._parent.socket is None:
            return None
        value: str = self._parent.query(":trigger:output:source")
        if value.isnumeric():
            return int(value)
        return value

    @source.setter
    def source(self, new_value: int | str) -> None:
        """Selects the source channel for the trigger output and the RF output valid signal."""
        if isinstance(new_value, str) and new_value.casefold() != "all":
            raise ValueError(f"Invalid Trigger Output Source: {new_value}")
        self._parent.issue(":trigger:output:source", str(new_value))


class _Trigger:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent
        self.output: Final[_TriggerOutput] = _TriggerOutput(self._parent)

    def immediate(self) -> None:
        """Triggers the device immediately if it is configured to wait for trigger events."""
        self._parent.communicate(":trigger:immediate")

    @property
    def type(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":trigger:type")

    @type.setter
    def type(self, new_value: str) -> None:
        """Sets the trigger type that controls the waveform’s playback."""
        if new_value.casefold() == "norm":
            new_value = "normal"
        elif new_value.casefold() == "point"[:-1]:
            new_value = "point"
        elif new_value.casefold() not in ("normal", "gate", "point"):
            raise ValueError(f"Invalid Trigger Type: {new_value}")
        self._parent.issue(":trigger:type", new_value)

    @property
    def source(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":trigger:source")

    @source.setter
    def source(self, new_value: str) -> None:
        """Sets the trigger source."""
        if new_value.casefold() == "imm":
            new_value = "immediate"
        elif new_value.casefold() not in ("immediate", "key", "ext", "bus"):
            raise ValueError(f"Invalid Trigger Source: {new_value}")
        self._parent.issue(":trigger:source", new_value)

    @property
    def delay(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(":trigger:delay"))

    @delay.setter
    def delay(self, new_value: float) -> None:
        """Sets the amount of time to delay the synthesizer response to an external trigger."""
        if isnan(new_value):
            return
        self._parent.issue(":trigger:delay", new_value)

    @property
    def slope(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":trigger:slope")

    @slope.setter
    def slope(self, new_value: str) -> None:
        """Sets the polarity for an external trigger signal while using the continuous, single triggering mode."""
        if new_value.casefold() == "pos":
            new_value = "positive"
        if new_value.casefold() == "neg":
            new_value = "negative"
        elif new_value.casefold() not in ("positive", "negative", "np", "pn"):
            raise ValueError(f"Invalid Trigger Slope: {new_value}")
        self._parent.issue(":trigger:slope", new_value)

    @property
    def every_count(self) -> int:
        if self._parent.socket is None:
            return -1
        return int(self._parent.query(":trigger:e" "count"))

    @every_count.setter
    def every_count(self, new_value: int) -> None:
        """Sets a modulus counter on consecutive trigger events.
        Setting the value to N means that only every Nth trigger event will be considered.
        Setting it to one means will use every trigger event that does not occur during a running sweep.
        """
        if new_value < 1 or new_value > 255:
            raise ValueError(f"Invalid Trigger E.Count: {new_value}")
        self._parent.issue(":trigger:e" "count", new_value)


class _Power:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent
        self.alc: Final[_ALC] = _ALC(self._parent)

    def __float__(self) -> float:
        return self.level

    @property
    def level(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query("power"))

    @level.setter
    def level(self, new_value: float) -> None:
        if isnan(new_value):
            return
        self._parent.issue("power", new_value)


class _Init:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent

    def immediate(self) -> None:
        """Sets trigger to the armed state."""
        self._parent.communicate(":init:immediate")

    @property
    def continuous(self) -> bool:
        if self._parent.socket is None:
            return False
        return bool(int(self._parent.query(":init:continuous")))

    @continuous.setter
    def continuous(self, new_value: bool) -> None:
        """Continuously rearms the trigger system after completion of a triggered sweep."""
        self._parent.issue(":init:continuous", bool(new_value))


class _SystemError:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent

    @property
    def next(self) -> tuple[int, str]:
        if self._parent.socket is None:
            return 0, ""
        response: list[str] = self._parent.query("system:error:next").split(",", maxsplit=1)
        return int(response[0]), response[1].strip('"')

    @property
    def all(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query("system:error:all")


class _System:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent
        self.error: Final[_SystemError] = _SystemError(parent)

    def preset(self) -> None:
        """Resets most signal generator functions to factory- defined conditions.
        This command is similar to the *RST command."""
        self._parent.communicate("system:preset")

    @property
    def version(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query("system:version")

    def lock(self) -> None:
        """Locks (disables) front panel control."""
        self._parent.communicate("system:lock")

    def unlock(self) -> None:
        """Unlocks (enables) front panel control."""
        self._parent.communicate("system:lock:release")


class APUASYN20(SCPIDevice):
    _PORT: Final[int] = 18  # always port 18

    def __init__(self, ip: str | None = None, *, expected: bool = True) -> None:
        super().__init__(ip, APUASYN20._PORT, terminator=b"\n", expected=expected)
        self.am: Final[_AmplitudeModulation] = _AmplitudeModulation(self)
        self.lf_output: Final[_LFOutput] = _LFOutput(self)
        self.pulse_modulation: Final[_PulseModulation] = _PulseModulation(self)
        self.trigger: Final[_Trigger] = _Trigger(self)
        self.power: Final[_Power] = _Power(self)
        self.init: Final[_Init] = _Init(self)
        self.system: Final[_System] = _System(self)

    @property
    def output(self) -> bool:
        if self.socket is None:
            return False
        return bool(int(self.query(":output1:state")))

    @output.setter
    def output(self, new_value: bool) -> None:
        self.issue(":output1:state", bool(new_value))

    @property
    def frequency(self) -> float:
        if self.socket is None:
            return nan
        return float(self.query("frequency"))

    @frequency.setter
    def frequency(self, new_value: float) -> None:
        if isnan(new_value):
            return
        self.issue("frequency", new_value)


if __name__ == "__main__":
    s: APUASYN20 = APUASYN20()
    print(s.idn)
