# coding: utf-8
from __future__ import annotations

from math import nan
from typing import Final, Literal

from .scpi_device import SCPIDevice

__all__ = ["SMA100B"]


class _AmplitudeModulationPerChannel:
    def __init__(self, parent: SCPIDevice, channel: Literal[1, 2]) -> None:
        self._parent: Final[SCPIDevice] = parent
        self._channel: Literal[1, 2] = channel
        if self._channel not in (1, 2):
            raise ValueError(f"Invalid channel: {channel}")

    @property
    def state(self) -> bool:
        if self._parent.socket is None:
            return False
        return bool(int(self._parent.query(f":am{self._channel}:state")))

    @state.setter
    def state(self, new_value: bool) -> None:
        self._parent.issue(f":am{self._channel}:state", int(bool(new_value)))

    @property
    def source(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(f":am{self._channel}:source")

    @source.setter
    def source(self, new_value: str) -> None:
        if new_value.casefold() == "ext":
            new_value = "external"
        elif new_value.casefold() == "int":
            new_value = "internal"
        elif new_value.casefold() not in ("external", "internal"):
            raise ValueError(f"Invalid AM Source: {new_value}")
        self._parent.issue(f":am{self._channel}:source", new_value)

    @property
    def depth(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(f":am{self._channel}:depth"))

    @depth.setter
    def depth(self, new_value: float) -> None:
        if not (0 <= new_value <= 100):
            raise ValueError(f"Invalid AM Depth: {new_value}")
        self._parent.issue(f":am{self._channel}:depth", new_value)

    @property
    def linear_depth(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(f":am{self._channel}:depth:linear"))

    @linear_depth.setter
    def linear_depth(self, new_value: float) -> None:
        """Sets the depth of the linear amplitude modulation in percent / volt."""
        if not (0 <= new_value <= 100):
            raise ValueError(f"Invalid AM Depth: {new_value}")
        self._parent.issue(f":am{self._channel}:depth:linear", new_value)

    @property
    def exponential_depth(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(f":am{self._channel}:depth:exponential"))

    @exponential_depth.setter
    def exponential_depth(self, new_value: float) -> None:
        """Sets the depth of the exponential amplitude modulation in dB/volt."""
        if not (0 <= new_value <= 100):
            raise ValueError(f"Invalid AM Depth: {new_value}")
        self._parent.issue(f":am{self._channel}:depth:exponential", new_value)

    @property
    def linear_sensitivity(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(f":am{self._channel}:sensitivity:linear"))

    @linear_sensitivity.setter
    def linear_sensitivity(self, new_value: float) -> None:
        """Sets the sensitivity of the external signal source for amplitude modulation."""
        if not (0 <= new_value <= 100):
            raise ValueError(f"Invalid AM Sensitivity: {new_value}")
        self._parent.issue(f":am{self._channel}:sensitivity:linear", new_value)

    @property
    def exponential_sensitivity(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(f":am{self._channel}:sensitivity:exponential"))

    @exponential_sensitivity.setter
    def exponential_sensitivity(self, new_value: float) -> None:
        """Sets the sensitivity of the external signal source for amplitude modulation."""
        if not (0 <= new_value <= 100):
            raise ValueError(f"Invalid AM Sensitivity: {new_value}")
        self._parent.issue(f":am{self._channel}:sensitivity:exponential", new_value)


class _AmplitudeModulation:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent
        self.channel1: Final[
            _AmplitudeModulationPerChannel
        ] = _AmplitudeModulationPerChannel(parent, 1)
        self.channel2: Final[
            _AmplitudeModulationPerChannel
        ] = _AmplitudeModulationPerChannel(parent, 2)

    @property
    def am_mode(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":am:mode")

    @am_mode.setter
    def am_mode(self, new_value: str) -> None:
        """Selects the mode of the amplitude modulation: scan or normal."""
        if new_value.casefold() not in ("scan", "norm", "normal"):
            raise ValueError(f"Invalid AM Mode: {new_value}")
        self._parent.issue(":am:mode", new_value)

    @property
    def am_type(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":am:type")

    @am_type.setter
    def am_type(self, new_value: str) -> None:
        """Selects the type of amplitude modulation: linear or exponential."""
        if new_value.casefold() not in ("lin", "exp", "linear", "exponential"):
            raise ValueError(f"Invalid AM Type: {new_value}")
        self._parent.issue(":am:type", new_value)

    @property
    def am_deviation_mode(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":am:deviation:mode")

    @am_deviation_mode.setter
    def am_deviation_mode(self, new_value: str) -> None:
        """Selects the coupling mode.
        The coupling mode parameter also determines the mode for fixing the total depth.
        """
        if new_value.casefold() not in ("uncoupled", "total", "ratio"):
            raise ValueError(f"Invalid AM Deviation Mode: {new_value}")
        self._parent.issue(":am:deviation:mode", new_value)


class _LFOutputPerChannel:
    def __init__(self, parent: SCPIDevice, channel: Literal[1, 2]) -> None:
        self._parent: Final[SCPIDevice] = parent
        self._channel: Literal[1, 2] = channel
        if self._channel not in (1, 2):
            raise ValueError(f"Invalid channel: {channel}")

    @property
    def state(self) -> bool:
        if self._parent.socket is None:
            return False
        return bool(int(self._parent.query(f":lfOutput{self._channel}")))

    @state.setter
    def state(self, new_value: bool) -> None:
        """Activates LF signal output."""
        self._parent.issue(f":lfOutput{self._channel}", int(bool(new_value)))

    @property
    def source(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(f":lfOutput{self._channel}:source")

    @source.setter
    def source(self, new_value: str) -> None:
        """Determines the LF signal to be synchronized, when monitoring is enabled."""
        if new_value.casefold() not in (
            "lf1",
            "lf2",
            "noise",
            "am",
            "fm" "pm",
            "ext1",
            "ext2",
        ):
            raise ValueError(f"Invalid LF Output Source: {new_value}")
        self._parent.issue(f":lfOutput{self._channel}:source", new_value)


class _LFOutput:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent
        self.channel1: Final[_LFOutputPerChannel] = _LFOutputPerChannel(parent, 1)
        self.channel2: Final[_LFOutputPerChannel] = _LFOutputPerChannel(parent, 2)


class _ExternalModulationPerChannel:
    def __init__(self, parent: SCPIDevice, channel: Literal[1, 2]) -> None:
        self._parent: Final[SCPIDevice] = parent
        self._channel: Literal[1, 2] = channel
        if self._channel not in (1, 2):
            raise ValueError(f"Invalid channel: {channel}")

    @property
    def coupling(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(f":input:modExt:coupling{self._channel}")

    @coupling.setter
    def coupling(self, new_value: str) -> None:
        """Selects the coupling mode for an externally applied modulation signal: AC or DC."""
        if new_value.casefold() not in ("ac", "dc"):
            raise ValueError(f"Invalid Coupling: {new_value}")
        self._parent.issue(f":input:modExt:coupling{self._channel}", new_value)


class _ExternalModulation:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent
        self.channel1: Final[
            _ExternalModulationPerChannel
        ] = _ExternalModulationPerChannel(parent, 1)
        self.channel2: Final[
            _ExternalModulationPerChannel
        ] = _ExternalModulationPerChannel(parent, 2)


class _Input:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent
        self.external_modulation: Final[_ExternalModulation] = _ExternalModulation(
            parent
        )


class _PulseModulation:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: Final[SCPIDevice] = parent

    @property
    def state(self) -> bool:
        if self._parent.socket is None:
            return False
        return bool(int(self._parent.query(":pul" "m:state")))

    @state.setter
    def state(self, new_value: bool) -> None:
        """Activates pulse modulation."""
        self._parent.issue(":pul" "m:state", int(bool(new_value)))

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
        if new_value.casefold() not in ("internal", "external"):
            raise ValueError(f"Invalid Pulse Modulation Transition Mode: {new_value}")
        self._parent.issue(":pul" "m:source", new_value)

    @property
    def transition_type(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":pul" "m:t" "type")

    @transition_type.setter
    def transition_type(self, new_value: str) -> None:
        """Selects between Fast or Smoothed slew rate (slope)."""
        if new_value.casefold() == "smo":
            new_value = "smoothed"
        if new_value.casefold() not in ("smoothed", "fast"):
            raise ValueError(f"Invalid Pulse Modulation Transition Mode: {new_value}")
        self._parent.issue(":pul" "m:t" "type", new_value)

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
        if new_value.casefold() not in ("normal", "inverted"):
            raise ValueError(f"Invalid Pulse Modulation Polarity: {new_value}")
        self._parent.issue(":pul" "m:polarity", new_value)

    @property
    def impedance(self) -> str:
        if self._parent.socket is None:
            return ""
        return self._parent.query(":pul" "m:impedance")

    @impedance.setter
    def impedance(self, new_value: str) -> None:
        """Sets the impedance for the external pulse trigger and pulse modulation input."""
        if new_value.casefold() not in ("g50", "g10k"):
            raise ValueError(f"Invalid Pulse Impedance: {new_value}")
        self._parent.issue(":pul" "m:impedance", new_value)

    @property
    def threshold(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(":pul" "m:threshold"))

    @threshold.setter
    def threshold(self, new_value: float) -> None:
        """Sets the threshold for the input signal at the Pulse Ext connector."""
        if not (0.0 <= new_value <= 2.0):
            raise ValueError(f"Invalid Pulse Modulation Threshold: {new_value}")
        self._parent.issue(":pul" "m:threshold", new_value)


class SMA100B(SCPIDevice):
    def __init__(self, ip: str | None, port: int, *, expected: bool = True) -> None:
        super().__init__(ip, port, expected=expected)
        self.am: Final[_AmplitudeModulation] = _AmplitudeModulation(self)
        self.lf_output: Final[_LFOutput] = _LFOutput(self)
        self.input: Final[_Input] = _Input(self)
        self.pulse_modulation: Final[_PulseModulation] = _PulseModulation(self)

    @property
    def power(self) -> float:
        if self.socket is None:
            return nan
        return float(self.query("power"))

    @power.setter
    def power(self, new_value: float) -> None:
        self.issue("power", new_value)

    @property
    def output(self) -> bool:
        if self.socket is None:
            return False
        return bool(int(self.query("output")))

    @output.setter
    def output(self, new_value: bool) -> None:
        self.issue("output", int(bool(new_value)))

    @property
    def frequency(self) -> float:
        if self.socket is None:
            return nan
        return float(self.query("frequency"))

    @frequency.setter
    def frequency(self, new_value: float) -> None:
        self.issue("frequency", new_value)


if __name__ == "__main__":
    s: SMA100B = SMA100B(None, 5025)
    print(s.idn)
