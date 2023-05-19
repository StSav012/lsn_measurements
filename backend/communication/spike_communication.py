# coding: utf-8
from __future__ import annotations

from math import isnan, nan
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .scpi_device import SCPIDevice

__all__ = ['Spike']


class _Instrument:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: SCPIDevice = parent

    @property
    def select(self) -> str:
        if self._parent.socket is None:
            return ''
        return self._parent.query(':instrument:select')

    @select.setter
    def select(self, new_value: str) -> None:
        if not new_value:
            return
        self._parent.issue(':instrument:select', new_value)


class _Initiate:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: SCPIDevice = parent

    @property
    def continuous(self) -> bool:
        if self._parent.socket is None:
            return False
        return bool(int(self._parent.query(':init:continuous')))

    @continuous.setter
    def continuous(self, new_value: bool) -> None:
        self._parent.issue(':init:continuous', bool(new_value))


class _Bandwidth:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: SCPIDevice = parent

    def __float__(self) -> float:
        return self.resolution

    @property
    def resolution(self) -> float:
        if self._parent.socket is None:
            return nan
        if self._parent.query(':bandwidth:resolution:auto'):
            return nan
        return float(self._parent.query(':bandwidth:resolution'))

    @resolution.setter
    def resolution(self, new_value: float) -> None:
        if isnan(new_value):
            self._parent.issue(':bandwidth:resolution:auto', True)
        else:
            self._parent.issue(':bandwidth:resolution:auto', False)
            self._parent.issue(':bandwidth:resolution', new_value)

    @property
    def video(self) -> float:
        if self._parent.socket is None:
            return nan
        if self._parent.query(':bandwidth:video:auto'):
            return nan
        return float(self._parent.query(':bandwidth:video'))

    @video.setter
    def video(self, new_value: float) -> None:
        if isnan(new_value):
            self._parent.issue(':bandwidth:video:auto', True)
        else:
            self._parent.issue(':bandwidth:video:auto', False)
            self._parent.issue(':bandwidth:video', new_value)

    @property
    def shape(self) -> str:
        if self._parent.socket is None:
            return ''
        return self._parent.query(':bandwidth:shape')

    @shape.setter
    def shape(self, new_value: str) -> None:
        # FLATtop|NUTTall|GAUSsian
        if not new_value:
            return
        self._parent.issue(':bandwidth:shape', new_value)


class _Frequency:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: SCPIDevice = parent

    @property
    def center(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(':frequency:center'))

    @center.setter
    def center(self, new_value: float) -> None:
        if isnan(new_value):
            return
        self._parent.issue(':frequency:center', new_value)

    @property
    def span(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(':frequency:span'))

    @span.setter
    def span(self, new_value: float) -> None:
        if isnan(new_value):
            return
        self._parent.issue(':frequency:span', new_value)

    @property
    def start(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(':frequency:start'))

    @start.setter
    def start(self, new_value: float) -> None:
        if isnan(new_value):
            return
        self._parent.issue(':frequency:start', new_value)

    @property
    def stop(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(':frequency:stop'))

    @stop.setter
    def stop(self, new_value: float) -> None:
        if isnan(new_value):
            return
        self._parent.issue(':frequency:stop', new_value)


class _Power:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: SCPIDevice = parent

    def __float__(self) -> float:
        return self.reference_level

    @property
    def reference_level(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(':power:r''level'))

    @reference_level.setter
    def reference_level(self, new_value: float) -> None:
        if isnan(new_value):
            return
        self._parent.issue(':power:r''level', f'{new_value}dbm')

    @property
    def division(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(':power:p''division'))

    @division.setter
    def division(self, new_value: float) -> None:
        if isnan(new_value):
            return
        self._parent.issue(':power:p''division', new_value)


class _Detector:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: SCPIDevice = parent

    @property
    def function(self) -> str:
        if self._parent.socket is None:
            return ''
        return self._parent.query(':sweep:detector:function')

    @function.setter
    def function(self, new_value: str) -> None:
        if new_value.casefold() == 'aver':
            new_value = 'average'
        elif new_value.casefold() not in ('average', 'min''max', 'min', 'max'):
            raise ValueError(f'Invalid Sweep Detector Function: {new_value}')
        self._parent.issue(':sweep:detector:function', new_value)

    @property
    def units(self) -> str:
        if self._parent.socket is None:
            return ''
        return self._parent.query(':sweep:detector:units')

    @units.setter
    def units(self, new_value: str) -> None:
        if new_value.casefold() == 'pow':
            new_value = 'power'
        if new_value.casefold() == 'sam''pl':
            new_value = 'sample'
        if new_value.casefold() == 'volt':
            new_value = 'voltage'
        elif new_value.casefold() not in ('power', 'sample', 'voltage', 'log'):
            raise ValueError(f'Invalid Sweep Detector Units: {new_value}')
        self._parent.issue(':sweep:detector:units', new_value)


class _Sweep:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: SCPIDevice = parent
        self.detector: _Detector = _Detector(self._parent)

    @property
    def time(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(':sweep:time'))

    @time.setter
    def time(self, new_value: float) -> None:
        if isnan(new_value):
            return
        self._parent.issue(':sweep:time', new_value)


class _Average:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: SCPIDevice = parent

    @property
    def count(self) -> int:
        if self._parent.socket is None:
            return -1
        return int(self._parent.query(':trace:average:count'))

    @count.setter
    def count(self, new_value: int) -> None:
        if isnan(new_value):
            return
        self._parent.issue(':trace:average:count', int(new_value))

    @property
    def current(self) -> int:
        if self._parent.socket is None:
            return -1
        return int(self._parent.query(':trace:average:current'))


class _Trace:
    def __init__(self, parent: SCPIDevice) -> None:
        self._parent: SCPIDevice = parent
        self.average: _Average = _Average(self._parent)

    @property
    def select(self) -> int:
        if self._parent.socket is None:
            return -1
        return int(self._parent.query(':trace:select'))

    @select.setter
    def select(self, new_value: int) -> None:
        if isnan(new_value):
            return
        self._parent.issue(':trace:select', int(new_value))

    @property
    def type(self) -> str:
        if self._parent.socket is None:
            return ''
        return self._parent.query(':trace:type')

    @type.setter
    def type(self, new_value: str) -> None:
        if new_value.casefold() == 'wr''it':
            new_value = 'write'
        if new_value.casefold() == 'aver':
            new_value = 'average'
        if new_value.casefold() == 'max':
            new_value = 'max''hold'
        if new_value.casefold() == 'min':
            new_value = 'min''hold'
        elif new_value.casefold() not in ('off', 'write', 'average', 'max''hold', 'min''hold', 'min''max'):
            raise ValueError(f'Invalid Sweep Detector Units: {new_value}')
        self._parent.issue(':trace:type', new_value)

    @property
    def update(self) -> bool:
        if self._parent.socket is None:
            return False
        return bool(int(self._parent.query(':trace:update')))

    @update.setter
    def update(self, new_value: bool) -> None:
        self._parent.issue(':trace:update', bool(new_value))

    @property
    def display(self) -> bool:
        if self._parent.socket is None:
            return False
        return bool(int(self._parent.query(':trace:display')))

    @display.setter
    def display(self, new_value: bool) -> None:
        self._parent.issue(':trace:display', bool(new_value))

    @property
    def points(self) -> int:
        if self._parent.socket is None:
            return -1
        return int(self._parent.query(':trace:points'))

    @property
    def data(self) -> NDArray[np.float64]:
        if self._parent.socket is None:
            return np.empty(0)
        return np.fromiter(map(float, self._parent.query(':trace:data').split(',')), dtype=np.float64)

    @property
    def start_frequency(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(':trace:x''start'))

    @property
    def frequency_step(self) -> float:
        if self._parent.socket is None:
            return nan
        return float(self._parent.query(':trace:x''increment'))


class Spike(SCPIDevice):
    def __init__(self, ip: str | None, port: int) -> None:
        super().__init__(ip, port, terminator=b'\n')

        self.instrument: _Instrument = _Instrument(self)
        self.initiate: _Initiate = _Initiate(self)
        self.bandwidth: _Bandwidth = _Bandwidth(self)
        self.frequency: _Frequency = _Frequency(self)
        self.power: _Power = _Power(self)
        self.sweep: _Sweep = _Sweep(self)
        self.trace: _Trace = _Trace(self)

    def init(self) -> None:
        self.communicate(':init')


if __name__ == '__main__':
    s: Spike = Spike('localhost', 5025)
    if s.socket is None:
        print('no socket')
        exit(0)
    print(f'{s.idn = }')
    # Set the measurement mode to sweep
    s.instrument.select = 'SA'
    # Disable continuous measurement operation
    s.initiate.continuous = False

    # Configure a 20MHz span sweep at 1GHz
    # Set the RBW/VBW to auto
    s.bandwidth.resolution = nan
    s.bandwidth.video = nan
    s.bandwidth.shape = 'flattop'

    # Center/span
    s.frequency.span = 2e6
    s.frequency.center = 1e9
    # Reference level/Div
    s.power.reference_level = -20  # dbm
    s.power.division = 10
    # Peak detector
    s.sweep.detector.function = 'minmax'
    s.sweep.detector.units = 'power'

    # Configure the trace. Ensures trace 1 is active and enabled for clear-and-write.
    # These commands are not required to be sent everytime, this is for illustrative purposes only.
    # Select trace 1
    s.trace.select = 1
    # Set clear and write mode
    s.trace.type = 'write'
    # Set update state to on
    s.trace.update = True
    # Set un-hidden
    s.trace.display = True

    for i in range(3):
        # Trigger a sweep, and wait for it to complete
        s.init()
        s.opc()

        # Sweep data is returned as comma separated values
        points: NDArray[np.float64] = s.trace.data

        # Query information needed to know what frequency each point in the sweep refers to
        start_freq: float = s.trace.start_frequency
        bin_size: float = s.trace.frequency_step

        # Find the peak point in the sweep
        peak_idx: int = cast(int, np.argmax(points))
        peak_val: float = points[peak_idx]
        peak_freq: float = start_freq + peak_idx * bin_size

        # Print out peak information
        print(f"Peak Freq {peak_freq / 1.0e6} MHz, Peak Ampl {peak_val} dBm")
