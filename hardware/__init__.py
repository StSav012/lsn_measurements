# coding: utf-8
from __future__ import annotations

from typing import Final

import numpy as np
from nidaqmx.constants import ChannelType, FillMode, UsageTypeAI
from nidaqmx.system.device import Device
from nidaqmx.system.physical_channel import PhysicalChannel
from nidaqmx.system.system import System
from nidaqmx.task import Task
from numpy.typing import NDArray

from .anapico import APUASYN20
from .rs import SMA100B
from .spike import Spike
from .triton import Triton

__all__ = [
    "APUASYN20",
    "SMA100B",
    "Spike",
    "Triton",
    "R",
    "R_SERIES",
    "VOLTAGE_GAIN",
    "DIVIDER",
    "DIVIDER_RESISTANCE",
    "system",
    "device_dac",
    "device_adc",
    "adc_sync",
    "adc_voltage",
    "adc_current",
    "dac_sync",
    "dac_current",
    "dac_aux",
    "dac_synth_pulse",
    "offsets",
]

R: float = 200.0e3
R_SERIES: float = 0.0e3
VOLTAGE_GAIN: float = 1000.0
DIVIDER: float = 1.058
DIVIDER_RESISTANCE: float = 5.65e3

system: Final[System] = System.local()

device_adc: Final[Device] = system.devices[0]
device_dac: Final[Device] = system.devices[1]

adc_voltage: Final[PhysicalChannel] = device_adc.ai_physical_chans[1]
adc_current: Final[PhysicalChannel] = device_adc.ai_physical_chans[0]
adc_sync: Final[PhysicalChannel] = device_adc.ai_physical_chans[4]
dac_current: Final[PhysicalChannel] = device_dac.ao_physical_chans[1]
dac_aux: Final[PhysicalChannel] = device_dac.ao_physical_chans[2]
dac_synth_pulse: Final[PhysicalChannel] = device_dac.ao_physical_chans[3]
dac_sync: Final[PhysicalChannel] = device_dac.ao_physical_chans[4]

offsets: dict[str, float] = {adc_current.name: 0.0, adc_voltage.name: 0.0}


if not hasattr(Task, "output_onboard_buffer_size"):
    import ctypes
    import nidaqmx
    from _ctypes import CFuncPtr
    from nidaqmx._library_interpreter import LibraryInterpreter

    def get_output_onboard_buffer_size(self) -> int:
        """
        int: Indicates in samples per channel the size of the onboard output buffer of the device.
        """
        val: ctypes.c_uint32 = ctypes.c_uint32()

        lib_importer = getattr(nidaqmx, "_lib").lib_importer
        c_func: CFuncPtr = lib_importer.windll.DAQmxGetBufOutputOnbrdBufSize
        if c_func.argtypes is None:
            with c_func.arglock:
                if c_func.argtypes is None:
                    c_func.argtypes = [
                        lib_importer.task_handle,
                        ctypes.POINTER(ctypes.c_uint),
                    ]

        error_code: int = c_func(self._handle, ctypes.byref(val))
        LibraryInterpreter().check_for_error(error_code)

        return val.value

    def set_output_onboard_buffer_size(self, buffer_size: int) -> None:
        """
        int: Specifies in samples per channel the size of the onboard output buffer of the device.
        """
        val: ctypes.c_uint32 = ctypes.c_uint32(buffer_size)

        lib_importer = getattr(nidaqmx, "_lib").lib_importer
        c_func: CFuncPtr = lib_importer.windll.DAQmxSetBufOutputOnbrdBufSize
        if c_func.argtypes is None:
            with c_func.arglock:
                if c_func.argtypes is None:
                    c_func.argtypes = [
                        lib_importer.task_handle,
                        ctypes.POINTER(ctypes.c_uint32),
                    ]

        error_code: int = c_func(self._handle, val)
        LibraryInterpreter().check_for_error(error_code)

    Task.output_onboard_buffer_size = property(
        fget=get_output_onboard_buffer_size,
        fset=set_output_onboard_buffer_size,
        fdel=lambda self: None,
        doc="int: Specifies in samples per channel the size of the onboard output buffer of the device.",
    )

if not hasattr(Task, "input_onboard_buffer_size"):
    import ctypes
    import nidaqmx
    from _ctypes import CFuncPtr
    from nidaqmx._library_interpreter import LibraryInterpreter

    def get_input_onboard_buffer_size(self) -> int:
        """
        int: Indicates in samples per channel the size of the onboard input buffer of the device.
        """
        val: ctypes.c_uint = ctypes.c_uint()

        lib_importer = getattr(nidaqmx, "_lib").lib_importer
        c_func: CFuncPtr = lib_importer.windll.DAQmxGetBufInputOnbrdBufSize
        if c_func.argtypes is None:
            with c_func.arglock:
                if c_func.argtypes is None:
                    c_func.argtypes = [
                        lib_importer.task_handle,
                        ctypes.POINTER(ctypes.c_uint32),
                    ]

        error_code: int = c_func(self._handle, ctypes.byref(val))
        LibraryInterpreter().check_for_error(error_code)

        return val.value

    Task.input_onboard_buffer_size = property(
        fget=get_input_onboard_buffer_size,
        fdel=lambda self: None,
        doc="int: Indicates in samples per channel the size of the onboard input buffer of the device.",
    )


# don't convert the samples read from NDArray into a list
# the following function is an almost exact copy of what is in the NI sources, except there is no `np.tolist` used
def _read_ai_faster(
    self: Task,
    number_of_samples_per_channel=nidaqmx.task.NUM_SAMPLES_UNSET,
    timeout: float = 10.0,
) -> np.float64 | NDArray[np.float64]:
    channels_to_read = self.in_stream.channels_to_read
    read_chan_type: ChannelType = channels_to_read.chan_type

    if read_chan_type != ChannelType.ANALOG_INPUT or any(
        chan.ai_meas_type == UsageTypeAI.POWER for chan in channels_to_read
    ):
        return self._read()  # use the NI function, backed up as `_read` prior to `_read_ai_faster` function use

    num_samples_not_set: bool = number_of_samples_per_channel is nidaqmx.task.NUM_SAMPLES_UNSET
    number_of_samples_per_channel: int = self._calculate_num_samps_per_chan(number_of_samples_per_channel)
    number_of_channels: int = len(channels_to_read.channel_names)

    # Determine the array shape and size to create
    if number_of_channels > 1:
        if not num_samples_not_set:
            array_shape = (number_of_channels, number_of_samples_per_channel)
        else:
            array_shape = number_of_channels
    else:
        array_shape = number_of_samples_per_channel

    # Analog Input Only
    data: NDArray[np.float64] = np.zeros(array_shape, dtype=np.float64)
    samples_read: int
    _, samples_read = self._interpreter.read_analog_f64(
        self._handle, number_of_samples_per_channel, timeout, FillMode.GROUP_BY_CHANNEL.value, data
    )

    if num_samples_not_set and array_shape == 1:
        return data[0]

    if samples_read != number_of_samples_per_channel:
        if number_of_channels > 1:
            return data[:, :samples_read]
        else:
            return data[:samples_read]

    return data


Task._read = Task.read
Task.read = _read_ai_faster
